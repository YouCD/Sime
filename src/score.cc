#include "score.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

namespace sime {

namespace {

constexpr std::uint32_t CompactMagic = 0x51434D53;  // "SMCQ"

// Align pointer forward to 4-byte boundary.
const char* Align4(const char* p) {
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    auto rem = addr % 4;
    return (rem == 0) ? p : p + (4 - rem);
}

} // namespace

bool Scorer::Load(const std::filesystem::path& path) {
    Reset();

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    mmap_len_ = static_cast<std::size_t>(st.st_size);
    if (mmap_len_ == 0) { close(fd); return false; }

    mmap_addr_ = mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mmap_addr_ == MAP_FAILED) {
        mmap_addr_ = nullptr;
        mmap_len_ = 0;
        return false;
    }

    const char* p = static_cast<const char*>(mmap_addr_);
    const char* end = p + mmap_len_;

    // Check magic.
    if (p + sizeof(std::uint32_t) > end) { Reset(); return false; }
    std::uint32_t magic = 0;
    std::memcpy(&magic, p, sizeof(magic));
    if (magic != CompactMagic) { Reset(); return false; }
    p += sizeof(std::uint32_t);

    // Read header: num.
    if (p + sizeof(int) > end) { Reset(); return false; }
    std::memcpy(&num_, p, sizeof(int));
    p += sizeof(int);
    if (num_ <= 0) { Reset(); return false; }

    // Read sizes.
    std::size_t sizes_bytes = static_cast<std::size_t>(num_ + 1) * sizeof(int);
    if (p + sizes_bytes > end) { Reset(); return false; }
    sizes_.assign(num_ + 1, 0);
    std::memcpy(sizes_.data(), p, sizes_bytes);
    p += sizes_bytes;

    // Read quantization tables: per-level (pro16 + bow8), then leaf pro16.
    constexpr std::size_t ProTableBytes = 65536 * sizeof(float);
    constexpr std::size_t BowTableBytes = 256 * sizeof(float);
    qt_pro_.resize(static_cast<std::size_t>(num_));
    qt_bow_.resize(static_cast<std::size_t>(num_));
    for (int lvl = 0; lvl < num_; ++lvl) {
        if (p + ProTableBytes + BowTableBytes > end) { Reset(); return false; }
        qt_pro_[static_cast<std::size_t>(lvl)].resize(65536);
        std::memcpy(qt_pro_[static_cast<std::size_t>(lvl)].data(), p, ProTableBytes);
        p += ProTableBytes;
        qt_bow_[static_cast<std::size_t>(lvl)].resize(256);
        std::memcpy(qt_bow_[static_cast<std::size_t>(lvl)].data(), p, BowTableBytes);
        p += BowTableBytes;
    }
    if (p + ProTableBytes > end) { Reset(); return false; }
    qt_leaf_pro_.resize(65536);
    std::memcpy(qt_leaf_pro_.data(), p, ProTableBytes);
    p += ProTableBytes;

    // Map SoA node levels: tokens | pro_q (pad4) | bow_q (pad4) | down.
    node_levels_.resize(static_cast<std::size_t>(num_));
    for (int lvl = 0; lvl < num_; ++lvl) {
        auto size = static_cast<std::size_t>(sizes_[static_cast<std::size_t>(lvl)]);
        if (size == 0) { Reset(); return false; }

        std::size_t tok_bytes = size * sizeof(TokenID);
        std::size_t pro_bytes = size * sizeof(std::uint16_t);
        std::size_t bow_bytes = size;  // uint8_t per entry
        std::size_t down_bytes = size * sizeof(std::uint32_t);

        auto& lv = node_levels_[static_cast<std::size_t>(lvl)];
        lv.tokens = reinterpret_cast<const TokenID*>(p);
        p += tok_bytes;
        lv.pro_q = reinterpret_cast<const std::uint16_t*>(p);
        p += pro_bytes;
        p = Align4(p);
        lv.bow_q = reinterpret_cast<const std::uint8_t*>(p);
        p += bow_bytes;
        p = Align4(p);
        lv.down = reinterpret_cast<const std::uint32_t*>(p);
        p += down_bytes;
        lv.size = size;
    }

    BuildL1Index();

    // Map SoA leaves: tokens | pro_q16.
    auto leave_sz = static_cast<std::size_t>(sizes_.back());
    if (leave_sz == 0) { Reset(); return false; }
    std::size_t leaf_tok_bytes = leave_sz * sizeof(TokenID);
    std::size_t leaf_pro_bytes = leave_sz * sizeof(std::uint16_t);
    if (p + leaf_tok_bytes + leaf_pro_bytes > end) { Reset(); return false; }
    leaf_tokens_ = reinterpret_cast<const TokenID*>(p);
    p += leaf_tok_bytes;
    leaf_pro_q_ = reinterpret_cast<const std::uint16_t*>(p);
    leave_size_ = leave_sz;

    return true;
}

void Scorer::Reset() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_len_);
    }
    mmap_addr_ = nullptr;
    mmap_len_ = 0;
    num_ = 0;
    sizes_.clear();
    node_levels_.clear();
    leaf_tokens_ = nullptr;
    leaf_pro_q_ = nullptr;
    leave_size_ = 0;
    qt_pro_.clear();
    qt_bow_.clear();
    qt_leaf_pro_.clear();
    l1_index_.clear();
}

void Scorer::BuildL1Index() {
    l1_index_.clear();
    if (num_ < 2) return;
    const auto& lv = node_levels_[1];
    if (lv.size <= 1) return;
    // Find max token ID to size the array.
    TokenID max_token = 0;
    for (std::size_t i = 0; i < lv.size - 1; ++i) {
        if (lv.tokens[i] > max_token) max_token = lv.tokens[i];
    }
    l1_index_.assign(static_cast<std::size_t>(max_token) + 1, UINT32_MAX);
    for (std::size_t i = 0; i < lv.size - 1; ++i) {
        l1_index_[lv.tokens[i]] = static_cast<std::uint32_t>(i);
    }
}

Scorer::Pos Scorer::StartPos() const {
    return Pos{};
}

std::size_t Scorer::FindNode(int level, TokenID token) const {
    if (level < 1 || level >= num_) return SIZE_MAX;
    // O(1) lookup for level 1 (unigram → bigram).
    if (level == 1 && !l1_index_.empty()) {
        if (static_cast<std::size_t>(token) >= l1_index_.size())
            return SIZE_MAX;
        auto idx = l1_index_[static_cast<std::size_t>(token)];
        return (idx == UINT32_MAX) ? SIZE_MAX : static_cast<std::size_t>(idx);
    }
    const auto& lv = node_levels_[static_cast<std::size_t>(level)];
    std::size_t lo = 0;
    std::size_t hi = (lv.size > 0) ? lv.size - 1 : 0;
    while (lo < hi) {
        std::size_t mid = lo + (hi - lo) / 2;
        if (lv.tokens[mid] < token) lo = mid + 1;
        else hi = mid;
    }
    if (lo < lv.size - 1 && lv.tokens[lo] == token) return lo;
    return SIZE_MAX;
}

} // namespace sime