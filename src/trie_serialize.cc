#include "trie.h"
#include "trie_detail.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_set>

namespace trie {
// --- Serialize / Deserialize ---

void DoubleArray::Serialize(std::vector<char>& buffer) const {
    auto sz = static_cast<uint32_t>(size_);
    std::size_t offset = buffer.size();
    buffer.resize(offset + sizeof(sz) + sz * sizeof(ArrayUnit));
    std::memcpy(buffer.data() + offset, &sz, sizeof(sz));
    offset += sizeof(sz);
    if (sz > 0) {
        std::memcpy(buffer.data() + offset, array_, sz * sizeof(ArrayUnit));
    }
}

bool DoubleArray::Deserialize(const char* data, std::size_t size) {
    if (size < sizeof(uint32_t)) return false;
    uint32_t sz = 0;
    std::memcpy(&sz, data, sizeof(sz));
    if (size < sizeof(sz) + sz * sizeof(ArrayUnit)) return false;

    size_ = sz;
    owned_ = std::make_unique<ArrayUnit[]>(sz);
    if (sz > 0) {
        std::memcpy(owned_.get(), data + sizeof(sz), sz * sizeof(ArrayUnit));
    }
    array_ = owned_.get();
    alphabet_ = ScanAlphabet(array_, size_);
    return true;
}

bool DoubleArray::MmapAttach(const char* data, std::size_t size,
                             std::size_t* consumed) {
    if (size < sizeof(uint32_t)) return false;
    uint32_t sz = 0;
    std::memcpy(&sz, data, sizeof(sz));
    std::size_t need = sizeof(sz) + sz * sizeof(ArrayUnit);
    if (size < need) return false;
    // Zero-copy: array_ points into mmap'd memory.
    auto array_addr = reinterpret_cast<std::uintptr_t>(data + sizeof(sz));
    if (array_addr % alignof(ArrayUnit) != 0) return false;

    size_ = sz;
    owned_.reset();
    array_ = reinterpret_cast<const ArrayUnit*>(data + sizeof(sz));
    alphabet_ = ScanAlphabet(array_, size_);
    if (consumed) *consumed = need;
    return true;
}

// --- Pinyin-aware matching ---

std::size_t DoubleArray::TryChild(std::size_t pos, uint8_t ch) const {
    if (pos >= size_) return SIZE_MAX;
    std::size_t child = pos ^ array_[pos].index ^ ch;
    if (child >= size_ || child == pos) return SIZE_MAX;
    if (array_[child].label != ch || array_[child].parent != pos)
        return SIZE_MAX;
    return child;
}

void DoubleArray::FindSepDescendants(std::size_t pos,
                                     std::vector<std::size_t>& out,
                                     int max_depth) const {
    constexpr std::size_t MaxSeps = 256;
    if (max_depth <= 0 || pos >= size_ || out.size() >= MaxSeps) return;
    uint32_t base = array_[pos].index;
    for (int ch = 1; ch <= 255; ++ch) {
        if (out.size() >= MaxSeps) return;
        std::size_t child = pos ^ base ^ static_cast<unsigned>(ch);
        if (child >= size_ || child == pos) continue;
        if (array_[child].label != ch || array_[child].parent != pos)
            continue;
        if (ch == '\'') {
            out.push_back(child);
        } else {
            FindSepDescendants(child, out, max_depth - 1);
        }
    }
}

void DoubleArray::ClearSepCache() const {
    for (auto& v : sep_cache_) {
        std::vector<std::size_t>().swap(v);  // free inner buffers
    }
    std::fill(sep_cache_computed_.begin(),
              sep_cache_computed_.end(), false);
}

void DoubleArray::ResetSepCache() const {
    std::vector<std::vector<std::size_t>>().swap(sep_cache_);
    std::vector<bool>().swap(sep_cache_computed_);
}

const std::vector<std::size_t>& DoubleArray::GetSepDescendants(
    std::size_t pos) const {
    if (sep_cache_computed_.size() <= pos) {
        sep_cache_.resize(size_);
        sep_cache_computed_.resize(size_, false);
    }
    if (!sep_cache_computed_[pos]) {
        FindSepDescendants(pos, sep_cache_[pos], 6);
        sep_cache_computed_[pos] = true;
    }
    return sep_cache_[pos];
}

} // namespace trie
