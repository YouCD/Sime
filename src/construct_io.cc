#include "construct.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace sime {
namespace {

struct DiskNode {
    TokenID id = 0;
    float pro = 0.0f;
    std::uint32_t down = 0;
    float bow = 0.0f;
    std::uint32_t bon = 0;
    std::uint32_t boe = 0;
};

struct DiskLeave {
    TokenID id = 0;
    float pro = 0.0f;
    std::uint32_t bon = 0;
    std::uint32_t boe = 0;
};

}  // namespace

void Constructor::GetBack(int length, const TokenID* seq,
                          std::uint32_t& boe, std::uint32_t& bon) const {
    boe = 0;
    bon = 0;
    if (length <= 1 || seq == nullptr) return;
    const TokenID* hw = seq;
    int n = length;
    while (n > 1) {
        --n;
        ++hw;
        // Find this suffix in the model
        const Node* cur = &node_levels_[0][0];
        bool found = true;
        int result_idx = 0;
        for (int i = 0; i < n; ++i) {
            const void* down = FindDown(i, cur, hw[i]);
            if (!down) { found = false; break; }
            if (i + 1 == opts_.num) {
                // leaf level — can't have children
                found = false;
                break;
            }
            cur = static_cast<const Node*>(down);
            result_idx = static_cast<int>(cur - node_levels_[i + 1].data());
        }
        if (found && n < opts_.num) {
            // Check if this node has children
            const Node& node = node_levels_[n][result_idx];
            const Node& next_node = node_levels_[n][result_idx + 1];
            if (next_node.down > node.down) {
                boe = static_cast<std::uint32_t>(n);
                bon = static_cast<std::uint32_t>(result_idx);
                return;
            }
        }
    }
}

void Constructor::Write(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open output file");
    }
    int order = opts_.num;
    out.write(reinterpret_cast<const char*>(&order), sizeof(order));
    for (int lvl = 0; lvl <= order; ++lvl) {
        std::uint32_t size = 0;
        if (lvl == order) {
            size = static_cast<std::uint32_t>(leaves_.size());
        } else {
            size = static_cast<std::uint32_t>(node_levels_[lvl].size());
        }
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }

    // Build parent indices for token sequence recovery
    std::vector<std::vector<int>> node_ups(static_cast<std::size_t>(order));
    for (int lvl = 1; lvl < order; ++lvl) {
        auto& ups = node_ups[static_cast<std::size_t>(lvl)];
        ups.resize(node_levels_[lvl].size());
        std::size_t up = 0;
        for (std::size_t idx = 0; idx + 1 < node_levels_[lvl].size(); ++idx) {
            while (up + 1 < node_levels_[lvl - 1].size() &&
                   node_levels_[lvl - 1][up + 1].down <= idx) {
                ++up;
            }
            ups[idx] = static_cast<int>(up);
        }
    }

    std::vector<int> leave_ups(leaves_.size());
    if (order > 0) {
        std::size_t up = 0;
        for (std::size_t idx = 0; idx + 1 < leaves_.size(); ++idx) {
            while (up + 1 < node_levels_[order - 1].size() &&
                   node_levels_[order - 1][up + 1].down <= idx) {
                ++up;
            }
            leave_ups[idx] = static_cast<int>(up);
        }
    }

    // Helper to get token sequence for a node
    auto get_tokens = [&](int level, int index, std::vector<TokenID>& tokens) {
        tokens.assign(static_cast<std::size_t>(level), TokenID{0});
        if (level == 0) return;
        if (level == order) {
            tokens.back() = leaves_[static_cast<std::size_t>(index)].id;
            int up_idx = leave_ups[static_cast<std::size_t>(index)];
            int cur_level = order - 1;
            for (int pos = level - 1; pos > 0; --pos) {
                tokens[static_cast<std::size_t>(pos - 1)] =
                    node_levels_[cur_level][static_cast<std::size_t>(up_idx)].id;
                if (pos - 1 == 0) break;
                up_idx = node_ups[static_cast<std::size_t>(cur_level)]
                             [static_cast<std::size_t>(up_idx)];
                --cur_level;
            }
            return;
        }
        int cur_idx = index;
        int cur_level = level;
        for (int pos = level; pos > 0; --pos) {
            tokens[static_cast<std::size_t>(pos - 1)] =
                node_levels_[cur_level][static_cast<std::size_t>(cur_idx)].id;
            if (pos - 1 == 0) break;
            cur_idx = node_ups[static_cast<std::size_t>(cur_level)]
                         [static_cast<std::size_t>(cur_idx)];
            --cur_level;
        }
    };

    // Write nodes with bon/boe
    std::vector<TokenID> history;
    for (int lvl = 0; lvl < order; ++lvl) {
        const auto& level = node_levels_[lvl];
        for (std::size_t i = 0; i < level.size(); ++i) {
            const auto& node = level[i];
            DiskNode raw;
            raw.id = node.id;
            raw.pro = static_cast<float>(node.pro);
            raw.down = node.down;
            raw.bow = static_cast<float>(node.bow);
            raw.bon = 0;
            raw.boe = 0;
            if (lvl > 0 && i + 1 < level.size()) {
                get_tokens(lvl, static_cast<int>(i), history);
                GetBack(lvl, history.data(), raw.boe, raw.bon);
            }
            out.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
        }
    }

    // Write leaves with bon/boe
    for (std::size_t i = 0; i + 1 < leaves_.size(); ++i) {
        DiskLeave raw;
        raw.id = leaves_[i].id;
        raw.pro = static_cast<float>(leaves_[i].pro);
        get_tokens(order, static_cast<int>(i), history);
        GetBack(order, history.data(), raw.boe, raw.bon);
        out.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
    }
    // Write sentinel leaf
    if (!leaves_.empty()) {
        DiskLeave raw;
        raw.id = leaves_.back().id;
        raw.pro = static_cast<float>(leaves_.back().pro);
        raw.bon = 0;
        raw.boe = 0;
        out.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
    }
}


namespace {

std::filesystem::path WithSuffix(const std::filesystem::path& base,
                                 const std::string& suffix) {
    return base.string() + suffix;
}

// Reads a sorted binary n-gram file with records (N x TokenID, Cnt).
// Invokes cb(token_array, cnt) for each record.
template <std::size_t N, typename Cb>
void ForEachNgram(const std::filesystem::path& path, Cb cb) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open: " + path.string());
    }
    std::array<TokenID, N> ids{};
    std::uint32_t cnt = 0;
    while (in.read(reinterpret_cast<char*>(ids.data()),
                   static_cast<std::streamsize>(N * sizeof(TokenID)))) {
        if (!in.read(reinterpret_cast<char*>(&cnt),
                     static_cast<std::streamsize>(sizeof(cnt)))) {
            break;
        }
        cb(ids, cnt);
    }
}

}  // namespace

void RunConstruct(ConstructOptions options) {
    auto input_base = options.input;
    auto output_path = options.output;
    int num = options.num;
    auto prune_reserves = std::move(options.prune_reserves);

    Constructor builder(std::move(options));

    if (num >= 1) {
        auto p = WithSuffix(input_base, ".1gram");
        std::cerr << "loading " << p.string() << "\n";
        ForEachNgram<1>(p,
            [&](const std::array<TokenID, 1>& ids, std::uint32_t cnt) {
                builder.InsertUnigram(ids[0], cnt);
            });
    }
    if (num >= 2) {
        auto p = WithSuffix(input_base, ".2gram");
        std::cerr << "loading " << p.string() << "\n";
        ForEachNgram<2>(p,
            [&](const std::array<TokenID, 2>& ids, std::uint32_t cnt) {
                builder.InsertBigram(ids[0], ids[1], cnt);
            });
    }
    if (num >= 3) {
        auto p = WithSuffix(input_base, ".3gram");
        std::cerr << "loading " << p.string() << "\n";
        ForEachNgram<3>(p,
            [&](const std::array<TokenID, 3>& ids, std::uint32_t cnt) {
                builder.InsertTrigram(ids[0], ids[1], ids[2], cnt);
            });
    }

    builder.Finalize();
    if (!prune_reserves.empty()) {
        builder.Prune(prune_reserves);
    }

    builder.Write(output_path);
}

} // namespace sime
