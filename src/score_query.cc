#include "score.h"

#include <algorithm>
#include <unordered_set>

namespace sime {

void Scorer::Back(Pos& pos) const {
    if (pos.level <= 0) return;

    if (pos.level >= static_cast<std::uint32_t>(num_)) {
        if (pos.level != static_cast<std::uint32_t>(num_)) {
            pos = Pos{};
            return;
        }
        TokenID w = leaf_tokens_[pos.index];

        if (num_ >= 3) {
            // Find parent at level (num_-1) to get token v.
            const auto& parent_lv = node_levels_[static_cast<std::size_t>(num_ - 1)];
            std::size_t lo = 0;
            std::size_t hi = (parent_lv.size > 1) ? parent_lv.size - 1 : 0;
            while (lo + 1 < hi) {
                std::size_t mid = lo + (hi - lo) / 2;
                if (parent_lv.down[mid] <= pos.index) lo = mid;
                else hi = mid;
            }
            TokenID v = parent_lv.tokens[lo];
            auto v_idx = FindNode(1, v);
            if (v_idx != SIZE_MAX && v_idx + 1 < node_levels_[1].size) {
                auto begin = node_levels_[1].down[v_idx];
                auto end_d = node_levels_[1].down[v_idx + 1];
                auto vw_idx = GetNode(2, begin, end_d, w);
                if (vw_idx != end_d) {
                    const auto& l2 = node_levels_[2];
                    if (vw_idx + 1 < l2.size &&
                        l2.down[vw_idx + 1] > l2.down[vw_idx]) {
                        pos.level = 2;
                        pos.index = static_cast<std::uint32_t>(vw_idx);
                        return;
                    }
                }
            }
        }

        auto w_idx = FindNode(1, w);
        if (w_idx != SIZE_MAX && w_idx + 1 < node_levels_[1].size &&
            node_levels_[1].down[w_idx + 1] > node_levels_[1].down[w_idx]) {
            pos.level = 1;
            pos.index = static_cast<std::uint32_t>(w_idx);
            return;
        }

        pos = Pos{};
        return;
    }

    // Node level: stay put if has children.
    const auto& lv = node_levels_[pos.level];
    if (pos.index + 1 >= lv.size) {
        pos = Pos{};
        return;
    }
    if (lv.down[pos.index + 1] > lv.down[pos.index]) {
        return;
    }

    if (pos.level == 1) {
        pos = Pos{};
        return;
    }
    TokenID v = lv.tokens[pos.index];
    auto idx = FindNode(static_cast<int>(pos.level - 1), v);
    if (idx != SIZE_MAX) {
        pos.level = pos.level - 1;
        pos.index = static_cast<std::uint32_t>(idx);
        Back(pos);
    } else {
        pos = Pos{};
    }
}

float_t Scorer::ScoreMove(Pos s, TokenID w, Pos& r) const {
    std::uint32_t level = s.level;
    std::uint32_t index = s.index;

    float_t cost = 0.0;
    if (w == NotToken) {
        r = Pos{};
        return cost;
    }

    while (level < static_cast<std::uint32_t>(num_)) {
        const auto& lv = node_levels_[level];
        std::size_t node_index = (level == 0) ? 0 : index;
        if (node_index + 1 >= lv.size) {
            break;
        }
        auto begin = lv.down[node_index];
        auto end_d = lv.down[node_index + 1];
        if (level == static_cast<std::uint32_t>(num_ - 1)) {
            auto down_idx = GetLeave(begin, end_d, w);
            if (down_idx != end_d) {
                r.level = static_cast<std::uint32_t>(num_);
                r.index = static_cast<std::uint32_t>(down_idx);
                return cost + LeafPro(down_idx);
            }
        } else {
            auto down_idx = GetNode(static_cast<int>(level + 1), begin, end_d, w);
            if (down_idx != end_d) {
                r.level = level + 1;
                r.index = static_cast<std::uint32_t>(down_idx);
                return cost + NodePro(static_cast<int>(level + 1), down_idx);
            }
        }

        cost += NodeBow(static_cast<int>(level), node_index);
        if (level == 0) {
            break;
        }
        if (level == 1) {
            level = 0;
            index = 0;
        } else {
            TokenID v = lv.tokens[node_index];
            auto idx = FindNode(static_cast<int>(level - 1), v);
            if (idx != SIZE_MAX) {
                level = level - 1;
                index = static_cast<std::uint32_t>(idx);
            } else {
                level = 0;
                index = 0;
            }
        }
    }

    r = Pos{};
    return cost + NodePro(0, 0);
}

float_t Scorer::UnknownPenalty() const {
    if (node_levels_.empty() || node_levels_[0].size == 0) {
        constexpr float_t DefaultUnknownPenalty = -20.0;
        return DefaultUnknownPenalty;
    }
    return NodePro(0, 0);
}

std::vector<std::pair<TokenID, float_t>> Scorer::NextTokens(
    Pos& context, std::size_t num) const {
    std::vector<std::pair<TokenID, float_t>> result;
    if (num_ < 2) return result;

    Pos ctx = context;
    while (true) {
        if (ctx.level == 0) return result;

        if (ctx.level >= static_cast<std::uint32_t>(num_)) {
            Back(ctx);
            if (ctx.level >= static_cast<std::uint32_t>(num_)) return result;
            continue;
        }

        const auto& lv = node_levels_[ctx.level];
        std::size_t node_index = ctx.index;
        if (node_index + 1 >= lv.size) return result;
        auto begin = lv.down[node_index];
        auto end_d = lv.down[node_index + 1];
        if (begin < end_d) break;

        Pos backed = ctx;
        Back(backed);
        if (backed.level == ctx.level && backed.index == ctx.index) {
            return result;
        }
        ctx = backed;
    }

    const float zero_pro = NodePro(0, 0);

    auto collect = [&](std::uint32_t clevel, std::uint32_t cindex,
                       std::vector<std::pair<TokenID, float_t>>& out) {
        if (clevel >= static_cast<std::uint32_t>(num_)) return;
        const auto& lv = node_levels_[clevel];
        std::size_t ni = cindex;
        if (ni + 1 >= lv.size) return;
        auto begin = lv.down[ni];
        auto end_d = lv.down[ni + 1];

        if (clevel == static_cast<std::uint32_t>(num_ - 1)) {
            for (auto i = begin; i < end_d && i < leave_size_; ++i) {
                float pro = LeafPro(i);
                if (pro >= zero_pro) continue;
                out.emplace_back(leaf_tokens_[i], pro);
            }
        } else {
            const auto& children = node_levels_[clevel + 1];
            for (auto i = begin; i < end_d && i < children.size; ++i) {
                float pro = NodePro(static_cast<int>(clevel + 1), i);
                if (pro >= zero_pro) continue;
                out.emplace_back(children.tokens[i], pro);
            }
        }
    };

    collect(ctx.level, ctx.index, result);
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (result.size() < num && ctx.level > 1) {
        std::unordered_set<TokenID> seen;
        for (const auto& [tid, _] : result) seen.insert(tid);

        // Walk down to lower-order contexts to fill remaining slots.
        Pos backed = ctx;
        while (result.size() < num && backed.level > 1) {
            // Force descent: find the current node's token at level-1.
            const auto& lv = node_levels_[backed.level];
            TokenID v = lv.tokens[backed.index];
            if (backed.level == 2) {
                // bigram → unigram: find v at level 1
                auto idx = FindNode(1, v);
                if (idx == SIZE_MAX) break;
                backed.level = 1;
                backed.index = static_cast<std::uint32_t>(idx);
            } else {
                backed.level = backed.level - 1;
                auto idx = FindNode(static_cast<int>(backed.level), v);
                if (idx == SIZE_MAX) break;
                backed.index = static_cast<std::uint32_t>(idx);
            }

            std::vector<std::pair<TokenID, float_t>> backoff;
            collect(backed.level, backed.index, backoff);
            std::sort(backoff.begin(), backoff.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
            for (const auto& entry : backoff) {
                if (result.size() >= num) break;
                if (seen.insert(entry.first).second) {
                    result.push_back(entry);
                }
            }
        }
    }

    context = ctx;
    if (result.size() > num) result.resize(num);
    return result;
}

std::vector<Scorer::NGram> Scorer::DumpLevel(int level) const {
    std::vector<NGram> results;
    if (level < 1 || level > num_) return results;

    if (level == 1) {
        if (num_ < 2 || node_levels_.size() < 2) return results;
        const auto& root = node_levels_[0];
        const auto& unigrams = node_levels_[1];
        auto begin = root.down[0];
        auto end_d = (root.size > 1) ? root.down[1]
                                     : static_cast<std::uint32_t>(unigrams.size);
        for (auto i = begin; i < end_d && i < unigrams.size; ++i) {
            NGram ng;
            ng.tokens.push_back(unigrams.tokens[i]);
            ng.pro = NodePro(1, i);
            results.push_back(std::move(ng));
        }
    } else if (level == 2) {
        if (num_ < 2 || node_levels_.size() < 2) return results;
        const auto& unigrams = node_levels_[1];
        if (num_ == 2) {
            for (std::size_t i = 0; i + 1 < unigrams.size; ++i) {
                auto begin = unigrams.down[i];
                auto end_d = unigrams.down[i + 1];
                for (auto j = begin; j < end_d && j < leave_size_; ++j) {
                    NGram ng;
                    ng.tokens.push_back(unigrams.tokens[i]);
                    ng.tokens.push_back(leaf_tokens_[j]);
                    ng.pro = LeafPro(j);
                    results.push_back(std::move(ng));
                }
            }
        } else {
            const auto& bigrams = node_levels_[2];
            for (std::size_t i = 0; i + 1 < unigrams.size; ++i) {
                auto begin = unigrams.down[i];
                auto end_d = unigrams.down[i + 1];
                for (auto j = begin; j < end_d && j + 1 < bigrams.size; ++j) {
                    NGram ng;
                    ng.tokens.push_back(unigrams.tokens[i]);
                    ng.tokens.push_back(bigrams.tokens[j]);
                    ng.pro = NodePro(2, j);
                    results.push_back(std::move(ng));
                }
            }
        }
    } else if (level == 3 && num_ >= 3) {
        const auto& unigrams = node_levels_[1];
        const auto& bigrams = node_levels_[2];
        for (std::size_t i = 0; i + 1 < unigrams.size; ++i) {
            auto bi_begin = unigrams.down[i];
            auto bi_end = unigrams.down[i + 1];
            for (auto j = bi_begin; j < bi_end && j + 1 < bigrams.size; ++j) {
                auto leaf_begin = bigrams.down[j];
                auto leaf_end = bigrams.down[j + 1];
                for (auto k = leaf_begin; k < leaf_end && k < leave_size_; ++k) {
                    NGram ng;
                    ng.tokens.push_back(unigrams.tokens[i]);
                    ng.tokens.push_back(bigrams.tokens[j]);
                    ng.tokens.push_back(leaf_tokens_[k]);
                    ng.pro = LeafPro(k);
                    results.push_back(std::move(ng));
                }
            }
        }
    }

    return results;
}

std::size_t Scorer::GetNode(int level,
                            std::size_t begin,
                            std::size_t end,
                            TokenID w) const {
    // O(1) for level-1 full-range lookups (unigram context).
    if (level == 1 && !l1_index_.empty()) {
        if (static_cast<std::size_t>(w) >= l1_index_.size()) return end;
        auto idx = l1_index_[static_cast<std::size_t>(w)];
        if (idx == UINT32_MAX) return end;
        auto i = static_cast<std::size_t>(idx);
        if (i >= begin && i < end) return i;
        return end;
    }
    const auto* tokens = node_levels_[static_cast<std::size_t>(level)].tokens;
    const auto* first = tokens + begin;
    const auto* last = tokens + end;
    auto it = std::lower_bound(first, last, w);
    if (it == last || *it != w) {
        return end;
    }
    return static_cast<std::size_t>(it - tokens);
}

std::size_t Scorer::GetLeave(std::size_t begin,
                             std::size_t end,
                             TokenID w) const {
    const auto* first = leaf_tokens_ + begin;
    const auto* last = leaf_tokens_ + end;
    auto it = std::lower_bound(first, last, w);
    if (it == last || *it != w) {
        return end;
    }
    return static_cast<std::size_t>(it - leaf_tokens_);
}

} // namespace sime