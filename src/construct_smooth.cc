#include "construct.h"
#include "construct_detail.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace sime {
void Constructor::PruneLevel(int level) {
    if (prune_cutoffs_.empty() || prune_cutoffs_[level] <= 0) {
        return;
    }
    int n = prune_sizes_[level] - 1;
    if (n <= 0) {
        return;
    }
    std::vector<NodeScore> candidates;
    candidates.reserve(static_cast<std::size_t>(n));
    std::vector<int> indices(opts_.num + 1, 0);
    std::vector<TokenID> words(opts_.num + 2, 0);

    for (int idx = 0; idx < n; ++idx) {
        indices[level] = idx;
        if (level == opts_.num) {
            words[level] = leaves_[idx].id;
        } else {
            words[level] = node_levels_[level][idx].id;
        }
        for (int j = level - 1; j >= 0; --j) {
            int up_idx = indices[j];
            const Node* up = &node_levels_[j][up_idx];
            while ((up + 1)->down <= static_cast<std::uint32_t>(indices[j + 1])) {
                ++up;
                ++indices[j];
            }
            words[j] = up->id;
        }
        bool has_down = false;
        if (level != opts_.num) {
            const Node& node = node_levels_[level][idx];
            has_down = ((node_levels_[level][idx + 1].down - node.down) > 0);
        }
        // Unprunable nodes (those with surviving children) sort to the tail
        // via a +inf sentinel so the natural ascending order places them
        // after all CalcScore-ranked candidates.
        float_t dist = has_down
            ? std::numeric_limits<float_t>::max()
            : CalcScore(level, indices, words);
        candidates.push_back(
            NodeScore{dist, static_cast<std::uint32_t>(idx), has_down});
    }
    int cuts = std::min(prune_cutoffs_[level], static_cast<int>(candidates.size()));
    std::nth_element(candidates.begin(),
                     candidates.begin() + cuts,
                     candidates.end());
    float_t mark = 0.0;
    for (int i = 0; i < cuts; ++i) {
        if (candidates[i].has_down) continue;
        // No special-token protection: closed IME vocabulary, no <unk>.
        if (level == opts_.num) {
            leaves_[candidates[i].index].pro = mark;
        } else {
            node_levels_[level][candidates[i].index].pro = mark;
        }
    }
    if (level == opts_.num) {
        auto new_size = CutLevelByMark(node_levels_[level - 1], leaves_, mark);
        leaves_.resize(new_size);
        prune_sizes_[level] = static_cast<int>(new_size);
    } else {
        auto new_size = CutLevelByMark(node_levels_[level - 1], node_levels_[level], mark);
        node_levels_[level].resize(new_size);
        prune_sizes_[level] = static_cast<int>(new_size);
    }
}

void Constructor::PruneBigramByPMI() {
    constexpr int level = 2;
    if (opts_.num < 2) return;
    int n = prune_sizes_[level] - 1;
    if (n <= 0) return;

    auto& l1 = node_levels_[1];
    auto& l2 = node_levels_[2];

    const std::uint32_t pmi_min_count = opts_.pmi_min_count;

    // Pre-count bigrams below the count threshold; they'll all be cut
    // regardless of reserve budget.
    std::size_t below_threshold = 0;
    if (pmi_min_count > 0) {
        for (int idx = 0; idx < n; ++idx) {
            if (static_cast<std::uint32_t>(l2[idx].cnt) <= pmi_min_count) {
                ++below_threshold;
            }
        }
    }

    int budget = (prune_cutoffs_.empty() ? 0 : prune_cutoffs_[level]);
    int cuts = std::max(static_cast<int>(below_threshold), budget);
    cuts = std::min(cuts, n);
    if (cuts <= 0) return;

    // count_back[v] = Σ_u count(u, v). Dense vector keyed by token id.
    std::vector<std::uint64_t> count_back(opts_.token_count + 1, 0);
    for (int idx = 0; idx < n; ++idx) {
        TokenID v = l2[idx].id;
        if (v < count_back.size()) {
            count_back[v] += static_cast<std::uint64_t>(l2[idx].cnt);
        }
    }

    std::vector<NodeScore> cands;
    cands.reserve(static_cast<std::size_t>(n));

    // No has_down carve-out: cascade deletion below takes any orphan
    // trigram children along with the bigram so the structural invariant
    // (every trigram has a parent bigram) is preserved.
    std::size_t up_idx = 0;
    for (int idx = 0; idx < n; ++idx) {
        while (up_idx + 1 < l1.size() &&
               l1[up_idx + 1].down <= static_cast<std::uint32_t>(idx)) {
            ++up_idx;
        }
        float_t cnt_uv = l2[idx].cnt;
        float_t score;
        if (pmi_min_count > 0
            && static_cast<std::uint32_t>(cnt_uv) <= pmi_min_count) {
            // Below count floor: assign the lowest possible score so this
            // bigram is cut first, before any PMI-ranking among survivors.
            score = 0.0;
        } else {
            float_t cnt_u = l1[up_idx].cnt;
            TokenID v = l2[idx].id;
            float_t cnt_back_v = (v < count_back.size())
                ? static_cast<float_t>(count_back[v]) : 0.0;
            // S(u, v) = count(u, v)^2 / (count(u) * count_back(v))
            score = (cnt_uv * cnt_uv) / (cnt_u * cnt_back_v);
        }
        cands.push_back(NodeScore{score, static_cast<std::uint32_t>(idx),
                                  false});
    }
    std::sort(cands.begin(), cands.end());

    constexpr float_t mark = 0.0;
    for (int i = 0; i < cuts; ++i) {
        std::uint32_t idx = cands[i].index;
        l2[idx].pro = mark;
        // Cascade: mark all trigram children so they're removed too.
        if (opts_.num >= 3) {
            std::uint32_t begin = l2[idx].down;
            std::uint32_t end = l2[idx + 1].down;
            for (std::uint32_t j = begin; j < end; ++j) {
                if (j < leaves_.size()) {
                    leaves_[j].pro = mark;
                }
            }
        }
    }
    auto new_l2_size = CutLevelByMark(l1, l2, mark);
    l2.resize(new_l2_size);
    prune_sizes_[level] = static_cast<int>(new_l2_size);

    // Cascade into leaves: cut marked trigrams, update new l2's .down.
    if (opts_.num >= 3) {
        auto new_leaf_size = CutLevelByMark(l2, leaves_, mark);
        leaves_.resize(new_leaf_size);
        prune_sizes_[opts_.num] = static_cast<int>(new_leaf_size);
    }
}

void Constructor::Discount() {
    constexpr float_t kDefaultD[] = {0.0005, 0.5, 0.5};
    for (int lvl = 1; lvl <= opts_.num; ++lvl) {
        std::size_t idx = static_cast<std::size_t>(lvl - 1);
        float_t d = (idx < opts_.discounts.size())
            ? opts_.discounts[idx]
            : (idx < 3 ? kDefaultD[idx] : 0.5);
        discounters_[lvl].Init(d);
    }

    Node& root = node_levels_[0][0];
    std::uint32_t effective_vocab = std::max<std::uint32_t>(opts_.token_count, 1);
    float_t base = 1.0 / effective_vocab;
    root.pro = static_cast<float>(-std::log(base));

    for (int lvl = 0; lvl < opts_.num; ++lvl) {
        auto& level = node_levels_[lvl];
        if (lvl == opts_.num - 1) {
            DiscountLevel(level, leaves_, discounters_[lvl + 1], lvl);
        } else {
            DiscountLevel(level, node_levels_[lvl + 1], discounters_[lvl + 1], lvl);
        }
    }
}

const void* Constructor::FindDown(int level, const Node* node, TokenID id) const {
    std::uint32_t begin = node->down;
    std::uint32_t end = (node + 1)->down;
    if (level == opts_.num - 1) {
        const Leave* base = leaves_.data();
        auto it = std::lower_bound(base + begin, base + end, id, [](const Leave& leaf, TokenID value) {
            return leaf.id < value;
        });
        if (it != base + end && it->id == id) {
            return it;
        }
        return nullptr;
    }
    const Node* base = node_levels_[level + 1].data();
    auto it = std::lower_bound(base + begin, base + end, id, [](const Node& down, TokenID value) {
        return down.id < value;
    });
    if (it != base + end && it->id == id) {
        return it;
    }
    return nullptr;
}

float_t Constructor::GetPro(int n, const TokenID* words) const {
    // Returns the LM probability P(words[n-1] | words[0..n-2]).
    // Descend the Trie through the history; on a miss, multiply by gamma
    // (stored as -log in bow) and retry with a shorter history.
    if (n <= 0 || node_levels_[0].empty()) {
        return std::exp(-node_levels_[0][0].pro);  // uniform prior
    }
    const Node* cur = &node_levels_[0][0];
    int lvl = 0;
    for (int i = 0; i < n; ++i) {
        const void* down = FindDown(lvl, cur, words[i]);
        if (down == nullptr) {
            float_t bow = std::exp(-cur->bow);
            return bow * GetPro(n - 1, words + 1);
        }
        if (lvl + 1 == opts_.num) {
            const Leave* leaf = static_cast<const Leave*>(down);
            return std::exp(-leaf->pro);
        }
        cur = static_cast<const Node*>(down);
        ++lvl;
    }
    return std::exp(-cur->pro);
}

void Constructor::Finalize() {
    FixDownPointers();
    { auto tmp = decltype(bigram_parents_)(); bigram_parents_.swap(tmp); }
    { auto tmp = decltype(leaf_parents_)(); leaf_parents_.swap(tmp); }
    AppendTails();
    Cut();
    Discount();
    for (int lvl = 1; lvl <= opts_.num; ++lvl) {
        std::cerr << "  D[" << lvl << "] = " << discounters_[lvl].D() << "\n";
    }
}

void Constructor::Prune(const std::vector<int>& reserves) {
    // reserves[0] = bigram reserve, reserves[1] = trigram reserve.
    // Unigrams are never pruned (full vocabulary is kept).
    prune_sizes_.assign(opts_.num + 1, 0);
    for (int lvl = 0; lvl < opts_.num; ++lvl) {
        prune_sizes_[lvl] = static_cast<int>(node_levels_[lvl].size());
    }
    prune_sizes_[opts_.num] = static_cast<int>(leaves_.size());

    auto reserve_for = [&](int lvl) -> int {
        // lvl=2 -> reserves[0], lvl=3 -> reserves[1]
        std::size_t slot = static_cast<std::size_t>(lvl - 2);
        if (slot >= reserves.size()) return 0;
        return reserves[slot];
    };

    prune_cutoffs_.assign(opts_.num + 1, 0);
    for (int lvl = 2; lvl <= opts_.num; ++lvl) {
        int remaining = prune_sizes_[lvl] - 1;
        prune_cutoffs_[lvl] = std::max(0, remaining - reserve_for(lvl));
    }

    // Order for a 3-gram IME model:
    //   1. Prune bigrams by PMI, cascade-deleting their trigram children.
    //      This escapes the `has_down` lock that Stolcke-pruned trigrams
    //      impose on bigrams, so universal-suffix bigrams like (X, 的) can
    //      actually be killed.
    //   2. If the cascade left more trigrams than the reserve allows, run
    //      the usual Stolcke pass on the survivors.
    if (opts_.num >= 2) {
        PruneBigramByPMI();
    }
    if (opts_.num >= 3 && prune_cutoffs_[opts_.num] > 0) {
        // Recompute probabilities after bigram PMI pruning so that the
        // Stolcke entropy scores for trigrams use up-to-date P(w|h) / γ(h).
        Discount();
        prune_cache_level_ = -1;
        prune_cache_index_ = -1;
        int remaining = prune_sizes_[opts_.num] - 1;
        int reserve = reserve_for(opts_.num);
        if (remaining > reserve) {
            prune_cutoffs_[opts_.num] = remaining - reserve;
            PruneLevel(opts_.num);
        }
    }
    // Entropy pruning removed entries; rerun Discount so gamma(h) and P_I(w|h)
    // are recomputed over the surviving set.
    Discount();
}

} // namespace sime
