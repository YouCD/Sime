#include "construct.h"
#include "construct_detail.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace sime {

namespace {

constexpr TokenID TailMarker = 0x00FFFFFF;

} // namespace

bool Constructor::NodeScore::operator<(const NodeScore& other) const {
    return score < other.score;
}

Constructor::Constructor(ConstructOptions opts) : opts_(std::move(opts)) {
    if (opts_.num < 1 || opts_.num > 3) {
        throw std::invalid_argument("order must be in [1, 3]");
    }
    node_levels_.resize(opts_.num);
    for (auto& level : node_levels_) {
        level.reserve(1024);
    }
    leaves_.reserve(1024);

    node_levels_[0].push_back(Node{0, 0, 0.0, 0.0, 0.0});

    cuts_.assign(opts_.num + 1, 0);
    for (std::size_t i = 0; i < opts_.cutoffs.size() && i < static_cast<std::size_t>(opts_.num); ++i) {
        cuts_[i + 1] = opts_.cutoffs[i];
    }

    discounters_.resize(opts_.num + 1);
}

void Constructor::InsertUnigram(TokenID w, std::uint32_t cnt) {
    if (opts_.num < 1) return;
    node_levels_[0][0].cnt += cnt;
    // down set later in FixDownPointers.
    node_levels_[1].push_back(Node{w, 0, static_cast<float_t>(cnt), 0.0, 0.0});
}

void Constructor::InsertBigram(TokenID u, TokenID v, std::uint32_t cnt) {
    if (opts_.num < 2) return;
    if (cnt <= cuts_[2]) {
        return;
    }
    node_levels_[2].push_back(Node{v, 0, static_cast<float_t>(cnt), 0.0, 0.0});
    bigram_parents_.push_back(u);
}

void Constructor::InsertTrigram(TokenID u, TokenID v, TokenID w, std::uint32_t cnt) {
    if (opts_.num < 3) return;
    if (cnt > cuts_[opts_.num]) {
        leaves_.push_back(Leave{w, static_cast<float_t>(cnt), 0.0});
        leaf_parents_.push_back({u, v});
    }
}

void Constructor::FixDownPointers() {
    // l1[i].down = start index in l2 where bigrams whose parent is l1[i].id begin.
    if (opts_.num >= 2) {
        auto& l1 = node_levels_[1];
        std::size_t bi = 0;
        for (std::size_t i = 0; i < l1.size(); ++i) {
            l1[i].down = static_cast<std::uint32_t>(bi);
            while (bi < bigram_parents_.size() && bigram_parents_[bi] == l1[i].id) {
                ++bi;
            }
        }
    }

    // l2[j].down = start in leaves_ where trigrams whose parent is (u, v) begin.
    if (opts_.num >= 3) {
        auto& l1 = node_levels_[1];
        auto& l2 = node_levels_[2];
        std::size_t li = 0;
        std::size_t tri = 0;
        for (std::size_t j = 0; j < l2.size(); ++j) {
            while (li + 1 < l1.size() && l1[li + 1].down <= j) {
                ++li;
            }
            TokenID u = l1[li].id;
            TokenID v = l2[j].id;
            l2[j].down = static_cast<std::uint32_t>(tri);
            while (tri < leaf_parents_.size() &&
                   leaf_parents_[tri][0] == u &&
                   leaf_parents_[tri][1] == v) {
                ++tri;
            }
        }
    }
}

void Constructor::AppendTails() {
    const float_t tail_pro = std::numeric_limits<float_t>::denorm_min();
    for (int lvl = 0; lvl < opts_.num; ++lvl) {
        std::uint32_t down_count = 0;
        if (lvl == opts_.num - 1) {
            down_count = static_cast<std::uint32_t>(leaves_.size());
        } else {
            down_count = static_cast<std::uint32_t>(node_levels_[lvl + 1].size());
        }
        node_levels_[lvl].push_back(Node{TailMarker, down_count, 1.0, 0.0, 0.0});
        node_levels_[lvl].back().pro = tail_pro;
    }
    leaves_.push_back(Leave{0, 1.0, tail_pro});
}

void Constructor::Cut() {
    for (int lvl = opts_.num; lvl > 0; --lvl) {
        if (cuts_[lvl] <= 0) {
            continue;
        }
        auto& up_level = node_levels_[lvl - 1];
        if (lvl == opts_.num) {
            int new_size = CutLevel(up_level, leaves_, cuts_[lvl]);
            leaves_.resize(static_cast<std::size_t>(new_size));
        } else {
            auto& level = node_levels_[lvl];
            int new_size = CutLevel(up_level, level, cuts_[lvl]);
            level.resize(static_cast<std::size_t>(new_size));
        }
    }
}

float_t Constructor::CalcScore(int level,
                               std::vector<int>& indices,
                               std::vector<TokenID>& words) {
    // Relative entropy contribution of removing this single n-gram (h, w).
    // Mirrors SRILM NgramLM.cc::pruneProbs (Stolcke 1998 Eq. 6):
    //
    //   gamma'(h) = (numerator + P(w|h))    /   (denominator + P(w|h'))
    //   deltaProb = log P(w|h') + log gamma'(h) - log P(w|h)
    //   deltaH    = -P(h) * [ P(w|h) * deltaProb
    //                        + numerator * (log gamma'(h) - log gamma(h)) ]
    //
    // where numerator = 1 - sum_{kept} P(w'|h),
    //       denominator = 1 - sum_{kept} P(w'|h')  (backoff query at shorter
    //       history), which are exactly Sime's prune_cache_pa_ / pb_.
    //
    // Returns the KL contribution -deltaH (>= 0); caller sorts ascending and
    // prunes the smallest contributors.

    const Node& up = node_levels_[level - 1][indices[level - 1]];
    // up.bow stores -log gamma(h), so log gamma(h) = -up.bow.
    float_t log_gamma_old = -up.bow;

    // P(w|h): stored probability at this n-gram entry.
    float_t p_orig;
    if (level == opts_.num) {
        p_orig = std::exp(-leaves_[indices[level]].pro);
    } else {
        p_orig = std::exp(-node_levels_[level][indices[level]].pro);
    }
    if (p_orig <= 0.0) return 0.0;
    float_t log_p_orig = std::log(p_orig);

    // P(w|h_shorter) via the LM's backoff query.
    float_t p_backoff = GetPro(level - 1, words.data() + 2);
    if (p_backoff <= 0.0) return 0.0;
    float_t log_p_backoff = std::log(p_backoff);

    // Cache numerator / denominator for all siblings sharing this history.
    if (prune_cache_level_ != level - 1 ||
        prune_cache_index_ != indices[level - 1]) {
        prune_cache_level_ = level - 1;
        prune_cache_index_ = indices[level - 1];
        prune_cache_pa_ = 1.0;
        prune_cache_pb_ = 1.0;
        std::size_t begin = up.down;
        std::size_t end =
            node_levels_[level - 1][indices[level - 1] + 1].down;
        for (std::size_t j = begin; j < end; ++j) {
            float_t p_child;
            TokenID wid;
            if (level == opts_.num) {
                p_child = std::exp(-leaves_[j].pro);
                wid = leaves_[j].id;
            } else {
                p_child = std::exp(-node_levels_[level][j].pro);
                wid = node_levels_[level][j].id;
            }
            prune_cache_pa_ -= p_child;
            words[level] = wid;
            prune_cache_pb_ -= GetPro(level - 1, words.data() + 2);
        }
    }
    float_t num = prune_cache_pa_;
    float_t den = prune_cache_pb_;
    if (num <= 1e-12 || den <= 1e-12) return 0.0;

    // gamma' after removing this (h, w): the removed ngram joins the "unseen"
    // mass, so both numerator and denominator gain its probability share.
    float_t num_new = num + p_orig;
    float_t den_new = den + p_backoff;
    if (num_new <= 0.0 || den_new <= 0.0) return 0.0;
    float_t log_gamma_new = std::log(num_new) - std::log(den_new);

    // deltaProb = log p_new(w|h) - log p_old(w|h)
    //           = log P(w|h') + log gamma'(h) - log P(w|h)
    float_t log_delta_prob = log_p_backoff + log_gamma_new - log_p_orig;

    // p_hist = P(h) estimated from the current LM, joint of the history
    // tokens: prod_{k=1..level-1} P(words[k] | words[1..k-1]).
    float_t p_hist = 1.0;
    for (int k = 1; k < level; ++k) {
        p_hist *= GetPro(k, words.data() + 1);
    }

    // SRILM pruneProbs stores deltaEntropy with this sign convention: it is
    // the (non-negative) relative-entropy increase caused by pruning, and
    // ngrams with the smallest value are the safest to drop. The inner sum
    // is typically negative (log p' - log p < 0 dominates), and the leading
    // `-p_hist` flips it to the positive KL value.
    float_t delta_entropy = -p_hist * (
        p_orig * log_delta_prob +
        num * (log_gamma_new - log_gamma_old)
    );

    // Negative values come from numerical noise; clamp to 0.
    return std::max(delta_entropy, 0.0);
}

} // namespace sime
