#include "user.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sime {

namespace {

// Ney's absolute-discount values, mirroring the main LM's defaults in
// construct.cc (DefaultDiscounts = {0.0005, 0.5, 0.5}). Subtracted
// from each raw count; lost mass is redistributed via gamma to the
// lower-order distribution.
constexpr float_t kBigramDiscount = 0.5;
constexpr float_t kUnigramDiscount = 0.0005;

void IncCount(std::unordered_map<TokenID, std::uint32_t>& m,
              TokenID key, std::uint32_t delta) {
    m[key] += delta;
}

void DecCount(std::unordered_map<TokenID, std::uint32_t>& m,
              TokenID key, std::uint32_t delta) {
    auto it = m.find(key);
    if (it == m.end()) return;
    if (it->second <= delta) m.erase(it);
    else it->second -= delta;
}

// Bigram inc/dec also tracks N1+(prev, *) — distinct-successor count.
// We need to know when count crosses 0 in either direction to update
// the successor count atomically.
void IncBigramTrackSucc(
    std::unordered_map<std::uint64_t, std::uint32_t>& bigram,
    std::unordered_map<TokenID, std::uint32_t>& succ,
    TokenID prev, std::uint64_t key, std::uint32_t delta) {
    auto& count = bigram[key];  // creates entry with 0 if absent
    bool was_zero = (count == 0);
    count += delta;
    if (was_zero && count > 0) {
        ++succ[prev];
    }
}

void DecBigramTrackSucc(
    std::unordered_map<std::uint64_t, std::uint32_t>& bigram,
    std::unordered_map<TokenID, std::uint32_t>& succ,
    TokenID prev, std::uint64_t key, std::uint32_t delta) {
    auto it = bigram.find(key);
    if (it == bigram.end()) return;
    bool was_nonzero = (it->second > 0);
    bool became_zero;
    if (it->second <= delta) {
        bigram.erase(it);
        became_zero = was_nonzero;
    } else {
        it->second -= delta;
        became_zero = false;
    }
    if (became_zero) {
        auto sit = succ.find(prev);
        if (sit != succ.end()) {
            if (sit->second <= 1) succ.erase(sit);
            else --sit->second;
        }
    }
}

} // namespace

void UserSentence::Clear() {
    sentences_.clear();
    unigram_.clear();
    bigram_.clear();
    distinct_successors_.clear();
    total_unigram_count_ = 0;
    seed_unigram_.clear();
    seed_total_ = 0;
    seed_tick_.clear();
    decay_w_.clear();
    decay_t_.clear();
    decay_total_ = 0;
    decay_total_t_ = 0;
}

void UserSentence::SetMaxSentenceCount(std::size_t count) {
    max_sentence_count_ = count;
    const std::uint32_t now = Now();
    while (sentences_.size() > max_sentence_count_) {
        AdjustEntryCounts(sentences_.back(), -1);
        AdjustEntryDecay(sentences_.back(), -1, now);
        sentences_.pop_back();
    }
}

void UserSentence::SetWeight(float_t weight) {
    if (weight < 0.0) {
        weight_ = 0.0;
    } else {
        weight_ = weight;
    }
}

void UserSentence::Add(const Sentence& context, const Sentence& tokens) {
    Sentence clean = CleanSentence(tokens);
    if (clean.empty()) {
        return;
    }
    Selection sel{std::move(clean), Now()};

    Sentence clean_context = CleanSentence(context);
    if (!clean_context.empty() && !sentences_.empty()) {
        Sentence latest_flat = Flatten(sentences_.front());
        if (EndsWith(latest_flat, clean_context)) {
            AppendToLatest(std::move(sel));
            return;
        }
    }
    StartParagraph(std::move(sel));
}

std::uint32_t UserSentence::UnigramCount(TokenID token) const {
    if (token == NotToken) {
        return 0;
    }
    std::uint32_t count = 0;
    auto iter = unigram_.find(token);
    if (iter != unigram_.end()) {
        count += iter->second;
    }
    auto seed = seed_unigram_.find(token);
    if (seed != seed_unigram_.end()) {
        count += seed->second;
    }
    return count;
}

std::uint32_t UserSentence::ParagraphCount(TokenID token) const {
    if (token == NotToken) {
        return 0;
    }
    auto iter = unigram_.find(token);
    return iter == unigram_.end() ? 0 : iter->second;
}

void UserSentence::SeedUnigram(TokenID token, std::uint32_t count,
                               std::uint32_t tick) {
    if (token == NotToken || count == 0) {
        return;
    }
    if (tick == 0) {
        tick = Now();
    }
    seed_unigram_[token] += count;
    seed_total_ += count;
    seed_tick_[token] = tick;
    // Runtime import seeds are fresh (tick == now); during load this
    // provisional weight is discarded and reconstructed by Rebuild at
    // the stored historical tick.
    TouchDecay(token, static_cast<double>(count), Now());
}

std::uint32_t UserSentence::SeedCount(TokenID token) const {
    if (token == NotToken) {
        return 0;
    }
    auto iter = seed_unigram_.find(token);
    return iter == seed_unigram_.end() ? 0 : iter->second;
}

std::uint32_t UserSentence::BigramCount(TokenID prev, TokenID cur) const {
    if (prev == NotToken || cur == NotToken) {
        return 0;
    }
    auto iter = bigram_.find(BigramKey(prev, cur));
    if (iter == bigram_.end()) {
        return 0;
    }
    return iter->second;
}

std::uint32_t UserSentence::DistinctSuccessors(TokenID prev) const {
    if (prev == NotToken) {
        return 0;
    }
    auto it = distinct_successors_.find(prev);
    return it == distinct_successors_.end() ? 0 : it->second;
}

float_t UserSentence::CostAdjustment(TokenID prev, TokenID cur,
                                     float_t lm_step) const {
    if (cur == NotToken || weight_ <= 0.0 || weight_ >= 1.0 ||
        TotalUnigramCount() == 0) {
        return 0.0;
    }
    auto cur_count = UnigramCount(cur);
    if (cur_count == 0) {
        return 0.0;
    }

    // Interpolated Absolute Discounting at both levels — same family
    // and same per-order D values as the main LM (construct.cc):
    //   gamma(h)    = D · N1+(h, *) / c(h)
    //   P(w|h)      = max(c(h,w) - D, 0) / c(h)  +  gamma(h) · P(w|h')
    // The unigram tail uses exponentially decayed weights (Rime-style
    // recency): recent picks outweigh stale ones. Bigram alpha/gamma
    // below still use raw paragraph counts so transition probabilities
    // are never distorted by decay. The tail "terminates at zero"
    // rather than backing off to a uniform base; the outer mixture-with-
    // LM provides the real fallback for tokens the user never picked.
    const std::uint32_t now_q = Now();
    const double total_w =
        decay_total_ * DecayFactor(decay_total_t_, now_q);
    if (total_w <= 0.0) {
        return 0.0;
    }
    double cur_w = 0.0;
    auto wit = decay_w_.find(cur);
    if (wit != decay_w_.end()) {
        auto tit = decay_t_.find(cur);
        const std::uint32_t cur_t =
            (tit != decay_t_.end()) ? tit->second : now_q;
        cur_w = wit->second * DecayFactor(cur_t, now_q);
    }
    const float_t pr_unigram = static_cast<float_t>(
        std::max(cur_w - static_cast<double>(kUnigramDiscount), 0.0) /
        total_w);

    float_t pr_user = pr_unigram;
    if (prev != NotToken) {
        auto prev_count = ParagraphCount(prev);
        if (prev_count > 0) {
            auto bigram_count = BigramCount(prev, cur);
            auto n_succ = DistinctSuccessors(prev);
            const float_t denom = static_cast<float_t>(prev_count);
            const float_t alpha =
                std::max(static_cast<float_t>(bigram_count) - kBigramDiscount,
                         0.0) /
                denom;
            const float_t gamma =
                kBigramDiscount * static_cast<float_t>(n_succ) / denom;
            pr_user = alpha + gamma * pr_unigram;
        }
    }
    pr_user = std::min<float_t>(pr_user, 1.0);
    if (pr_user <= 0.0) {
        return 0.0;
    }

    // Mixture with LM in probability space, max-floored at lm_log_prob
    // so user history can only improve the score, never worsen it.
    const float_t lm_log_prob = -lm_step;
    const float_t user_log_prob = std::log(pr_user);
    const float_t a = lm_log_prob + std::log(1.0 - weight_);
    const float_t b = user_log_prob + std::log(weight_);
    float_t mix;
    if (a > b) {
        mix = a + std::log1p(std::exp(b - a));
    } else {
        mix = b + std::log1p(std::exp(a - b));
    }
    const float_t mixed_log_prob = std::max(lm_log_prob, mix);
    return lm_log_prob - mixed_log_prob;
}

std::uint64_t UserSentence::BigramKey(TokenID prev, TokenID cur) {
    return (static_cast<std::uint64_t>(prev) << 32U) |
           static_cast<std::uint64_t>(cur);
}

UserSentence::Sentence UserSentence::CleanSentence(const Sentence& sentence) {
    Sentence clean;
    clean.reserve(sentence.size());
    for (auto token : sentence) {
        if (token != NotToken) {
            clean.push_back(token);
        }
    }
    return clean;
}

UserSentence::Sentence UserSentence::Flatten(const Entry& entry) {
    Sentence flat;
    std::size_t total = 0;
    for (const auto& sel : entry.selections) {
        total += sel.tokens.size();
    }
    flat.reserve(total);
    for (const auto& sel : entry.selections) {
        flat.insert(flat.end(), sel.tokens.begin(), sel.tokens.end());
    }
    return flat;
}

bool UserSentence::EndsWith(const Sentence& sentence, const Sentence& suffix) {
    if (suffix.size() > sentence.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), sentence.rbegin());
}

void UserSentence::StartParagraph(Selection selection) {
    if (max_sentence_count_ == 0) {
        return;
    }
    const std::uint32_t now = Now();
    Entry entry;
    entry.selections.push_back(std::move(selection));
    AdjustEntryCounts(entry, +1);
    AdjustEntryDecay(entry, +1, now);
    sentences_.push_front(std::move(entry));
    while (sentences_.size() > max_sentence_count_) {
        AdjustEntryCounts(sentences_.back(), -1);
        AdjustEntryDecay(sentences_.back(), -1, now);
        sentences_.pop_back();
    }
}

void UserSentence::AppendToLatest(Selection selection) {
    if (sentences_.empty()) {
        StartParagraph(std::move(selection));
        return;
    }
    auto& latest = sentences_.front();

    // Incremental count update: tail of existing flattened tokens is
    // the "previous" for the first token of the new selection. After
    // that, internal bigrams of the new selection contribute too.
    TokenID prev_tail = NotToken;
    if (!latest.selections.empty()) {
        const auto& last_sel = latest.selections.back();
        if (!last_sel.tokens.empty()) {
            prev_tail = last_sel.tokens.back();
        }
    }
    for (auto token : selection.tokens) {
        IncCount(unigram_, token, 1);
        ++total_unigram_count_;
    }
    const std::uint32_t now = Now();
    const double fresh = DecayFactor(selection.tick, now);
    for (auto token : selection.tokens) {
        TouchDecay(token, fresh, now);
    }
    if (prev_tail != NotToken && !selection.tokens.empty()) {
        IncBigramTrackSucc(bigram_, distinct_successors_, prev_tail,
                           BigramKey(prev_tail, selection.tokens.front()), 1);
    }
    for (std::size_t i = 1; i < selection.tokens.size(); ++i) {
        IncBigramTrackSucc(bigram_, distinct_successors_,
                           selection.tokens[i - 1],
                           BigramKey(selection.tokens[i - 1],
                                     selection.tokens[i]), 1);
    }

    latest.selections.push_back(std::move(selection));
}

void UserSentence::AdjustEntryCounts(const Entry& entry, int delta) {
    if (delta == 0) {
        return;
    }
    Sentence flat = Flatten(entry);
    if (flat.empty()) {
        return;
    }
    if (delta > 0) {
        auto d = static_cast<std::uint32_t>(delta);
        for (auto token : flat) {
            IncCount(unigram_, token, d);
            total_unigram_count_ += d;
        }
        for (std::size_t i = 1; i < flat.size(); ++i) {
            IncBigramTrackSucc(bigram_, distinct_successors_, flat[i - 1],
                               BigramKey(flat[i - 1], flat[i]), d);
        }
    } else {
        auto d = static_cast<std::uint32_t>(-delta);
        for (auto token : flat) {
            DecCount(unigram_, token, d);
            if (total_unigram_count_ >= d) {
                total_unigram_count_ -= d;
            } else {
                total_unigram_count_ = 0;
            }
        }
        for (std::size_t i = 1; i < flat.size(); ++i) {
            DecBigramTrackSucc(bigram_, distinct_successors_, flat[i - 1],
                               BigramKey(flat[i - 1], flat[i]), d);
        }
    }
}

void UserSentence::Rebuild() {
    unigram_.clear();
    bigram_.clear();
    distinct_successors_.clear();
    total_unigram_count_ = 0;
    for (const auto& entry : sentences_) {
        AdjustEntryCounts(entry, +1);
    }

    // Reconstruct decayed weights at a common reference time (now).
    // Unlike the runtime TouchDecay path (monotonic timestamps), load
    // replays arbitrary historical ticks, so each contribution is
    // decayed straight to now and summed — no incremental refresh.
    const std::uint32_t now = Now();
    decay_w_.clear();
    decay_t_.clear();
    decay_total_ = 0;
    decay_total_t_ = now;
    for (const auto& entry : sentences_) {
        for (const auto& sel : entry.selections) {
            const double f = DecayFactor(sel.tick, now);
            for (auto token : sel.tokens) {
                decay_w_[token] += f;
                decay_total_ += f;
            }
        }
    }
    for (const auto& [token, count] : seed_unigram_) {
        auto tit = seed_tick_.find(token);
        const std::uint32_t t = (tit != seed_tick_.end()) ? tit->second : now;
        const double w = static_cast<double>(count) * DecayFactor(t, now);
        decay_w_[token] += w;
        decay_total_ += w;
    }
    for (const auto& [token, w] : decay_w_) {
        decay_t_[token] = now;
    }
}

} // namespace sime