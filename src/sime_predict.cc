#include "sime.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace sime {

std::vector<DecodeResult> Sime::NextTokens(
    const std::vector<TokenID>& context,
    std::size_t num,
    bool en) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_ || num == 0) return results;
    MaybeTrimCaches();

    // Only the last (num-1) context tokens matter — that's the maximum
    // history a num-gram model can condition on.  Walking from root with
    // at most (num-1) steps also guarantees we never reach leaf level.
    Scorer::Pos pos{};
    const auto tail = static_cast<std::size_t>(std::max(scorer_.Num() - 1, 1));
    std::size_t start = context.size() > tail ? context.size() - tail : 0;
    for (std::size_t i = start; i < context.size(); ++i) {
        Scorer::Pos next{};
        scorer_.ScoreMove(pos, context[i], next);
        pos = next;
    }

    // When filtering to English only, widen the scorer pool — many top
    // predictions will be Chinese and get dropped.
    const std::size_t pool = en ? num * 16 : num * 4;
    auto next_tokens = scorer_.NextTokens(pos, pool);

    // English tokens in Sime's corpora are ASCII-letter words (possibly
    // containing an apostrophe, e.g. don't, 's), or SentencePiece-style
    // pieces prefixed with U+2581 (word boundary). Contraction pieces like
    // "'s" / "'re" start with the apostrophe, so accept it too.
    auto is_english = [](const char32_t* chars) {
        char32_t c = chars[0];
        if (c == 0x2581) return true;
        if (c == U'\'') return true;
        return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
    };

    // User-aware ranking: fold the decayed user unigram/bigram adjustment
    // into each candidate's LM cost. CostAdjustment is <= 0, so a recent
    // user pick lowers the cost (lifts the word). With no history (or user
    // learning disabled) adj_cost == pro and the ascending stable_sort
    // reproduces the pure-LM order exactly — zero regression.
    const TokenID prev = context.empty() ? NotToken : context.back();
    const bool use_user = user_sentence_enabled_;

    struct Candidate {
        std::string text;
        TokenID tid;
        float_t adj_cost;
    };
    std::vector<Candidate> candidates;
    std::unordered_set<std::string> seen;
    for (const auto& [tid, pro] : next_tokens) {
        if (tid < StartToken) continue;

        const char32_t* chars = dict_.TokenAt(tid);
        if (!chars || chars[0] == 0) continue;
        if (en && !is_english(chars)) continue;

        std::u32string u32;
        for (std::size_t i = 0; chars[i] != 0; ++i)
            u32.push_back(chars[i]);
        std::string text = TextFromU32(u32);
        if (!seen.insert(text).second) continue;

        float_t adj_cost = pro;
        if (use_user) {
            adj_cost += user_sentence_.CostAdjustment(prev, tid, pro);
        }
        candidates.push_back({std::move(text), tid, adj_cost});
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                         return a.adj_cost < b.adj_cost;
                     });

    for (auto& c : candidates) {
        if (results.size() >= num) break;
        results.push_back({std::move(c.text), {}, {c.tid}, -c.adj_cost, 0});
    }

    return results;
}

std::vector<DecodeResult> Sime::GetTokens(
    std::string_view prefix,
    std::size_t num,
    bool en) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_ || num == 0 || prefix.empty()) return results;
    MaybeTrimCaches();

    struct Candidate {
        std::string text;
        TokenID id;
        float_t score;
    };
    std::vector<Candidate> candidates;

    auto collect = [&](Dict::DatType type,
                       const std::vector<trie::SearchResult>& matches) {
        for (const auto& r : matches) {
            auto entry = dict_.GetEntry(type, r.value);
            for (uint32_t i = 0; i < entry.count; ++i) {
                TokenID tid = entry.items[i].id;
                const char32_t* chars = dict_.TokenAt(tid);
                if (!chars || chars[0] == 0) continue;

                std::u32string u32;
                for (std::size_t j = 0; chars[j] != 0; ++j)
                    u32.push_back(chars[j]);
                std::string text = TextFromU32(u32);

                Scorer::Pos pos{};
                Scorer::Pos next{};
                float_t cost = scorer_.ScoreMove(pos, tid, next);

                candidates.push_back({std::move(text), tid, cost});
            }
        }
    };

    collect(Dict::LetterEn,
            dict_.Dat(Dict::LetterEn).FindWordsWithPrefix(prefix, num * 4));
    if (!en) {
        // Mixed mode: also search the pinyin DAT. Plain prefix match is
        // exactly what we want — user typed a literal pinyin prefix (possibly
        // with '\'' syllable separators) and wants every token whose stored
        // pinyin starts with it, including deeper multi-syllable completions.
        collect(Dict::LetterPinyin,
                dict_.Dat(Dict::LetterPinyin)
                    .FindWordsWithPrefix(prefix, num * 4));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });

    std::unordered_set<std::string> seen;
    for (auto& c : candidates) {
        if (results.size() >= num) break;
        if (!seen.insert(c.text).second) continue;
        results.push_back({std::move(c.text), {}, {c.id}, -c.score,
                           prefix.size()});
    }

    return results;
}

} // namespace sime