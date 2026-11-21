#include "sime.h"

#include "pinyin_correct.h"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sime {

void Sime::InitNet(std::string_view input,
                      std::vector<Node>& net,
                      bool expansion) const {
    const std::size_t total = input.size();
    net.clear();
    net.resize(total + 2);

    auto emit = [&](std::size_t s, std::size_t new_col,
                    TokenID tid, const char* pieces,
                    bool en = false) {
        net[s].es.push_back({s, new_col, tid, pieces, 0, false, en});
    };

    // Greedy syllable walk. seg_is_syllable[i] is true iff segment i is a
    // *terminal* (non-extendable) known pinyin syllable. Extendable
    // syllables like "xu" (→ xue/xun/xuan) are flagged false so the
    // expansion gate fires through them — the user may have abbreviated a
    // longer syllable to its prefix. best=1 fallback and apostrophe
    // segments are also incomplete.
    std::vector<std::size_t> seg_bounds;
    std::vector<bool> seg_is_syllable;
    {
        seg_bounds.push_back(0);
        std::size_t pos = 0;
        while (pos < total) {
            if (input[pos] == '\'') {
                ++pos;
                seg_bounds.push_back(pos);
                seg_is_syllable.push_back(false);
                continue;
            }
            if (input[pos] == static_cast<char>(0xE2)) {
                // ▁ (U+2581, 3 bytes) word boundary from a typed space.
                // Consume it as one terminal segment so the expansion latch
                // does not treat the space as three incomplete fallbacks.
                pos += 3;
                seg_bounds.push_back(pos);
                seg_is_syllable.push_back(true);
                continue;
            }
            bool terminal = false;
            std::size_t best = 1;
            for (std::size_t len = std::min(total - pos, std::size_t(6));
                 len >= 2; --len) {
                std::string p(input.substr(pos, len));
                if (Dict::IsKnownPinyin(p)) {
                    best = len;
                    terminal = !Dict::IsExtendablePinyin(p);
                    break;
                }
            }
            pos += best;
            seg_bounds.push_back(pos);
            seg_is_syllable.push_back(terminal);
        }
    }

    for (std::size_t s = 0; s < total; ++s) {
        if (input[s] == '\'') {
            net[s].es.push_back({s, s + 1, NotToken});
            continue;
        }
        if (input[s] == static_cast<char>(0xE2)) {
            // ▁ boundary: free pass-through edge (like apostrophe). The two
            // continuation bytes get no column edges of their own.
            net[s].es.push_back({s, s + 3, NotToken});
            continue;
        }
        if (input[s] == static_cast<char>(0x96) ||
            input[s] == static_cast<char>(0x81)) {
            continue;
        }

        std::string_view suffix = input.substr(s);

        auto emit_results = [&](const std::vector<trie::SearchResult>& results,
                                Dict::DatType type, bool en) {
            for (const auto& r : results) {
                auto entry = dict_.GetEntry(type, r.value);
                for (uint32_t i = 0; i < entry.count; ++i) {
                    emit(s, s + r.length, entry.items[i].id,
                         entry.items[i].pieces, en);
                }
            }
        };

        emit_results(
            dict_.Dat(Dict::LetterPinyin).PrefixSearchPinyin(suffix, 512, 2),
            Dict::LetterPinyin, false);
        emit_results(
            dict_.Dat(Dict::LetterEn).PrefixSearch(suffix, 512),
            Dict::LetterEn, true);
    }

    // Boundary-driven expansion: walk forward; once any segment in the
    // span is incomplete (non-terminal pinyin or best=1 fallback), fire
    // FWWPP at every subsequent boundary target. Latching catches
    // "incomplete-then-terminal" spans like "qchuang" → 起床, where the
    // first seg (q) is incomplete but the trailing seg (chuang) is
    // terminal — without the latch, PSPinyin can't reach 起床 because
    // it drops fuzzy states. English tail completion fires per boundary
    // if any non-terminal segment remains in the suffix.
    if (expansion) {
        for (std::size_t s_idx = 0; s_idx + 1 < seg_bounds.size(); ++s_idx) {
            std::size_t s = seg_bounds[s_idx];
            if (s >= total || input[s] == '\'' ||
                input[s] == static_cast<char>(0xE2)) continue;

            bool saw_incomplete = false;
            for (std::size_t bi = s_idx + 1; bi < seg_bounds.size(); ++bi) {
                // A word does not cross a space: stop the boundary walk once
                // the next segment is a ▁ boundary.
                if (input[seg_bounds[bi - 1]] == static_cast<char>(0xE2)) break;
                if (!seg_is_syllable[bi - 1]) saw_incomplete = true;
                if (!saw_incomplete) continue;
                std::size_t target = seg_bounds[bi];
                auto raw_prefix = input.substr(s, target - s);
                std::string prefix;
                prefix.reserve(raw_prefix.size());
                for (char c : raw_prefix) {
                    if (c != '\'') prefix.push_back(c);
                }
                auto results = dict_.Dat(Dict::LetterPinyin)
                    .FindWordsWithPrefixPinyin(prefix, 512, 2);
                for (const auto& r : results) {
                    auto entry = dict_.GetEntry(Dict::LetterPinyin, r.value);
                    for (uint32_t i = 0; i < entry.count; ++i) {
                        net[s].es.push_back({s, target, entry.items[i].id,
                                             entry.items[i].pieces, 0, true,
                                             false});
                    }
                }
            }
            if (saw_incomplete) {
                // English tail completion must not cross a space: truncate
                // the searched suffix (and the emitted edge) at the first ▁.
                std::size_t end = total;
                for (std::size_t k = s; k < total; ++k) {
                    if (input[k] == static_cast<char>(0xE2)) { end = k; break; }
                }
                std::string_view suffix = input.substr(s, end - s);
                auto results = dict_.Dat(Dict::LetterEn)
                    .FindWordsWithPrefix(suffix, 1024);
                for (const auto& r : results) {
                    if (r.length <= suffix.size()) continue;
                    auto entry = dict_.GetEntry(Dict::LetterEn, r.value);
                    for (uint32_t i = 0; i < entry.count; ++i) {
                        net[s].es.push_back({s, end, entry.items[i].id,
                                             entry.items[i].pieces, 0, true,
                                             true});
                    }
                }
            }
        }
    }

    // Additive word-level pinyin correction. The LM is word-based, so a
    // corrected syllable must be looked up as part of a whole word key
    // (e.g. zongguo -> zhong'guo -> 中国) to inherit the word's LM score;
    // chaining corrected single chars scores far worse. A bounded DFS
    // enumerates corrected syllable sequences (<=3 syllables) over the
    // ORIGINAL input span, joins them with '\'' and looks up the word DAT.
    // Fuzzy pronunciation variants fire on valid syllables, typo
    // (edit-distance-1) variants on invalid ones. Every emitted edge carries
    // a correction cost; the exact path (correction=0) is never displaced, so
    // correct input does not regress. Spans cross greedy seg boundaries so a
    // typo like "jei" (split j+ei by the walk) is still recovered as jie.
    if (enable_fuzzy_correction_ || enable_typo_correction_) {
        const auto& py_dat = dict_.Dat(Dict::LetterPinyin);
        struct SylCands {
            bool known = false;
            std::vector<std::string> variants;
        };
        std::unordered_map<std::size_t, SylCands> ccache;
        auto cands_at = [&](std::size_t pos, std::size_t len) -> const SylCands& {
            const std::size_t key = pos * 8 + len;
            auto it = ccache.find(key);
            if (it != ccache.end()) return it->second;
            SylCands sc;
            std::string seg(input.substr(pos, len));
            sc.known = Dict::IsKnownPinyin(seg);
            if (len >= 2) {
                if (sc.known) {
                    if (enable_fuzzy_correction_)
                        sc.variants = pinyin_correct::FuzzyVariants(seg);
                } else if (enable_typo_correction_
                           && !Dict::IsExtendablePinyin(seg)) {
                    // Extendable invalid prefixes (zho -> zhou) are already
                    // covered by abbreviation/expansion; only genuinely
                    // dead-end segments need typo correction.
                    sc.variants = pinyin_correct::TypoVariants(seg);
                }
            }
            return ccache.emplace(key, std::move(sc)).first->second;
        };

        constexpr int kMaxSyl = 3;
        constexpr int kLookupBudget = 64;
        std::vector<std::string> parts;
        parts.reserve(kMaxSyl);

        auto dfs = [&](auto&& self, std::size_t span_start, std::size_t pos,
                       int nsyl, int ncorrected, float_t cost,
                       int& budget) -> void {
            // Skip fuzzy-only single-syllable corrections: a valid syllable
            // already has its exact homophones, so re-emitting them as
            // corrected edges only bloats the beam. Keep typo single-syllable
            // recovery (invalid input) and all multi-syllable word edges.
            const bool emit_here = nsyl >= 1 && ncorrected >= 1 && budget > 0
                && (nsyl >= 2 || cost >= typo_correction_penalty_);
            if (emit_here) {
                std::string key;
                for (int i = 0; i < nsyl; ++i) {
                    if (i) key.push_back('\'');
                    key += parts[static_cast<std::size_t>(i)];
                }
                uint32_t val = 0;
                --budget;
                if (py_dat.Get(key, val)) {
                    auto entry = dict_.GetEntry(Dict::LetterPinyin, val);
                    for (uint32_t i = 0; i < entry.count; ++i) {
                        net[span_start].es.push_back(
                            {span_start, pos, entry.items[i].id,
                             entry.items[i].pieces, 0, false, false, cost});
                    }
                }
            }
            if (nsyl >= kMaxSyl || budget <= 0) return;
            // A correction must occur within the first two syllables; stop
            // walking purely-original prefixes so valid input stays cheap.
            if (ncorrected == 0 && nsyl >= 2) return;
            for (std::size_t len = 1; len <= 6 && pos + len <= total; ++len) {
                const char c = input[pos + len - 1];
                // Never correct across a space: ▁ bytes must not be folded
                // into a syllable (TypoVariants could otherwise delete them).
                if (c == '\'' || c == static_cast<char>(0xE2) ||
                    c == static_cast<char>(0x96) || c == static_cast<char>(0x81))
                    break;
                const SylCands& sc = cands_at(pos, len);
                if (sc.known) {
                    // Keep the original syllable (uncorrected) so a word with
                    // a single corrected syllable can still be formed.
                    parts.push_back(std::string(input.substr(pos, len)));
                    self(self, span_start, pos + len, nsyl + 1, ncorrected,
                         cost, budget);
                    parts.pop_back();
                    for (const auto& v : sc.variants) {
                        parts.push_back(v);
                        self(self, span_start, pos + len, nsyl + 1,
                             ncorrected + 1, cost + fuzzy_pinyin_penalty_, budget);
                        parts.pop_back();
                        if (budget <= 0) return;
                    }
                } else {
                    for (const auto& v : sc.variants) {
                        parts.push_back(v);
                        self(self, span_start, pos + len, nsyl + 1,
                             ncorrected + 1, cost + typo_correction_penalty_, budget);
                        parts.pop_back();
                        if (budget <= 0) return;
                    }
                }
                if (budget <= 0) return;
            }
        };

        // Words begin at syllable boundaries, so only launch the correction
        // DFS from the greedy seg boundaries (≈1/3 of all character positions).
        // Span lengths inside the DFS are still enumerated freely, so a typo
        // that the greedy walk split (jei -> j+ei) is recovered from an earlier
        // boundary. This keeps the typo pass cheap on long valid sentences.
        for (std::size_t s : seg_bounds) {
            if (s >= total || input[s] == '\'' ||
                input[s] == static_cast<char>(0xE2)) continue;
            int budget = kLookupBudget;
            dfs(dfs, s, s, 0, 0, 0.0f, budget);
        }
    }

    // Two-track per-bucket tier filter: CN edges (!english) and English
    // edges (english) are filtered on independent tracks, so an exact
    // edge on one track doesn't suppress edges on the other.
    //   tier 0: !expansion (exact)
    //   tier 1: expansion (abbrev / completion)
    auto tier_of = [](const Link& e) -> uint8_t {
        if (e.id == NotToken) return 0;
        return e.expansion ? 1 : 0;
    };
    std::vector<uint8_t> best_py(total + 2, 0xFF);
    std::vector<uint8_t> best_en(total + 2, 0xFF);
    for (std::size_t i = 0; i < total; ++i) {
        auto& edges = net[i].es;
        if (edges.empty()) continue;
        for (const auto& e : edges) {
            uint8_t t = tier_of(e);
            auto& best = e.english ? best_en : best_py;
            if (t < best[e.end]) best[e.end] = t;
        }
        edges.erase(std::remove_if(edges.begin(), edges.end(),
            [&](const Link& e) {
                if (e.id == NotToken) return false;
                const auto& best = e.english ? best_en : best_py;
                return tier_of(e) > best[e.end];
            }), edges.end());
        for (const auto& e : edges) {
            best_py[e.end] = 0xFF;
            best_en[e.end] = 0xFF;
        }
    }

    net[total].es.push_back({total, total + 1, NotToken});
}

} // namespace sime