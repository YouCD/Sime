#include "sime.h"

#include "pinyin_correct.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sime {

void Sime::InitNumNet(std::string_view start,
                              std::string_view nums,
                              std::vector<Node>& net,
                              bool expansion) const {
    const std::size_t p = start.size();
    const std::size_t d = nums.size();
    const std::size_t total = p + d;

    net.clear();
    net.resize(total + 2);

    auto emit = [&](std::size_t s, std::size_t new_col,
                    TokenID tid, const char* pieces,
                    bool en = false) {
        net[s].es.push_back({s, new_col, tid, pieces, 0, false, en});
    };

    auto emit_dat = [&](std::size_t s, Dict::DatType type,
                        std::string_view suffix, std::size_t max_end,
                        bool pinyin, bool en) {
        auto results = pinyin
            ? dict_.Dat(type).PrefixSearchPinyin(suffix, 512)
            : dict_.Dat(type).PrefixSearch(suffix, 512);
        for (const auto& r : results) {
            std::size_t new_col = s + r.length;
            if (new_col > max_end) continue;
            auto entry = dict_.GetEntry(type, r.value);
            for (uint32_t i = 0; i < entry.count; ++i) {
                emit(s, new_col, entry.items[i].id, entry.items[i].pieces, en);
            }
        }
    };

    auto expand_dat = [&](std::size_t s, Dict::DatType type,
                          std::string_view tail, std::size_t target_col,
                          std::size_t max_num, bool pinyin, bool en) {
        auto results = pinyin
            ? dict_.Dat(type).FindWordsWithPrefixPinyin(tail, max_num)
            : dict_.Dat(type).FindWordsWithPrefix(tail, max_num);
        for (const auto& r : results) {
            // For English exact-stem matches PrefixSearch already emits
            // them, so skip duplicates. CN fuzzy walks (q→qi etc.) can
            // hit eow at length == tail.size() — keep those, since
            // PrefixSearchPinyin drops fuzzy states and would miss them.
            if (!pinyin && r.length <= tail.size()) continue;
            auto entry = dict_.GetEntry(type, r.value);
            for (uint32_t i = 0; i < entry.count; ++i) {
                net[s].es.push_back({s, target_col, entry.items[i].id,
                                     entry.items[i].pieces, 0, true, en});
            }
        }
    };

    // letter_bounds = greedy syllable walk (mirrors InitNet).
    // letter_is_syllable[i] is true iff segment i is a *terminal*
    // (non-extendable) pinyin syllable.
    std::vector<std::size_t> letter_bounds;
    std::vector<bool> letter_is_syllable;
    if (p > 0) {
        letter_bounds.push_back(0);
        std::size_t pos = 0;
        while (pos < p) {
            if (start[pos] == '\'') {
                ++pos;
                letter_bounds.push_back(pos);
                letter_is_syllable.push_back(false);
                continue;
            }
            bool terminal = false;
            std::size_t best = 1;
            for (std::size_t len = std::min(p - pos, std::size_t(6));
                 len >= 2; --len) {
                std::string ps(start.substr(pos, len));
                if (Dict::IsKnownPinyin(ps)) {
                    best = len;
                    terminal = !Dict::IsExtendablePinyin(ps);
                    break;
                }
            }
            pos += best;
            letter_bounds.push_back(pos);
            letter_is_syllable.push_back(terminal);
        }
    }

    // digit_bounds = greedy T9-syllable walk (mirrors letter_bounds).
    // digit_is_syllable[i] is true iff segment i is a *terminal*
    // (non-extendable) T9 syllable.
    std::vector<std::size_t> digit_bounds;
    std::vector<bool> digit_is_syllable;
    if (d > 0) {
        digit_bounds.push_back(0);
        std::size_t pos = 0;
        while (pos < d) {
            if (nums[pos] == '\'') {
                ++pos;
                digit_bounds.push_back(pos);
                digit_is_syllable.push_back(false);
                continue;
            }
            bool terminal = false;
            std::size_t best = 1;
            for (std::size_t len = std::min(d - pos, std::size_t(6));
                 len >= 2; --len) {
                std::string_view span = nums.substr(pos, len);
                if (Dict::IsKnownT9Syllable(span)) {
                    best = len;
                    terminal = !Dict::IsExtendableT9Syllable(span);
                    break;
                }
            }
            pos += best;
            digit_bounds.push_back(pos);
            digit_is_syllable.push_back(terminal);
        }
    }

    for (std::size_t s = 0; s < total; ++s) {
        // === Prefix letter columns ===
        if (s < p) {
            if (start[s] == '\'') {
                net[s].es.push_back({s, s + 1, NotToken});
                continue;
            }

            std::string_view suffix = start.substr(s);
            emit_dat(s, Dict::LetterPinyin, suffix, p, true, false);
            emit_dat(s, Dict::LetterEn, suffix, p, false, true);

            // Cross-boundary: search with letters + digits combined.
            // PrefixSearchT9 now handles letter chars via AdvancePinyin,
            // so a mixed string like "b'9'3" works directly.
            // Only search from the last few letter columns — earlier columns
            // would need impractically long words to span into digit territory.
            if (!nums.empty() && (p - s) <= 6) {
                std::string cross(suffix);
                cross += nums;
                auto cross_emit = [&](Dict::DatType type,
                                      trie::DoubleArray::CharExpander expander,
                                      std::size_t max_num, bool en) {
                    auto results = dict_.Dat(type).PrefixSearchT9(
                        cross, expander, max_num);
                    for (const auto& r : results) {
                        std::size_t new_col = s + r.length;
                        if (new_col <= p) continue;  // already covered by letter-only search
                        if (new_col > total) continue;
                        auto entry = dict_.GetEntry(type, r.value);
                        for (uint32_t i = 0; i < entry.count; ++i) {
                            emit(s, new_col, entry.items[i].id,
                                 entry.items[i].pieces, en);
                        }
                    }
                };
                cross_emit(Dict::LetterPinyin, Dict::NumToLettersLower, 512, false);
                cross_emit(Dict::LetterEn, Dict::NumToLetters, 512, true);
            }
            continue;
        }

        // === Digit columns ===
        const std::size_t dpos = s - p;

        if (nums[dpos] == '\'') {
            net[s].es.push_back({s, s + 1, NotToken});
            continue;
        }

        // T9: expand digits to letters and search letter DATs
        std::string_view num_suffix = nums.substr(dpos);
        auto t9_emit = [&](Dict::DatType type, trie::DoubleArray::CharExpander expander,
                           std::size_t max_num, bool en) {
            auto results = dict_.Dat(type).PrefixSearchT9(
                num_suffix, expander, max_num);
            for (const auto& r : results) {
                std::size_t new_col = s + r.length;
                if (new_col > total) continue;
                auto entry = dict_.GetEntry(type, r.value);
                for (uint32_t i = 0; i < entry.count; ++i) {
                    emit(s, new_col, entry.items[i].id, entry.items[i].pieces, en);
                }
            }
        };
        // Pinyin DAT: lowercase only (no uppercase in pinyin)
        t9_emit(Dict::LetterPinyin, Dict::NumToLettersLower, 512, false);
        // English DAT: both cases
        t9_emit(Dict::LetterEn, Dict::NumToLetters, 512, true);
    }

    // Letter-side boundary expansion (mirrors InitNet): fire FWWPP at
    // every later boundary target whose adjacent left segment is not
    // terminal. English tail completion fires if the suffix contains
    // any non-terminal segment.
    if (expansion && p > 0) {
        for (std::size_t s_idx = 0; s_idx + 1 < letter_bounds.size(); ++s_idx) {
            std::size_t s = letter_bounds[s_idx];
            if (s >= p || start[s] == '\'') continue;

            bool saw_incomplete = false;
            for (std::size_t bi = s_idx + 1; bi < letter_bounds.size(); ++bi) {
                if (!letter_is_syllable[bi - 1]) saw_incomplete = true;
                if (!saw_incomplete) continue;
                std::size_t target = letter_bounds[bi];
                auto prefix = start.substr(s, target - s);
                expand_dat(s, Dict::LetterPinyin, prefix, target, 512, true, false);
            }
            if (saw_incomplete) {
                std::string_view suffix = start.substr(s);
                expand_dat(s, Dict::LetterEn, suffix, p, 1024, false, true);
            }
        }
    }

    // Digit-side boundary expansion (T9 mirror of letter-side). For each
    // digit boundary s, walk AdvanceT9States forward; latch on first
    // non-terminal segment; emit completions at every later boundary
    // target. Drops the old per-column outer loop and 8-char hard cap.
    if (expansion && d > 0) {
        for (std::size_t s_bi = 0; s_bi + 1 < digit_bounds.size(); ++s_bi) {
            std::size_t s_dpos = digit_bounds[s_bi];
            std::size_t s = p + s_dpos;
            if (s_dpos >= d || nums[s_dpos] == '\'') continue;

            auto py_states = dict_.Dat(Dict::LetterPinyin).StartPinyinStates();
            auto en_states = dict_.Dat(Dict::LetterEn).StartPinyinStates();
            bool saw_incomplete = false;
            std::size_t next_bi = s_bi + 1;

            for (std::size_t pos = s_dpos; pos < d; ++pos) {
                auto ch = static_cast<uint8_t>(nums[pos]);
                if (!py_states.empty()) {
                    dict_.Dat(Dict::LetterPinyin).AdvanceT9States(
                        py_states, ch, Dict::NumToLettersLower);
                }
                if (!en_states.empty()) {
                    dict_.Dat(Dict::LetterEn).AdvanceT9States(
                        en_states, ch, Dict::NumToLetters);
                }
                if (py_states.empty() && en_states.empty()) break;

                std::size_t target_dpos = pos + 1;
                if (next_bi >= digit_bounds.size()
                    || digit_bounds[next_bi] != target_dpos) continue;

                std::size_t seg_idx = next_bi - 1;
                if (!digit_is_syllable[seg_idx]) saw_incomplete = true;
                ++next_bi;
                if (!saw_incomplete) continue;

                std::size_t k = target_dpos - s_dpos;
                auto emit_cn = [&](const std::vector<trie::DoubleArray::PinyinState>& sts) {
                    auto comps = dict_.Dat(Dict::LetterPinyin)
                        .CollectCompletionsPinyin(sts, k, 1024, /*stop_at_sep=*/true);
                    for (const auto& r : comps) {
                        auto entry = dict_.GetEntry(Dict::LetterPinyin, r.value);
                        for (uint32_t i = 0; i < entry.count; ++i) {
                            net[s].es.push_back({s, s + k, entry.items[i].id,
                                                 entry.items[i].pieces, 0, true,
                                                 false});
                        }
                    }
                };
                auto emit_en = [&](const std::vector<trie::DoubleArray::PinyinState>& sts) {
                    auto comps = dict_.Dat(Dict::LetterEn)
                        .CollectCompletionsPinyin(sts, k, 1024, /*stop_at_sep=*/true);
                    for (const auto& r : comps) {
                        if (r.length <= k) continue;
                        auto entry = dict_.GetEntry(Dict::LetterEn, r.value);
                        for (uint32_t i = 0; i < entry.count; ++i) {
                            net[s].es.push_back({s, s + k, entry.items[i].id,
                                                 entry.items[i].pieces, 0, true,
                                                 true});
                        }
                    }
                };
                emit_cn(py_states);
                emit_en(en_states);
            }
        }
    }

    // Additive T9 key-typo correction. Fires ONLY on digit spans that are
    // neither a known T9 syllable nor an extendable prefix (i.e. genuine
    // dead-end mistypes, not valid input or abbreviations). A bounded DFS
    // enumerates corrected syllable sequences (<=3) over the ORIGINAL digit
    // span; corrected syllables use length-preserving keypad edits that map
    // back to a known T9 syllable, uncorrected ones keep the original digits.
    // The digit key is resolved by PrefixSearchT9. Every emitted edge carries a
    // correction cost, so the exact path (correction=0) is never displaced.
    // Fuzzy-on-valid is deliberately NOT used here: T9 digit ambiguity would
    // emit longer-span corrected words that win Layer-2 via distance penalty
    // and regress correct input.
    if (t9_correction_enabled_ && enable_typo_correction_ && d > 0) {
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
            std::string_view span = nums.substr(pos, len);
            sc.known = Dict::IsKnownT9Syllable(span);
            if (!sc.known && len >= 2 && !Dict::IsExtendableT9Syllable(span)) {
                sc.variants = pinyin_correct::T9TypoVariants(span);
            }
            return ccache.emplace(key, std::move(sc)).first->second;
        };

        constexpr int kMaxSyl = 3;
        constexpr int kLookupBudget = 32;
        std::vector<std::string> parts;
        parts.reserve(kMaxSyl);

        auto dfs = [&](auto&& self, std::size_t span_start, std::size_t pos,
                       int nsyl, int ncorrected, float_t cost,
                       int& budget) -> void {
            const bool emit_here = nsyl >= 1 && ncorrected >= 1 && budget > 0;
            if (emit_here) {
                std::string key;
                for (int i = 0; i < nsyl; ++i) {
                    if (i) key.push_back('\'');
                    key += parts[static_cast<std::size_t>(i)];
                }
                --budget;
                auto results = py_dat.PrefixSearchT9(key, Dict::NumToLettersLower, 8);
                for (const auto& r : results) {
                    if (r.length != key.size()) continue;
                    auto entry = dict_.GetEntry(Dict::LetterPinyin, r.value);
                    for (uint32_t i = 0; i < entry.count; ++i) {
                        net[p + span_start].es.push_back(
                            {p + span_start, p + pos, entry.items[i].id,
                             entry.items[i].pieces, 0, false, false, cost});
                    }
                }
            }
            if (nsyl >= kMaxSyl || budget <= 0) return;
            if (ncorrected == 0 && nsyl >= 2) return;
            for (std::size_t len = 1; len <= 6 && pos + len <= d; ++len) {
                if (nums[pos + len - 1] == '\'') break;
                const SylCands& sc = cands_at(pos, len);
                if (sc.known) {
                    parts.push_back(std::string(nums.substr(pos, len)));
                    self(self, span_start, pos + len, nsyl + 1, ncorrected,
                         cost, budget);
                    parts.pop_back();
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

        for (std::size_t s_dpos : digit_bounds) {
            if (s_dpos >= d || nums[s_dpos] == '\'') continue;
            int budget = kLookupBudget;
            dfs(dfs, s_dpos, s_dpos, 0, 0, 0.0f, budget);
        }
    }

    // Two-track per-bucket tier filter: CN edges (!english) and English
    // edges (english) are filtered on independent tracks, so an exact
    // edge on one track doesn't suppress edges on the other.
    //   tier 0: !expansion (exact)
    //   tier 1: expansion (abbrev / completion)
    // Keeps both CN exact and English exact alive when they share a
    // bucket (e.g. 466453 → 攻克 + google), and keeps CN expansion
    // alongside English exact (e.g. 539 → 可以 + key).
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