#include "trie.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_set>

namespace trie {
void DoubleArray::AdvancePinyin(std::vector<PinyinState>& states,
                                uint8_t ch,
                                uint8_t max_skip_depth) const {
    auto canSkip = [&](const PinyinState& s) -> bool {
        return s.depth >= 1 && s.depth <= max_skip_depth;
    };

    static thread_local std::vector<PinyinState> next;
    next.clear();

    for (const auto& s : states) {
        uint8_t new_depth = (s.depth < 255) ? static_cast<uint8_t>(s.depth + 1) : 255;

        if (ch == static_cast<uint8_t>('\'')) {
            // Separator in input
            // 1. Direct '\'' match
            std::size_t sep = TryChild(s.pos, ch);
            if (sep != SIZE_MAX) {
                next.push_back({sep, 0, s.fuzzy});
            }
            // 2. Syllable skip to '\''
            if (canSkip(s)) {
                for (auto sp : GetSepDescendants(s.pos)) {
                    next.push_back({sp, 0, true});
                }
            }
        } else {
            // Normal character
            // 1. Direct match
            std::size_t child = TryChild(s.pos, ch);
            if (child != SIZE_MAX) {
                next.push_back({child, new_depth, s.fuzzy});
            }
            // 2. Skip '\'' then match (for non-separator input crossing boundary)
            std::size_t sep = TryChild(s.pos, static_cast<uint8_t>('\''));
            if (sep != SIZE_MAX) {
                std::size_t after_sep = TryChild(sep, ch);
                if (after_sep != SIZE_MAX) {
                    next.push_back({after_sep, 1, s.fuzzy});  // new syllable, first char
                }
            }
            // 3. Syllable skip
            if (canSkip(s)) {
                for (auto sp : GetSepDescendants(s.pos)) {
                    std::size_t after = TryChild(sp, ch);
                    if (after != SIZE_MAX) {
                        next.push_back({after, 1, true});  // new syllable
                    }
                }
            }
        }
    }

    // Deduplicate by pos (prefer exact path, then smallest depth)
    std::sort(next.begin(), next.end(),
              [](const PinyinState& a, const PinyinState& b) {
                  if (a.pos != b.pos) return a.pos < b.pos;
                  if (a.fuzzy != b.fuzzy) return !a.fuzzy && b.fuzzy;
                  return a.depth < b.depth;
              });
    next.erase(std::unique(next.begin(), next.end(),
                           [](const PinyinState& a, const PinyinState& b) {
                               return a.pos == b.pos;
                           }), next.end());

    // Cap state count to prevent combinatorial explosion (T9)
    constexpr std::size_t MaxStates = 2048;
    if (next.size() > MaxStates) next.resize(MaxStates);

    states.swap(next);
}

void DoubleArray::AdvanceT9(std::vector<PinyinState>& states,
                            const char* letters) const {
    auto canSkip = [&](const PinyinState& s) -> bool {
        return s.depth == 1;
    };

    // Reuse buffer across calls; ~2.8M calls/run → eliminates that many
    // small heap allocations.
    static thread_local std::vector<PinyinState> next;
    next.clear();

    for (const auto& s : states) {
        // Pre-compute shared state: separator child and sep descendants
        std::size_t sep_child = TryChild(s.pos, static_cast<uint8_t>('\''));
        bool need_skip = canSkip(s);
        const std::vector<std::size_t>* seps = nullptr;
        if (need_skip) {
            seps = &GetSepDescendants(s.pos);
        }

        for (const char* lp = letters; *lp; ++lp) {
            uint8_t ch = static_cast<uint8_t>(*lp);
            uint8_t new_depth = (s.depth < 255)
                ? static_cast<uint8_t>(s.depth + 1) : 255;

            // 1. Direct match
            std::size_t child = TryChild(s.pos, ch);
            if (child != SIZE_MAX) {
                next.push_back({child, new_depth, s.fuzzy});
            }
            // 2. Skip separator then match
            if (sep_child != SIZE_MAX) {
                std::size_t after_sep = TryChild(sep_child, ch);
                if (after_sep != SIZE_MAX) {
                    next.push_back({after_sep, 1, s.fuzzy});
                }
            }
            // 3. Syllable skip
            if (need_skip) {
                for (auto sp : *seps) {
                    std::size_t after = TryChild(sp, ch);
                    if (after != SIZE_MAX) {
                        next.push_back({after, 1, true});
                    }
                }
            }
        }
    }

    // Deduplicate by pos (prefer exact path, then smallest depth)
    std::sort(next.begin(), next.end(),
              [](const PinyinState& a, const PinyinState& b) {
                  if (a.pos != b.pos) return a.pos < b.pos;
                  if (a.fuzzy != b.fuzzy) return !a.fuzzy && b.fuzzy;
                  return a.depth < b.depth;
              });
    next.erase(std::unique(next.begin(), next.end(),
                           [](const PinyinState& a, const PinyinState& b) {
                               return a.pos == b.pos;
                           }), next.end());

    constexpr std::size_t MaxT9States = 512;
    if (next.size() > MaxT9States) next.resize(MaxT9States);

    states.swap(next);  // states gets new content; next keeps capacity for reuse
}

std::vector<SearchResult> DoubleArray::PrefixSearchPinyin(
    std::string_view str, std::size_t max_num,
    uint8_t max_skip_depth) const {
    // Exact-only prefix scan. Fuzzy abbreviation completions (last-syllable
    // DFS, syllable-skip) are delegated to FindWordsWithPrefixPinyin called
    // by InitNet at segment boundaries — so here we drop fuzzy states after
    // each AdvancePinyin step to keep this path cheap.
    std::vector<SearchResult> results;
    if (Empty() || max_num == 0) return results;

    std::vector<PinyinState> states = {{0, 0, false}};
    std::unordered_set<uint64_t> seen;

    auto record_matches = [&](std::size_t input_len) {
        for (const auto& s : states) {
            if (s.pos >= size_ || !array_[s.pos].eow) continue;
            std::size_t vp = s.pos ^ array_[s.pos].index;
            if (vp >= size_ || !array_[vp].HasValue()) continue;
            uint32_t val = static_cast<uint32_t>(array_[vp].value);
            uint64_t key = (static_cast<uint64_t>(val) << 32)
                         | static_cast<uint64_t>(input_len);
            if (!seen.insert(key).second) continue;
            results.push_back({val, input_len});
            if (results.size() >= max_num) return;
        }
    };

    record_matches(0);

    for (std::size_t i = 0; i < str.size() && !states.empty(); ++i) {
        if (results.size() >= max_num) break;
        AdvancePinyin(states, static_cast<uint8_t>(str[i]), max_skip_depth);
        std::erase_if(states, [](const PinyinState& s) { return s.fuzzy; });
        record_matches(i + 1);
    }
    return results;
}

std::vector<SearchResult> DoubleArray::FindWordsWithPrefixPinyin(
    std::string_view prefix, std::size_t max_num,
    uint8_t max_skip_depth) const {
    std::vector<SearchResult> results;
    if (Empty()) return results;

    // Advance through prefix using pinyin state machine
    std::vector<PinyinState> states = {{0, 0, false}};
    for (std::size_t i = 0; i < prefix.size() && !states.empty(); ++i) {
        AdvancePinyin(states, static_cast<uint8_t>(prefix[i]), max_skip_depth);
    }

    std::unordered_set<uint64_t> seen;
    const std::size_t pool = max_num * 8 + 8;
    auto collect_group = [&](bool fuzzy) {
        for (const auto& s : states) {
            if (results.size() >= max_num) break;
            if (s.fuzzy != fuzzy) continue;
            // Skip states that have just crossed a '\'' boundary (depth=0):
            // expanding here would synthesize letters for a syllable the
            // user never typed (e.g. "q'chuang'" → 起床了 by walking into
            // the 'l','e' children of "qi'chuang'").
            if (s.depth == 0) continue;
            std::vector<SearchResult> local;
            std::string word(prefix);
            CollectWords(s.pos, word, local, pool, /*stop_at_sep=*/true);
            for (const auto& r : local) {
                uint64_t key = (static_cast<uint64_t>(r.value) << 32)
                             | static_cast<uint64_t>(r.length);
                if (!seen.insert(key).second) continue;
                results.push_back(r);
                if (results.size() >= max_num) break;
            }
        }
    };

    collect_group(false);
    collect_group(true);

    return results;
}

} // namespace trie
