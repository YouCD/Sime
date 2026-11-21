#include "sime.h"

#include "sime_detail.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace sime {

void Sime::ComputeEdgePenalties(std::vector<Node>& net,
                                std::string_view input) const {
    // English penalty tiers for full-cover English edges:
    //   1. curated common short command (≤4, kCommonEnglish) →
    //      EnglishWhitelistPenalty (bonus): confident dev intent, must
    //      beat CN 简拼 / 同音 rivals (awk→爱玩客, bash→巴士, code→从的).
    //   2. length > 3 (google / hello / grep) → EnglishPenalty (light).
    //   3. everything else (partial-cover zg in zgkxjsdx, or ≤3
    //      full-cover up / nm / key / men) → EnglishPenaltyShort (heavy):
    //      likely coincidental English spellings of CN abbreviations or
    //      pinyin syllables.
    // Expansion penalty: short input gets ExpansionPenaltyShort (lets
    // 87→他说 beat short English exact); long input keeps stricter
    // ExpansionPenalty so abbrev edges don't crowd out the natural
    // full-pinyin path.
    const bool short_input = input.size() <= 4;
    const bool short_english = input.size() <= 3;
    const float_t expansion_penalty = short_input
        ? ExpansionPenaltyShort : ExpansionPenalty;

    // Mixed-script input (a typed space → ▁, or an apostrophe word boundary).
    // Pre-collect CN-track edge spans so a mid-sentence English span that shares
    // start/end with a Chinese reading keeps the heavy penalty (women / song
    // rival guard). Apostrophe-delimited pure pinyin (xi'an → 西安) has no
    // English edges, so the mid-sentence branch never fires for it.
    const bool has_boundary =
        input.find(static_cast<char>(0xE2)) != std::string_view::npos ||
        input.find('\'') != std::string_view::npos;
    std::unordered_set<std::uint64_t> cn_spans;
    std::unordered_set<std::uint64_t> cn_spans_exact;
    if (has_boundary) {
        for (const auto& col : net) {
            for (const auto& e : col.es) {
                if (e.english || e.id == NotToken) continue;
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(e.start) << 32) |
                    static_cast<std::uint64_t>(e.end);
                cn_spans.insert(key);
                if (!e.expansion) cn_spans_exact.insert(key);
            }
        }
    }

    for (auto& col : net) {
        for (auto& edge : col.es) {
            if (!edge.pieces || edge.id == NotToken) continue;
            float_t en_pen = 0.0f;
            if (edge.english) {
                bool full_cover =
                    edge.start == 0 && edge.end == input.size();
                if (full_cover) {
                    if (input.size() <= 4 && english_whitelist_enabled_
                        && detail::IsCommonEnglishWord(input)) {
                        en_pen = english_whitelist_penalty_;
                    } else if (input.size() == 2 && english_whitelist_enabled_
                               && detail::IsTwoLetterCommand(input)) {
                        // Gentle lift for 2-letter commands outside the
                        // strong whitelist (ps / cp / mv / rm / vi).
                        en_pen = english_penalty_;
                    } else {
                        en_pen = short_english ? english_penalty_short_
                                               : english_penalty_;
                    }
                } else if (has_boundary) {
                    std::string_view span =
                        input.substr(edge.start, edge.end - edge.start);
                    const std::uint64_t key =
                        (static_cast<std::uint64_t>(edge.start) << 32) |
                        static_cast<std::uint64_t>(edge.end);
                    en_pen = detail::ComputeEnglishPenalty(
                        span, cn_spans.contains(key),
                        cn_spans_exact.contains(key),
                        english_whitelist_enabled_, english_penalty_,
                        english_penalty_short_, english_whitelist_penalty_);
                } else {
                    en_pen = english_penalty_short_;
                }
            }
            edge.penalty = en_pen
                + (edge.expansion ? expansion_penalty : 0.0f)
                + edge.correction;
        }
    }
}

namespace detail {

std::string NormalizeInput(std::string_view input) {
    std::string lower;
    lower.reserve(input.size());
    for (char c : input) {
        if (c == '\'') {
            lower.push_back(c);
        } else if (c == ' ') {
            // Space → ▁ (U+2581, UTF-8: E2 96 81)
            lower.push_back(static_cast<char>(0xE2));
            lower.push_back(static_cast<char>(0x96));
            lower.push_back(static_cast<char>(0x81));
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            lower.push_back(c);
        }
    }
    return lower;
}

} // namespace detail
} // namespace sime