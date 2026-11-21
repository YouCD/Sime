#include "sime.h"

#include "sime_detail.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sime {

namespace {

bool BetterLayer2Entry(const DecodeResult& lhs, const DecodeResult& rhs) {
    // Dedup tie-break: pure score comparison. Penalty is already in score
    // so the cleaner-provenance entry (exact emit, no expansion penalty)
    // naturally wins.
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    if (lhs.cnt != rhs.cnt) return lhs.cnt > rhs.cnt;
    return lhs.units.size() > rhs.units.size();
}

void PushBestLayer2Entry(std::vector<DecodeResult>& best_entries,
                         std::unordered_map<std::string, std::size_t>& index_by_text,
                         DecodeResult candidate) {
    auto it = index_by_text.find(candidate.text);
    if (it == index_by_text.end()) {
        index_by_text.emplace(candidate.text, best_entries.size());
        best_entries.push_back(std::move(candidate));
        return;
    }
    auto& current = best_entries[it->second];
    if (BetterLayer2Entry(candidate, current)) {
        current = std::move(candidate);
    }
}

} // namespace

std::vector<DecodeResult> Sime::DecodeNumSentence(
    std::string_view nums,
    std::string_view start,
    std::size_t extra) const {
    return DecodeNumSentence(nums, start, {}, extra);
}

std::vector<DecodeResult> Sime::DecodeNumSentence(
    std::string_view nums,
    std::string_view start,
    const std::vector<TokenID>& context,
    std::size_t extra) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_) return results;
    MaybeTrimCaches();
    for (char c : nums) {
        if (c == '\'') continue;
        if (c < '2' || c > '9') return results;
    }
    if (nums.empty() && start.empty()) return results;

    const std::size_t p = start.size();
    const std::size_t d = nums.size();
    const std::size_t total = p + d;
    std::string combined_input(start);
    combined_input += nums;

    std::vector<Node> net;
    InitNumNet(start, nums, net, /*expansion=*/true);
    ComputeEdgePenalties(net, combined_input);
    for (auto& col : net) PruneNode(col.es);

    for (auto& col : net) col.states.SetMaxTop(BeamSize);
    State init = InitialState(context);
    net[0].states.Insert(init);
    Process(net);

    // See DecodeSentence: Layer 1 dedups with a local set so beam members
    // that don't make the 1+extra cut remain eligible for Layer 2.
    std::unordered_set<std::string> dedup;
    const float_t penalty_per_unit =
        std::abs(scorer_.UnknownPenalty()) / DistancePenalty;

    // === Layer 1: Full sentence N-best ===
    // (frag_penalty is already in beam scores via edge.penalty)
    {
        const auto tail = net[total + 1].states.GetStates();
        const std::size_t scan = std::min<std::size_t>(BeamSize, tail.size());
        const std::size_t full_cnt = start.size() + d;
        std::vector<DecodeResult> l1;
        l1.reserve(scan);
        std::unordered_set<std::string> l1_seen;
        for (std::size_t rank = 0; rank < scan; ++rank) {
            auto path = Backtrace(tail[rank], total + 1);
            std::string text = ExtractText(path, combined_input);
            if (text.empty() || !l1_seen.insert(text).second) continue;
            std::string py = ExtractUnits(path, combined_input);

            l1.push_back({std::move(text), std::move(py),
                          ExtractTokens(path),
                          -tail[rank].score, full_cnt});
        }
        std::sort(l1.begin(), l1.end(),
                  [](const DecodeResult& a, const DecodeResult& b) {
                      return a.score > b.score;
                  });
        detail::RerankWithGru(l1, gru_.get(), true, gru_t9_scale_,
                              adaptive_gru_enabled_, full_cnt);
        const std::size_t full_limit = 1 + extra;
        for (std::size_t i = 0; i < l1.size() && results.size() < full_limit;
             ++i) {
            dedup.insert(l1[i].text);
            results.push_back(std::move(l1[i]));
        }
    }

    // === Layer 2: unigram alternatives at column 0 ===
    // All entries (exact + expansion) compete on score; penalty is
    // already in the score so no separate exact/abbrev tiering.
    const std::size_t l2_start = results.size();
    std::vector<DecodeResult> best_l2;
    std::unordered_map<std::string, std::size_t> l2_index_by_text;

    // Skip leading apostrophe-only columns (those carry only a NotToken
    // sentinel edge). Without this, an input that starts with `'` (e.g.
    // a leftover after partial commit) leaves Layer 2 with zero candidates.
    std::size_t l2_col = 0;
    while (l2_col < total && net[l2_col].es.size() == 1 &&
           net[l2_col].es[0].id == NotToken) {
        ++l2_col;
    }
    for (const auto& edge : net[l2_col].es) {
        if (edge.id == NotToken) continue;

        std::u32string text_u32 = ToText(edge);
        if (text_u32.empty()) continue;

        std::string text_utf8 = TextFromU32(text_u32);
        if (dedup.contains(text_utf8)) continue;

        std::size_t distance = (total > edge.end) ? (total - edge.end) : 0;
        float_t dist_penalty =
            static_cast<float_t>(distance) * penalty_per_unit;

        auto slice = std::string_view(combined_input).substr(
            edge.start, edge.end - edge.start);

        // Use edge.penalty (English + expansion, set by
        // ComputeEdgePenalties) so L2 score sits on the same coordinate
        // system as L1: -(LM unigram + edge.penalty) - dist_penalty.
        Scorer::Pos epos{};
        Scorer::Pos enext{};
        float_t score = -scorer_.ScoreMove(epos, edge.id, enext)
                        - dist_penalty
                        - edge.penalty;

        std::string edge_py = edge.pieces
            ? AbbreviatePieces(edge.pieces, slice)
            : "";
        std::size_t cnt = edge.end;
        PushBestLayer2Entry(
            best_l2,
            l2_index_by_text,
            {std::move(text_utf8), std::move(edge_py),
             ExtractTokens({edge}), score, cnt});
    }

    for (auto& entry : best_l2) {
        results.push_back(std::move(entry));
    }
    auto by_score = [](const DecodeResult& a, const DecodeResult& b) {
        return a.score > b.score;
    };
    // Sort the L2 tail (everything after the L1 sentence head) by score —
    // exact and expansion entries compete on equal footing.
    std::sort(results.begin() + l2_start, results.end(), by_score);

    return results;
}

std::vector<DecodeResult> Sime::DecodeNumStr(
    std::string_view nums,
    std::string_view start,
    std::size_t num) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_) return results;
    MaybeTrimCaches();
    for (char c : nums) {
        if (c == '\'') continue;
        if (c < '2' || c > '9') return results;
    }
    if (nums.empty() && start.empty()) return results;

    const std::size_t max_top = num == 0 ? 1 : num;
    const std::size_t total = start.size() + nums.size();
    std::string combined_input(start);
    combined_input += nums;

    std::vector<Node> net;
    InitNumNet(start, nums, net, /*expansion=*/false);
    ComputeEdgePenalties(net, combined_input);
    for (auto& col : net) PruneNode(col.es);

    for (auto& col : net) col.states.SetMaxTop(BeamSize);
    State init = InitialState();
    net[0].states.Insert(init);
    Process(net);

    auto tail_states = net.back().states.GetStates();
    std::unordered_set<std::string> dedup;
    for (std::size_t rank = 0;
         rank < tail_states.size() && results.size() < max_top; ++rank) {
        auto path = Backtrace(tail_states[rank], total + 1);
        std::string text = ExtractText(path, combined_input);
        if (text.empty() || !dedup.insert(text).second) continue;
        std::string py = ExtractUnits(path, combined_input);
        results.push_back({std::move(text), std::move(py),
                           ExtractTokens(path),
                           -tail_states[rank].score,
                           start.size() + nums.size()});
    }

    return results;
}

std::vector<DecodeResult> Sime::DecodeStr(
    std::string_view input,
    std::size_t num) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_ || input.empty()) return results;
    MaybeTrimCaches();

    std::string lower = detail::NormalizeInput(input);
    if (lower.empty()) return results;

    std::vector<Node> net;
    InitNet(lower, net, /*expansion=*/false);
    ComputeEdgePenalties(net, lower);
    for (auto& col : net) PruneNode(col.es);

    const std::size_t max_top = num == 0 ? 1 : num;
    for (auto& col : net) col.states.SetMaxTop(BeamSize);
    State init = InitialState();
    net[0].states.Insert(init);
    Process(net);

    const auto tail_states = net.back().states.GetStates();
    std::unordered_set<std::string> dedup;
    for (std::size_t rank = 0;
         rank < tail_states.size() && results.size() < max_top; ++rank) {
        auto path = Backtrace(tail_states[rank], net.size() - 1);
        if (path.empty()) continue;
        std::string text = ExtractText(path, lower);
        if (text.empty() || !dedup.insert(text).second) continue;
        std::string py = ExtractUnits(path, lower);
        results.push_back({std::move(text), std::move(py),
                           ExtractTokens(path),
                           -tail_states[rank].score, input.size()});
    }
    return results;
}

std::vector<DecodeResult> Sime::DecodeSentence(
    std::string_view input,
    std::size_t extra) const {
    return DecodeSentence(input, {}, extra);
}

std::vector<DecodeResult> Sime::DecodeSentence(
    std::string_view input,
    const std::vector<TokenID>& context,
    std::size_t extra) const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    std::vector<DecodeResult> results;
    if (!ready_ || input.empty()) return results;
    MaybeTrimCaches();

    std::string lower = detail::NormalizeInput(input);
    if (lower.empty()) return results;

    const std::size_t total = lower.size();

    std::vector<Node> net;
    InitNet(lower, net, /*expansion=*/true);
    ComputeEdgePenalties(net, lower);
    for (auto& col : net) PruneNode(col.es);

    for (auto& col : net) col.states.SetMaxTop(BeamSize);
    State init = InitialState(context);
    net[0].states.Insert(init);
    Process(net);

    // `dedup` tracks texts already emitted in `results`; Layer 2 checks
    // against it to avoid duplicating a candidate that made it into Layer 1.
    // Layer 1 uses its own local dedup during beam scan so that beam members
    // that don't make the final 1+extra cut remain eligible for Layer 2.
    std::unordered_set<std::string> dedup;
    const float_t penalty_per_unit =
        std::abs(scorer_.UnknownPenalty()) / DistancePenalty;

    // === Layer 1: Full sentence N-best ===
    // Scan the whole beam, sort by score, take top (1 + extra).
    // (penalties are already in beam scores via edge.penalty)
    {
        const auto tail = net.back().states.GetStates();
        const std::size_t scan =
            std::min<std::size_t>(BeamSize, tail.size());
        std::vector<DecodeResult> l1;
        l1.reserve(scan);
        std::unordered_set<std::string> l1_seen;
        for (std::size_t rank = 0; rank < scan; ++rank) {
            auto path = Backtrace(tail[rank], net.size() - 1);
            if (path.empty()) continue;
            std::string text = ExtractText(path, lower);
            if (text.empty() || !l1_seen.insert(text).second) continue;
            std::string py = ExtractUnits(path, lower);

            l1.push_back({std::move(text), std::move(py),
                          ExtractTokens(path),
                          -tail[rank].score,
                          input.size()});
        }
        std::sort(l1.begin(), l1.end(),
                  [](const DecodeResult& a, const DecodeResult& b) {
                      return a.score > b.score;
                  });
        detail::RerankWithGru(l1, gru_.get(), false, gru_pinyin_scale_,
                              adaptive_gru_enabled_, input.size());
        const std::size_t full_limit = 1 + extra;
        for (std::size_t i = 0; i < l1.size() && results.size() < full_limit;
             ++i) {
            dedup.insert(l1[i].text);
            results.push_back(std::move(l1[i]));
        }
    }

    // === Layer 2: word/char alternatives at position 0 ===
    // All entries (exact + expansion) compete on score; penalty is
    // already in the score so no separate exact/abbrev tiering.
    const std::size_t l2_start = results.size();
    std::vector<DecodeResult> best_l2;
    std::unordered_map<std::string, std::size_t> l2_index_by_text;

    // Skip leading apostrophe-only columns (see DecodeNumSentence).
    std::size_t l2_col = 0;
    while (l2_col < total && net[l2_col].es.size() == 1 &&
           net[l2_col].es[0].id == NotToken) {
        ++l2_col;
    }
    for (const auto& edge : net[l2_col].es) {
        if (edge.id == NotToken) continue;

        std::u32string text_u32 = ToText(edge);
        if (text_u32.empty()) continue;
        std::string text_utf8 = TextFromU32(text_u32);
        if (dedup.contains(text_utf8)) continue;

        std::size_t distance = (total > edge.end) ? (total - edge.end) : 0;
        float_t dist_penalty =
            static_cast<float_t>(distance) * penalty_per_unit;

        auto slice = std::string_view(lower).substr(
            edge.start, edge.end - edge.start);

        // Use edge.penalty (English + expansion) so L2 score is on the
        // same coordinate system as L1: -(LM unigram + edge.penalty)
        // - dist_penalty.
        Scorer::Pos epos{};
        Scorer::Pos enext{};
        float_t score = -scorer_.ScoreMove(epos, edge.id, enext)
                        - dist_penalty
                        - edge.penalty;

        std::string edge_py = edge.pieces
            ? AbbreviatePieces(edge.pieces, slice)
            : "";
        PushBestLayer2Entry(
            best_l2,
            l2_index_by_text,
            {std::move(text_utf8), std::move(edge_py),
             ExtractTokens({edge}), score, edge.end});
    }

    for (auto& entry : best_l2) {
        results.push_back(std::move(entry));
    }
    auto by_score = [](const DecodeResult& a, const DecodeResult& b) {
        return a.score > b.score;
    };
    std::sort(results.begin() + l2_start, results.end(), by_score);

    return results;
}

} // namespace sime