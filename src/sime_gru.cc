#include "sime_detail.h"

#include "gru_reranker.h"

#include <algorithm>
#include <vector>

namespace sime::detail {

namespace {

constexpr std::size_t kGruCandidates = 10;

} // namespace

float_t AdaptiveGruScale(float_t base, float_t spread, std::size_t n,
                         std::size_t input_len) {
    // Heuristic domain bias: a tight score spread among the top candidates
    // signals LM ambiguity, so lean a little harder on the GRU; a wide spread
    // means the LM is confident, so ease off. Clamped to ±50% of base because
    // edge penalties (English / expansion) inflate spread independently of LM
    // ambiguity. Single-candidate and very short inputs keep the base scale.
    if (n <= 1 || input_len <= 2) return base;
    const float_t mult = std::clamp(1.0 + 0.1 * (2.0 - spread), 0.5, 1.5);
    return base * mult;
}

void RerankWithGru(std::vector<DecodeResult>& candidates,
                   const GruReranker* reranker,
                   bool t9, float_t scale, bool adaptive,
                   std::size_t input_len) {
    if (!reranker || !reranker->Ready()) return;
    if (adaptive && candidates.size() > 1) {
        const std::size_t k = std::min<std::size_t>(kGruCandidates, candidates.size());
        const float_t spread = candidates[0].score - candidates[k - 1].score;
        scale = AdaptiveGruScale(scale, spread, candidates.size(), input_len);
    }
    if (candidates.size() > kGruCandidates) {
        candidates.resize(kGruCandidates);
    }
    for (auto& candidate : candidates) {
        candidate.score += scale * reranker->Score(candidate.tokens, t9);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const DecodeResult& lhs, const DecodeResult& rhs) {
            return lhs.score > rhs.score;
        });
}

} // namespace sime::detail