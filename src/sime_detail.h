#pragma once

#include "sime.h"

#include <string>
#include <string_view>
#include <vector>

namespace sime {

class GruReranker;

// Helpers shared across the split Sime translation units. These were file-local
// (anonymous namespace) in the original monolithic sime.cc; moving them here
// gives them external linkage so the lattice / decode / penalty modules can
// call them without duplicating definitions.
namespace detail {

// Lowercase letters + '\'' kept; space -> ▁ (U+2581). See sime_penalty.cc.
std::string NormalizeInput(std::string_view input);

// Curated common short English command / tech word membership. sime_english.cc.
bool IsCommonEnglishWord(std::string_view word);

// Two-letter shell / dev command membership (wc / ps / sh / jq ...).
// sime_english.cc.
bool IsTwoLetterCommand(std::string_view word);

// Penalty for a mid-sentence (non-full-cover) English span. `has_cn_rival`
// counts any CN-track edge sharing the span; `has_exact_cn_rival` counts only
// non-expansion ones. `light` / `heavy` / `bonus` are the configured tiers.
// sime_english.cc.
float_t ComputeEnglishPenalty(std::string_view span, bool has_cn_rival,
                              bool has_exact_cn_rival, bool whitelist_on,
                              float_t light, float_t heavy, float_t bonus);

// Heuristic adaptive GRU scale from the top-candidate score spread.
// sime_gru.cc.
float_t AdaptiveGruScale(float_t base, float_t spread, std::size_t n,
                         std::size_t input_len);

// GRU rerank of Layer-1 candidates. `scale` is the base domain bias
// (full-pinyin / T9); when `adaptive` is set it is modulated by the candidate
// score spread over `input_len`. sime_gru.cc.
void RerankWithGru(std::vector<DecodeResult>& candidates,
                   const GruReranker* reranker, bool t9, float_t scale,
                   bool adaptive, std::size_t input_len);

} // namespace detail
} // namespace sime