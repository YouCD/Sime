#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace sime::pinyin_correct {

// Fuzzy-pronunciation equivalents of a *valid* pinyin syllable: flat/retroflex
// initials (zh/z, ch/c, sh/s), nasal/lateral (n/l), and front/back nasals
// (ang/an, eng/en, ing/in). Returns only known syllables, excluding the input
// itself. Cheap rule-based expansion; used on segments that already parse as
// valid pinyin so the exact path is never displaced.
std::vector<std::string> FuzzyVariants(std::string_view syllable);

// Edit-distance-1 typo corrections of a (usually *invalid*) segment: adjacent
// transposition, QWERTY-adjacent substitution, single deletion/insertion.
// Returns only known syllables, excluding the input itself. Intended for
// segments that do NOT parse as valid pinyin (e.g. "jei" -> "jie"), so correct
// input is not needlessly reinterpreted.
std::vector<std::string> TypoVariants(std::string_view segment);

} // namespace sime::pinyin_correct