#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "sime_detail.h"

using sime::detail::AdaptiveGruScale;
using sime::detail::ComputeEnglishPenalty;
using sime::detail::IsCommonEnglishWord;
using sime::detail::IsTwoLetterCommand;

namespace {
constexpr sime::float_t kLight = 2.5;
constexpr sime::float_t kHeavy = 8.0;
constexpr sime::float_t kBonus = -3.0;

sime::float_t Pen(std::string_view span, bool any_rival, bool exact_rival,
                  bool whitelist_on = true) {
    return ComputeEnglishPenalty(span, any_rival, exact_rival, whitelist_on,
                                 kLight, kHeavy, kBonus);
}
}  // namespace

TEST_CASE("IsTwoLetterCommand hits curated shell commands") {
    for (const char* w : {"wc", "ps", "sh", "jq", "ls", "cd", "cp", "mv",
                          "rm", "dd", "df", "ln", "vi"}) {
        CAPTURE(w);
        CHECK(IsTwoLetterCommand(w));
    }
}

TEST_CASE("IsTwoLetterCommand excludes pinyin and non-commands") {
    // du is a full pinyin syllable; nm/us/be/up/key are protected
    // coincidences; anything not exactly 2 letters is out.
    for (const char* w : {"du", "nm", "us", "be", "up", "key", "ping",
                          "a", "", "wcc", "WC", "Ps"}) {
        CAPTURE(w);
        CHECK_FALSE(IsTwoLetterCommand(w));
    }
}

TEST_CASE("IsCommonEnglishWord still guards pinyin syllables") {
    CHECK(IsCommonEnglishWord("grep"));
    CHECK(IsCommonEnglishWord("wc"));
    CHECK_FALSE(IsCommonEnglishWord("ping"));
    CHECK_FALSE(IsCommonEnglishWord("du"));
    CHECK_FALSE(IsCommonEnglishWord("women"));
}

TEST_CASE("ComputeEnglishPenalty: whitelist words ignore expansion rivals") {
    // grep: no rival → bonus; expansion-only rival (热盘) → still bonus;
    // exact CN rival → heavy.
    CHECK(Pen("grep", false, false) == kBonus);
    CHECK(Pen("grep", true, false) == kBonus);
    CHECK(Pen("grep", true, true) == kHeavy);
    // Whitelist disabled drops the bonus to the length-based tier.
    CHECK(Pen("grep", false, false, false) == kLight);
}

TEST_CASE("ComputeEnglishPenalty: pinyin / CN rivals keep heavy penalty") {
    // song is a known pinyin syllable even with no rival edge.
    CHECK(Pen("song", false, false) == kHeavy);
    // women is not a single syllable but has a CN reading (我们).
    CHECK(Pen("women", true, true) == kHeavy);
    CHECK(Pen("women", true, false) == kHeavy);
    // Non-whitelist long word with no rival relaxes to light.
    CHECK(Pen("hello", false, false) == kLight);
    // Short non-whitelist span stays heavy.
    CHECK(Pen("ab", false, false) == kHeavy);
}

TEST_CASE("AdaptiveGruScale bounds and damping") {
    // Single candidate or very short input keeps base.
    CHECK(AdaptiveGruScale(0.4, 0.0, 1, 10) == doctest::Approx(0.4));
    CHECK(AdaptiveGruScale(0.4, 0.0, 5, 2) == doctest::Approx(0.4));
    // spread == 2.0 is the neutral point (multiplier 1.0).
    CHECK(AdaptiveGruScale(0.4, 2.0, 5, 10) == doctest::Approx(0.4));
    // Tight spread (0) leans harder: multiplier 1.2.
    CHECK(AdaptiveGruScale(0.4, 0.0, 5, 10) == doctest::Approx(0.48));
    // Wide spread clamps the multiplier at 0.5.
    CHECK(AdaptiveGruScale(0.4, 10.0, 5, 10) == doctest::Approx(0.2));
    // Negative spread clamps the multiplier at 1.5.
    CHECK(AdaptiveGruScale(0.4, -20.0, 5, 10) == doctest::Approx(0.6));
    // Monotonic: tighter spread never lowers the scale.
    CHECK(AdaptiveGruScale(0.5, 1.0, 5, 10) >
          AdaptiveGruScale(0.5, 3.0, 5, 10));
}