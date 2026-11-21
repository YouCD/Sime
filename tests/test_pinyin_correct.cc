#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "pinyin_correct.h"
#include "dict.h"

#include <algorithm>
#include <string>
#include <vector>

using sime::Dict;
namespace pc = sime::pinyin_correct;

static bool Has(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

TEST_CASE("FuzzyVariants flat/retroflex initials") {
    auto v = pc::FuzzyVariants("zhang");
    CHECK(Has(v, "zang"));   // zh -> z
    CHECK(Has(v, "zhan"));   // ang -> an
    CHECK_FALSE(Has(v, "zhang"));  // excludes self

    CHECK(Has(pc::FuzzyVariants("shi"), "si"));   // sh -> s
    CHECK(Has(pc::FuzzyVariants("chi"), "ci"));   // ch -> c
    CHECK(Has(pc::FuzzyVariants("zong"), "zhong")); // z -> zh
}

TEST_CASE("FuzzyVariants nasal/lateral and front/back nasal") {
    auto v = pc::FuzzyVariants("nan");
    CHECK(Has(v, "lan"));    // n -> l
    CHECK(Has(v, "nang"));   // an -> ang

    auto w = pc::FuzzyVariants("lin");
    CHECK(Has(w, "nin"));    // l -> n
    CHECK(Has(w, "ling"));   // in -> ing
}

TEST_CASE("FuzzyVariants only returns known syllables, never self") {
    for (const char* syl : {"zhang", "nan", "lin", "shi", "zhong", "ang", "en"}) {
        auto v = pc::FuzzyVariants(syl);
        for (const auto& c : v) {
            CHECK(Dict::PinyinSyllableSet().contains(c));
            CHECK(c != syl);
        }
    }
}

TEST_CASE("FuzzyVariants short input") {
    CHECK(pc::FuzzyVariants("a").empty());
    CHECK(pc::FuzzyVariants("z").empty());
    CHECK(pc::FuzzyVariants("").empty());
}

TEST_CASE("TypoVariants adjacent transposition") {
    auto v = pc::TypoVariants("jei");
    CHECK(Has(v, "jie"));
    CHECK_FALSE(Has(v, "jei"));
}

TEST_CASE("TypoVariants only known syllables, never self") {
    for (const char* seg : {"jei", "zhog", "nii", "hao"}) {
        auto v = pc::TypoVariants(seg);
        for (const auto& c : v) {
            CHECK(Dict::PinyinSyllableSet().contains(c));
            CHECK(c != seg);
        }
    }
}

TEST_CASE("TypoVariants length bounds") {
    CHECK(pc::TypoVariants("a").empty());        // < 2
    CHECK(pc::TypoVariants("").empty());
    CHECK(pc::TypoVariants("abcdefg").empty());  // > 6
}

TEST_CASE("TypoVariants cache is stable") {
    auto a = pc::TypoVariants("jei");
    auto b = pc::TypoVariants("jei");
    CHECK(a == b);
}

TEST_CASE("T9TypoVariants keypad substitution recovers a mistyped syllable") {
    // 636 = men; 635 is a keypad mistype (5 and 6 are adjacent).
    auto v = pc::T9TypoVariants("635");
    CHECK(Has(v, "636"));
    CHECK_FALSE(Has(v, "635"));
}

TEST_CASE("T9TypoVariants only known T9 syllables, length-preserving, never self") {
    for (const char* span : {"635", "94665", "4266", "74664"}) {
        auto v = pc::T9TypoVariants(span);
        for (const auto& c : v) {
            CHECK(Dict::IsKnownT9Syllable(c));
            CHECK(c.size() == std::string(span).size());
            CHECK(c != span);
        }
    }
}

TEST_CASE("T9TypoVariants length bounds") {
    CHECK(pc::T9TypoVariants("9").empty());
    CHECK(pc::T9TypoVariants("").empty());
    CHECK(pc::T9TypoVariants("9466448").empty());  // > 6
}