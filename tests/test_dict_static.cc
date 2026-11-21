#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "dict.h"

#include <algorithm>
#include <string>
#include <vector>

using sime::Dict;

TEST_CASE("IsKnownPinyin") {
    CHECK(Dict::IsKnownPinyin("zhong"));
    CHECK(Dict::IsKnownPinyin("ni"));
    CHECK(Dict::IsKnownPinyin("hao"));
    CHECK(Dict::IsKnownPinyin("a"));
    CHECK_FALSE(Dict::IsKnownPinyin("xyz"));
    CHECK_FALSE(Dict::IsKnownPinyin("zhog"));
    CHECK_FALSE(Dict::IsKnownPinyin(""));
}

TEST_CASE("IsExtendablePinyin") {
    // "xu" is a strict prefix of xue/xun/xuan.
    CHECK(Dict::IsExtendablePinyin("xu"));
    CHECK(Dict::IsExtendablePinyin("zh"));
    CHECK(Dict::IsExtendablePinyin("a"));
    // Terminal syllables with no longer extension.
    CHECK_FALSE(Dict::IsExtendablePinyin("xue"));
    CHECK_FALSE(Dict::IsExtendablePinyin("zhuang"));
}

TEST_CASE("PinyinSyllableSet membership") {
    const auto& set = Dict::PinyinSyllableSet();
    CHECK(set.contains("zhong"));
    CHECK(set.contains("ni"));
    CHECK(set.contains("hao"));
    CHECK_FALSE(set.contains("xyz"));
    CHECK_FALSE(set.empty());
}

TEST_CASE("LetterToNum / LettersToNums") {
    CHECK(Dict::LetterToNum('a') == '2');
    CHECK(Dict::LetterToNum('C') == '2');
    CHECK(Dict::LetterToNum('n') == '6');
    CHECK(Dict::LetterToNum('i') == '4');
    CHECK(Dict::LetterToNum('z') == '9');
    CHECK(Dict::LetterToNum('1') == 0);
    CHECK(Dict::LettersToNums("ni") == "64");
    CHECK(Dict::LettersToNums("zhong") == "94664");
    CHECK(Dict::LettersToNums("a1b") == "22");  // non-letters dropped
}

TEST_CASE("NumToLetters / NumToLettersLower") {
    CHECK(std::string(Dict::NumToLettersLower('2')) == "abc");
    CHECK(std::string(Dict::NumToLettersLower('7')) == "pqrs");
    CHECK(std::string(Dict::NumToLetters('2')) == "abcABC");
    CHECK(Dict::NumToLettersLower('1') == nullptr);
    CHECK(Dict::NumToLetters('0') == nullptr);
}

TEST_CASE("T9 syllable helpers") {
    // "ni" -> 64 is a known T9 syllable.
    CHECK(Dict::IsKnownT9Syllable("64"));
    CHECK_FALSE(Dict::IsKnownT9Syllable(""));
    CHECK_FALSE(Dict::IsKnownT9Syllable("11"));
    // "zhong" -> 94664.
    CHECK(Dict::IsKnownT9Syllable("94664"));

    auto sylls = Dict::T9PinyinSyllables("64", 16);
    CHECK(std::find(sylls.begin(), sylls.end(), "ni") != sylls.end());
    // Every returned syllable must T9-encode to a prefix of the query.
    for (const auto& s : sylls) {
        const std::string nums = Dict::LettersToNums(s);
        CHECK(std::string("64").substr(0, nums.size()) == nums);
    }
    // limit respected
    CHECK(Dict::T9PinyinSyllables("64", 2).size() <= 2);
    // invalid digit -> empty
    CHECK(Dict::T9PinyinSyllables("14", 16).empty());
    CHECK(Dict::T9PinyinSyllables("64", 0).empty());
}

TEST_CASE("IsExtendableT9Syllable") {
    // "94" (zh...) extends to longer finals.
    CHECK(Dict::IsExtendableT9Syllable("94"));
    CHECK(Dict::IsExtendableT9Syllable(""));  // empty -> true by contract
}