#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "ustr.h"

#include <stdexcept>
#include <string>

using sime::ustr::FromU32;
using sime::ustr::ToU32;

TEST_CASE("ASCII round trip") {
    const std::string s = "hello world 123";
    CHECK(FromU32(ToU32(s)) == s);
    CHECK(ToU32(s).size() == s.size());
}

TEST_CASE("Chinese round trip") {
    const std::string s = "你好世界输入法";
    auto u32 = ToU32(s);
    CHECK(u32.size() == 7);
    CHECK(u32[0] == U'你');
    CHECK(FromU32(u32) == s);
}

TEST_CASE("Emoji (4-byte) round trip") {
    const std::string s = "a😀b🚀c";
    auto u32 = ToU32(s);
    CHECK(u32.size() == 5);
    CHECK(u32[1] == U'😀');
    CHECK(u32[3] == U'🚀');
    CHECK(FromU32(u32) == s);
}

TEST_CASE("SentencePiece boundary marker U+2581 round trip") {
    const std::string s = "\xE2\x96\x81grep";  // ▁grep
    auto u32 = ToU32(s);
    REQUIRE(u32.size() == 5);
    CHECK(u32[0] == 0x2581);
    CHECK(FromU32(u32) == s);
}

TEST_CASE("Empty input") {
    CHECK(ToU32("").empty());
    CHECK(FromU32(U"").empty());
}

TEST_CASE("Invalid UTF-8 throws runtime_error") {
    // Truncated 3-byte lead.
    CHECK_THROWS_AS(ToU32("\xE2\x96"), std::runtime_error);
    // Bad continuation byte.
    CHECK_THROWS_AS(ToU32("\xE2\x28\x81"), std::runtime_error);
    // Lone continuation byte.
    CHECK_THROWS_AS(ToU32("\x80"), std::runtime_error);
    // Invalid lead 0xFF.
    CHECK_THROWS_AS(ToU32("\xFF"), std::runtime_error);
}