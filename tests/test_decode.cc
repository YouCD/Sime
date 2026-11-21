#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "sime.h"

#include <filesystem>
#include <string>
#include <vector>

using sime::DecodeResult;
using sime::Sime;
using sime::TokenID;

#ifndef SIME_TEST_MODEL_DIR

TEST_CASE("decode integration skipped (SIME_TEST_MODEL_DIR not set)") {
    MESSAGE("Set -DSIME_TEST_MODEL_DIR=<model dir> to run decode integration tests");
}

#else

namespace {

const char* kDir = SIME_TEST_MODEL_DIR;

bool HasModel() {
    return std::filesystem::exists(std::filesystem::path(kDir) / "sime.dict") &&
           std::filesystem::exists(std::filesystem::path(kDir) / "sime.cnt");
}

struct Fixture {
    Sime engine;
    Fixture()
        : engine(std::filesystem::path(kDir) / "sime.dict",
                 std::filesystem::path(kDir) / "sime.cnt") {}
};

int RankOf(const std::vector<DecodeResult>& v, TokenID tid) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!v[i].tokens.empty() && v[i].tokens[0] == tid)
            return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

TEST_CASE("prediction: enabled-but-empty user history matches pure LM") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());
    auto decoded = f.engine.DecodeSentence("women", 0);
    REQUIRE(!decoded.empty());
    auto ctx = decoded[0].tokens;

    f.engine.SetUserSentenceEnabled(false);
    auto without = f.engine.NextTokens(ctx, 10, false);
    f.engine.SetUserSentenceEnabled(true);
    auto with = f.engine.NextTokens(ctx, 10, false);

    REQUIRE(with.size() == without.size());
    for (std::size_t i = 0; i < with.size(); ++i) {
        CHECK(with[i].text == without[i].text);
        CHECK(with[i].score == doctest::Approx(without[i].score));
    }
}

TEST_CASE("prediction: user seed lifts a low-ranked token") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());
    auto decoded = f.engine.DecodeSentence("women", 0);
    REQUIRE(!decoded.empty());
    auto ctx = decoded[0].tokens;

    f.engine.SetUserSentenceEnabled(true);
    auto base = f.engine.NextTokens(ctx, 12, false);
    REQUIRE(base.size() >= 4);

    const TokenID target = base.back().tokens[0];
    const int before = RankOf(base, target);
    REQUIRE(before >= 0);

    f.engine.SeedUserUnigram(target, 100000);
    auto boosted = f.engine.NextTokens(ctx, 12, false);
    const int after = RankOf(boosted, target);
    CHECK(after >= 0);
    CHECK(after < before);
}

TEST_CASE("prediction: English-only filter returns ASCII tokens") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());
    auto decoded = f.engine.DecodeSentence("women", 0);
    REQUIRE(!decoded.empty());
    auto preds = f.engine.NextTokens(decoded[0].tokens, 10, true);
    for (const auto& p : preds) {
        REQUIRE(!p.text.empty());
        unsigned char c = static_cast<unsigned char>(p.text[0]);
        // ASCII letter, apostrophe, or U+2581 (0xE2 lead) word-boundary piece.
        CHECK((c < 0x80 || c == 0xE2));
    }
}

TEST_CASE("two-letter commands surface without displacing pinyin") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());

    auto has_text = [](const std::vector<DecodeResult>& v,
                       const std::string& text) {
        for (const auto& r : v)
            if (r.text == text) return true;
        return false;
    };

    // Whitelisted wc keeps its strong bonus; gently-lifted ps now appears.
    auto wc = f.engine.DecodeStr("wc", 3);
    REQUIRE(!wc.empty());
    CHECK(wc[0].text == "wc");
    CHECK(has_text(f.engine.DecodeStr("ps", 5), "ps"));

    // Full pinyin syllables must stay Chinese-first.
    auto ping = f.engine.DecodeStr("ping", 3);
    REQUIRE(!ping.empty());
    CHECK(ping[0].text == "平");
    auto du = f.engine.DecodeStr("du", 3);
    REQUIRE(!du.empty());
    CHECK_FALSE(has_text(du, "du"));
}

TEST_CASE("mixed script: English words survive inside a pinyin sentence") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());

    auto top = [&](const char* in) {
        auto r = f.engine.DecodeSentence(in, 0);
        REQUIRE(!r.empty());
        return r[0].text;
    };

    CHECK(top("yong grep sousuo") == "用 grep 搜索");
    CHECK(top("yong vim bianji") == "用 vim 编辑");
    // song has an exact CN rival (送/宋) and must not be lifted.
    CHECK(top("chang song").find("song") == std::string::npos);
    // Pure pinyin / pure English full-cover paths are unchanged.
    CHECK(top("women") == "我们");
    CHECK(top("hello") == "hello");
}

TEST_CASE("apostrophe works as a mixed-script separator (space commits in IME)") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());

    auto top = [&](const char* in) {
        auto r = f.engine.DecodeSentence(in, 0);
        REQUIRE(!r.empty());
        return r[0].text;
    };

    // Apostrophe-delimited English renders with spaces, same as typed spaces.
    CHECK(top("yong'grep'sousuo") == "用 grep 搜索");
    CHECK(top("yong'vim'bianji") == "用 vim 编辑");
    CHECK(top("shuo'hello'women") == "说 hello 我们");
    // Pure-pinyin apostrophes stay seamless (no English adjacent).
    CHECK(top("xi'an") == "西安");
    CHECK(top("chang'an") == "长安");
    // CN-rival guard still applies through apostrophes.
    CHECK(top("chang'song").find("song") == std::string::npos);

    // Apostrophe and space paths score identically for the same sentence.
    auto apos = f.engine.DecodeSentence("yong'grep'sousuo", 0);
    auto space = f.engine.DecodeSentence("yong grep sousuo", 0);
    REQUIRE(!apos.empty());
    REQUIRE(!space.empty());
    CHECK(apos[0].text == space[0].text);
    CHECK(apos[0].score == doctest::Approx(space[0].score));
}

TEST_CASE("T9 key-typo correction is opt-in and recovers mistypes") {
    if (!HasModel()) return;
    Fixture f;
    REQUIRE(f.engine.Ready());
    CHECK_FALSE(f.engine.T9CorrectionEnabled());  // default off

    // 96635 = wo + mistyped men (636 -> 635). Off: no clean recovery.
    f.engine.SetT9CorrectionEnabled(false);
    auto off = f.engine.DecodeNumStr("96635", "", 3);
    REQUIRE(!off.empty());
    CHECK(off[0].text != "我们");

    // On: the keypad edit 635 -> 636 recovers 我们.
    f.engine.SetT9CorrectionEnabled(true);
    auto on = f.engine.DecodeNumStr("96635", "", 3);
    REQUIRE(!on.empty());
    CHECK(on[0].text == "我们");

    // Correct T9 input (94664486 = zhong'guo) is unaffected by the switch.
    f.engine.SetT9CorrectionEnabled(false);
    auto exact_off = f.engine.DecodeNumStr("94664486", "", 1);
    f.engine.SetT9CorrectionEnabled(true);
    auto exact_on = f.engine.DecodeNumStr("94664486", "", 1);
    REQUIRE(!exact_off.empty());
    REQUIRE(!exact_on.empty());
    CHECK(exact_off[0].text == exact_on[0].text);
}

#endif
