#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "user.h"

#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using sime::NotToken;
using sime::TokenID;
using sime::UserSentence;

namespace {

std::uint32_t NowSec() {
    return static_cast<std::uint32_t>(std::time(nullptr));
}
constexpr std::uint32_t kDay = 86400;

// Load a SEED-only V6 document with explicit @tick values so Rebuild
// reconstructs decay at the stored historical ticks (runtime SeedUnigram
// always adds fresh weight; only the load path honours historical ticks).
UserSentence LoadSeeds(const std::string& body, std::string_view sig = "sig") {
    UserSentence us;
    std::istringstream in("SIME_USER_SENTENCE_V6\nVOCAB " + std::string(sig) +
                          "\nSEED\n" + body);
    REQUIRE(us.LoadText(in, sig));
    return us;
}

std::string TokenText(TokenID id) { return "t" + std::to_string(id); }

}  // namespace

TEST_CASE("default state has no history") {
    UserSentence us;
    CHECK(us.SentenceCount() == 0);
    CHECK(us.TotalUnigramCount() == 0);
    CHECK(us.UnigramCount(10) == 0);
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) == 0.0);
    CHECK(us.Weight() == doctest::Approx(0.2));
    CHECK(us.MaxSentenceCount() == 8192);
}

TEST_CASE("Add paragraph grouping by context tail") {
    UserSentence us;
    us.Add({}, {10, 20});
    CHECK(us.SentenceCount() == 1);
    // context {20} matches the flattened tail {10,20} -> same paragraph.
    us.Add({20}, {30});
    CHECK(us.SentenceCount() == 1);
    // empty context -> new paragraph.
    us.Add({}, {40});
    CHECK(us.SentenceCount() == 2);
    // non-matching context -> new paragraph.
    us.Add({99}, {50});
    CHECK(us.SentenceCount() == 3);
}

TEST_CASE("unigram / bigram / distinct-successor counts") {
    UserSentence us;
    us.Add({}, {10, 20});
    us.Add({20}, {30});  // one paragraph, flat {10,20,30}
    CHECK(us.UnigramCount(10) == 1);
    CHECK(us.UnigramCount(20) == 1);
    CHECK(us.UnigramCount(30) == 1);
    CHECK(us.TotalUnigramCount() == 3);
    CHECK(us.BigramCount(10, 20) == 1);
    CHECK(us.BigramCount(20, 30) == 1);
    CHECK(us.BigramCount(10, 30) == 0);
    CHECK(us.DistinctSuccessors(10) == 1);
    CHECK(us.DistinctSuccessors(20) == 1);
    CHECK(us.DistinctSuccessors(30) == 0);
    CHECK(us.BigramCount(NotToken, 10) == 0);
    CHECK(us.UnigramCount(NotToken) == 0);
}

TEST_CASE("NotToken entries are cleaned out") {
    UserSentence us;
    us.Add({}, {NotToken, 10, NotToken, 20});
    CHECK(us.UnigramCount(10) == 1);
    CHECK(us.UnigramCount(20) == 1);
    CHECK(us.TotalUnigramCount() == 2);
    us.Add({}, {NotToken});
    CHECK(us.SentenceCount() == 1);  // all-NotToken selection ignored
}

TEST_CASE("SeedUnigram adds unigram mass only, no fabricated bigram") {
    UserSentence us;
    us.SeedUnigram(50, 100);
    CHECK(us.UnigramCount(50) == 100);
    CHECK(us.SeedCount(50) == 100);
    CHECK(us.TotalUnigramCount() == 100);
    CHECK(us.BigramCount(50, 60) == 0);
    CHECK(us.DistinctSuccessors(50) == 0);
    CHECK(us.SentenceCount() == 0);  // seeds are not paragraphs
    // aggregate
    us.SeedUnigram(50, 25);
    CHECK(us.SeedCount(50) == 125);
    // guards
    us.SeedUnigram(NotToken, 10);
    us.SeedUnigram(60, 0);
    CHECK(us.SeedCount(60) == 0);
}

TEST_CASE("paragraph count excludes seed mass for bigram denominator") {
    UserSentence us;
    us.Add({}, {10, 20});
    us.SeedUnigram(10, 1000);
    CHECK(us.UnigramCount(10) == 1001);
    CHECK(us.SeedCount(10) == 1000);
    // bigram still observed once; seed never inflates transition counts.
    CHECK(us.BigramCount(10, 20) == 1);
    CHECK(us.DistinctSuccessors(10) == 1);
}

TEST_CASE("CostAdjustment improves frequent token, never worsens") {
    UserSentence us;
    for (int i = 0; i < 5; ++i) us.Add({}, {10});
    const double lm = 2.0;
    double adj_seen = us.CostAdjustment(NotToken, 10, lm);
    double adj_unseen = us.CostAdjustment(NotToken, 99, lm);
    CHECK(adj_seen <= 0.0);
    CHECK(adj_seen < 0.0);          // history lifts the picked token
    CHECK(adj_unseen == 0.0);       // never-picked token: no adjustment
    CHECK(us.CostAdjustment(NotToken, NotToken, lm) == 0.0);
}

TEST_CASE("CostAdjustment weight bypass") {
    UserSentence us;
    us.Add({}, {10});
    us.SetWeight(0.0);
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) == 0.0);
    us.SetWeight(1.0);
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) == 0.0);
    us.SetWeight(0.2);
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) < 0.0);
}

TEST_CASE("SetWeight clamps negative, accepts positive") {
    UserSentence us;
    us.SetWeight(-0.5);
    CHECK(us.Weight() == 0.0);
    us.SetWeight(0.5);
    CHECK(us.Weight() == doctest::Approx(0.5));
}

TEST_CASE("decay: fresh seed outranks stale seed") {
    const std::uint32_t now = NowSec();
    std::ostringstream body;
    body << "10 100 @" << now << "\n"
         << "20 100 @" << (now - 90 * kDay) << "\n";
    UserSentence us = LoadSeeds(body.str());
    double adj_fresh = us.CostAdjustment(NotToken, 10, 2.0);
    double adj_stale = us.CostAdjustment(NotToken, 20, 2.0);
    CHECK(adj_fresh < adj_stale);   // more negative = bigger lift
    CHECK(adj_stale <= 0.0);
}

TEST_CASE("decay: 30-day half-life boundary") {
    const std::uint32_t now = NowSec();
    // 100 decayed 30 days ~= 50; competing fresh seed of 50 -> equal weight.
    std::ostringstream body;
    body << "10 100 @" << (now - 30 * kDay) << "\n"
         << "20 50 @" << now << "\n";
    UserSentence us = LoadSeeds(body.str());
    double a = us.CostAdjustment(NotToken, 10, 2.0);
    double b = us.CostAdjustment(NotToken, 20, 2.0);
    CHECK(a == doctest::Approx(b).epsilon(1e-3));
}

TEST_CASE("decay: single-seed normalization is time-invariant") {
    const std::uint32_t now = NowSec();
    UserSentence fresh = LoadSeeds("10 100 @" + std::to_string(now) + "\n");
    UserSentence stale =
        LoadSeeds("10 100 @" + std::to_string(now - 90 * kDay) + "\n");
    double a = fresh.CostAdjustment(NotToken, 10, 2.0);
    double b = stale.CostAdjustment(NotToken, 10, 2.0);
    // cur_w/total_w cancels for a lone seed -> ranking unchanged by time.
    CHECK(a == doctest::Approx(b).epsilon(1e-3));
}

TEST_CASE("decay: future tick clamps elapsed to zero") {
    const std::uint32_t now = NowSec();
    UserSentence future =
        LoadSeeds("10 100 @" + std::to_string(now + 100000) + "\n" +
                  "20 100 @" + std::to_string(now) + "\n");
    double a = future.CostAdjustment(NotToken, 10, 2.0);
    double b = future.CostAdjustment(NotToken, 20, 2.0);
    CHECK(a == doctest::Approx(b).epsilon(1e-6));
}

TEST_CASE("SetDecayHalfLife rebuilds decay at the new rate") {
    const std::uint32_t now = NowSec();
    // 100 @ -30d vs fresh 50: equal under the default 30-day half-life.
    std::ostringstream body;
    body << "10 100 @" << (now - 30 * kDay) << "\n"
         << "20 50 @" << now << "\n";
    UserSentence us = LoadSeeds(body.str());
    CHECK(us.DecayHalfLife() == doctest::Approx(30.0));
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) ==
          doctest::Approx(us.CostAdjustment(NotToken, 20, 2.0)).epsilon(1e-3));

    // 60-day half-life: 100 @ -30d ~= 70.7 > 50, so token 10 lifts more.
    us.SetDecayHalfLife(60.0);
    CHECK(us.DecayHalfLife() == doctest::Approx(60.0));
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) <
          us.CostAdjustment(NotToken, 20, 2.0));

    // Disabled decay: 100 keeps full mass > 50, token 10 still lifts more.
    us.SetDecayHalfLife(0.0);
    CHECK(us.DecayHalfLife() == doctest::Approx(0.0));
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) <
          us.CostAdjustment(NotToken, 20, 2.0));
}

TEST_CASE("SaveText / LoadText V6 round trip") {
    UserSentence us;
    us.Add({}, {10, 20});
    us.Add({20}, {30});
    us.SeedUnigram(50, 100);
    std::ostringstream out;
    REQUIRE(us.SaveText(out, "sig", TokenText));
    const std::string doc = out.str();
    CHECK(doc.rfind("SIME_USER_SENTENCE_V6\n", 0) == 0);
    CHECK(doc.find("VOCAB sig\n") != std::string::npos);
    CHECK(doc.find("SEED\n") != std::string::npos);
    CHECK(doc.find("@") != std::string::npos);
    CHECK(doc.find("\t") != std::string::npos);  // annotation column

    std::istringstream in(doc);
    UserSentence loaded;
    REQUIRE(loaded.LoadText(in, "sig"));
    CHECK(loaded.SentenceCount() == us.SentenceCount());
    CHECK(loaded.UnigramCount(10) == 1);
    CHECK(loaded.UnigramCount(20) == 1);
    CHECK(loaded.UnigramCount(30) == 1);
    CHECK(loaded.BigramCount(10, 20) == 1);
    CHECK(loaded.BigramCount(20, 30) == 1);
    CHECK(loaded.SeedCount(50) == 100);
}

TEST_CASE("LoadText rejects vocab mismatch") {
    UserSentence us;
    us.Add({}, {10});
    std::ostringstream out;
    REQUIRE(us.SaveText(out, "sig1"));
    std::istringstream in(out.str());
    UserSentence loaded;
    CHECK_FALSE(loaded.LoadText(in, "sig2"));
    // empty expected sig accepts any stored sig
    std::istringstream in2(out.str());
    UserSentence loaded2;
    CHECK(loaded2.LoadText(in2, ""));
}

TEST_CASE("LoadText accepts legacy V5 / V4 headers, absent tick -> now") {
    for (const char* hdr : {"SIME_USER_SENTENCE_V5", "SIME_USER_SENTENCE_V4"}) {
        std::istringstream in(std::string(hdr) + "\nVOCAB sig\n\n10 20\n");
        UserSentence us;
        REQUIRE(us.LoadText(in, "sig"));
        CHECK(us.UnigramCount(10) == 1);
        CHECK(us.UnigramCount(20) == 1);
        CHECK(us.BigramCount(10, 20) == 1);
        // fresh (tick=now) -> adjustment present
        CHECK(us.CostAdjustment(NotToken, 10, 2.0) < 0.0);
    }
}

TEST_CASE("LoadText rejects bad header / missing vocab line") {
    UserSentence us;
    std::istringstream bad("GARBAGE\nVOCAB sig\n");
    CHECK_FALSE(us.LoadText(bad, "sig"));
    std::istringstream novocab("SIME_USER_SENTENCE_V6\n10 20\n");
    CHECK_FALSE(us.LoadText(novocab, "sig"));
}

TEST_CASE("SetMaxSentenceCount evicts oldest and adjusts counts") {
    UserSentence us;
    us.Add({}, {10});
    us.Add({}, {20});
    us.Add({}, {30});
    CHECK(us.SentenceCount() == 3);
    us.SetMaxSentenceCount(2);
    CHECK(us.SentenceCount() == 2);
    CHECK(us.MaxSentenceCount() == 2);
    CHECK(us.UnigramCount(10) == 0);  // oldest evicted
    CHECK(us.UnigramCount(20) == 1);
    CHECK(us.UnigramCount(30) == 1);
    CHECK(us.TotalUnigramCount() == 2);
}

TEST_CASE("LoadAndMigrate re-tokenizes text and preserves tick") {
    const std::uint32_t tick = NowSec() - 10 * kDay;
    auto path = std::filesystem::temp_directory_path() /
                ("sime_migrate_test_" + std::to_string(tick) + ".txt");
    {
        std::ofstream out(path);
        out << "SIME_USER_SENTENCE_V6\nVOCAB oldsig\n\n"
            << "10 20 @" << tick << "\t你 好\n"
            << "SEED\n"
            << "30 100 @" << tick << "\t世界\n";
    }
    UserSentence us;
    auto tokenize = [](std::string_view text) {
        std::vector<std::pair<TokenID, std::string>> out;
        if (text == "你好") out.emplace_back(100, "你好");
        else if (text == "世界") out.emplace_back(200, "世界");
        return out;
    };
    REQUIRE(us.LoadAndMigrate(path, tokenize));
    CHECK(us.UnigramCount(100) == 1);
    CHECK(us.SeedCount(200) == 100);
    CHECK(us.SentenceCount() == 1);
    // tick preserved through migration
    std::ostringstream out;
    REQUIRE(us.SaveText(out, "newsig", TokenText));
    CHECK(out.str().find("@" + std::to_string(tick)) != std::string::npos);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("Clear resets everything") {
    UserSentence us;
    us.Add({}, {10, 20});
    us.SeedUnigram(50, 100);
    us.Clear();
    CHECK(us.SentenceCount() == 0);
    CHECK(us.TotalUnigramCount() == 0);
    CHECK(us.SeedCount(50) == 0);
    CHECK(us.CostAdjustment(NotToken, 10, 2.0) == 0.0);
}