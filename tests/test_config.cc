#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "sime.h"
#include "user.h"

using sime::Sime;
using sime::UserSentence;

TEST_CASE("Sime runtime config defaults match historical constants") {
    Sime s;
    CHECK(s.GruPinyinScale() == doctest::Approx(0.40));
    CHECK(s.GruT9Scale() == doctest::Approx(0.50));
    CHECK(s.FuzzyCorrectionEnabled());
    CHECK(s.TypoCorrectionEnabled());
    CHECK(s.EnglishWhitelistEnabled());
    CHECK_FALSE(s.T9CorrectionEnabled());  // opt-in, default off
    CHECK_FALSE(s.AdaptiveGruEnabled());   // opt-in, default off
}

TEST_CASE("Sime config setters update values") {
    Sime s;
    s.SetGruScales(0.1, 0.9);
    CHECK(s.GruPinyinScale() == doctest::Approx(0.1));
    CHECK(s.GruT9Scale() == doctest::Approx(0.9));

    s.SetCorrectionEnabled(false, false);
    CHECK_FALSE(s.FuzzyCorrectionEnabled());
    CHECK_FALSE(s.TypoCorrectionEnabled());
    s.SetCorrectionEnabled(true, true);
    CHECK(s.FuzzyCorrectionEnabled());
    CHECK(s.TypoCorrectionEnabled());

    s.SetEnglishWhitelistEnabled(false);
    CHECK_FALSE(s.EnglishWhitelistEnabled());

    s.SetT9CorrectionEnabled(true);
    CHECK(s.T9CorrectionEnabled());
    s.SetT9CorrectionEnabled(false);
    CHECK_FALSE(s.T9CorrectionEnabled());

    s.SetAdaptiveGruEnabled(true);
    CHECK(s.AdaptiveGruEnabled());
    s.SetAdaptiveGruEnabled(false);
    CHECK_FALSE(s.AdaptiveGruEnabled());

    // Penalty setters have no getters; exercise them for compile/link coverage.
    s.SetCorrectionPenalties(1.0, 2.0);
    s.SetEnglishPenalties(1.0, 2.0, -1.0);
}

TEST_CASE("UserSentence decay half-life default and setter") {
    UserSentence us;
    CHECK(us.DecayHalfLife() == doctest::Approx(30.0));
    us.SetDecayHalfLife(7.0);
    CHECK(us.DecayHalfLife() == doctest::Approx(7.0));
    us.SetDecayHalfLife(0.0);
    CHECK(us.DecayHalfLife() == doctest::Approx(0.0));
    us.SetDecayHalfLife(-5.0);
    CHECK(us.DecayHalfLife() == doctest::Approx(0.0));
}