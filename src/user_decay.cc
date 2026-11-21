#include "user.h"

#include <cmath>
#include <ctime>

namespace sime {

namespace {

constexpr double kLn2 = 0.69314718055994530942;
constexpr double kSecondsPerDay = 86400.0;

} // namespace

std::uint32_t UserSentence::Now() {
    return static_cast<std::uint32_t>(std::time(nullptr));
}

void UserSentence::SetDecayHalfLife(double days) {
    decay_lambda_ = (days > 0.0) ? kLn2 / (days * kSecondsPerDay) : 0.0;
    Rebuild();
}

double UserSentence::DecayHalfLife() const {
    if (decay_lambda_ <= 0.0) return 0.0;
    return kLn2 / (decay_lambda_ * kSecondsPerDay);
}

double UserSentence::DecayFactor(std::uint32_t from, std::uint32_t to) const {
    const double elapsed =
        (to > from) ? static_cast<double>(to - from) : 0.0;
    return std::exp(-elapsed * decay_lambda_);
}

void UserSentence::TouchDecay(TokenID token, double delta,
                              std::uint32_t now) {
    decay_total_ *= DecayFactor(decay_total_t_, now);
    decay_total_t_ = now;
    auto wit = decay_w_.find(token);
    const double old = (wit != decay_w_.end()) ? wit->second : 0.0;
    auto tit = decay_t_.find(token);
    const std::uint32_t old_t = (tit != decay_t_.end()) ? tit->second : now;
    double w = old * DecayFactor(old_t, now) + delta;
    if (w < 0.0) {
        w = 0.0;
    }
    decay_w_[token] = w;
    decay_t_[token] = now;
    decay_total_ += delta;
    if (decay_total_ < 0.0) {
        decay_total_ = 0.0;
    }
}

void UserSentence::AdjustEntryDecay(const Entry& entry, int delta,
                                    std::uint32_t now) {
    if (delta == 0) {
        return;
    }
    const double sign = static_cast<double>(delta);
    for (const auto& sel : entry.selections) {
        const double f = DecayFactor(sel.tick, now) * sign;
        for (auto token : sel.tokens) {
            TouchDecay(token, f, now);
        }
    }
}

} // namespace sime