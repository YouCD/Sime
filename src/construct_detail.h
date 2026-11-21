#pragma once

#include "construct.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace sime {

template <typename Level>
std::size_t Constructor::CutLevelByMark(std::vector<Node>& ups, Level& current, float_t mark_value) {
    if (current.empty()) {
        return 0;
    }
    const std::size_t total = current.size();
    std::vector<std::size_t> removed_prefix(total + 1, 0);
    std::size_t new_size = 0;
    for (std::size_t idx = 0; idx < total; ++idx) {
        bool keep = (current[idx].pro != mark_value) || (idx + 1 == total);
        if (keep) {
            if (new_size != idx) {
                current[new_size] = current[idx];
            }
            removed_prefix[idx + 1] = removed_prefix[idx];
            ++new_size;
        } else {
            removed_prefix[idx + 1] = removed_prefix[idx] + 1;
        }
    }

    for (auto& up : ups) {
        std::size_t down = up.down;
        if (down > total) {
            down = total;
        }
        std::size_t removed = removed_prefix[down];
        up.down = static_cast<std::uint32_t>(down - removed);
    }
    return new_size;
}

template <typename Level>
int Constructor::CutLevel(NodeLevel& up_level, Level& current, int threshold,
                          const std::vector<bool>& protect_mask) {
    if (threshold <= 0) {
        return static_cast<int>(current.size());
    }
    int write = 0;
    auto up_it = up_level.begin();
    auto up_last = up_level.end();
    for (int idx = 0; idx < static_cast<int>(current.size()); ++idx) {
        bool keep = false;
        bool is_protected = (idx < static_cast<int>(protect_mask.size()) &&
                             protect_mask[idx]);
        if constexpr (std::is_same_v<Level, LeaveLevel>) {
            keep = is_protected ||
                   (static_cast<int>(current[idx].cnt) > threshold) ||
                   (idx + 1 == static_cast<int>(current.size()));
        } else {
            keep = is_protected ||
                   (static_cast<int>(current[idx].cnt) > threshold) ||
                   (idx + 1 == static_cast<int>(current.size())) ||
                   (current[idx + 1].down != current[idx].down);
        }
        if (keep) {
            if (write != idx) {
                current[write] = current[idx];
            }
            while (up_it != up_last &&
                   up_it->down <= static_cast<std::uint32_t>(idx)) {
                up_it->down = static_cast<std::uint32_t>(write);
                ++up_it;
            }
            ++write;
        }
    }
    return write;
}

template <typename DownLevel>
void Constructor::DiscountLevel(NodeLevel& level,
                                DownLevel& down_level,
                                NeyDiscounter& disc,
                                int parent_lvl) {
    // Interpolated Absolute Discounting. For each parent h at `parent_lvl`:
    //   gamma(h) = (D * num_children + lost_mass) / denom
    //   P_I(w | h) = max(cnt(h,w) - D, 0) / denom + gamma(h) * P_I(w | h')

    constexpr float_t MinProb = 1e-12;
    constexpr float_t MaxProb = 1.0 - 1e-9;

    for (std::size_t idx = 0; idx + 1 < level.size(); ++idx) {
        Node& node = level[idx];
        Node& next = level[idx + 1];

        std::uint32_t num_children = next.down - node.down;
        float_t sum_children = 0.0;
        for (std::size_t i = node.down; i < next.down; ++i) {
            sum_children += down_level[i].cnt;
        }

        float_t denom = std::max(node.cnt, 1.0);

        float_t lost_mass = std::max(denom - sum_children, 0.0);

        float_t gamma;
        if (num_children == 0) {
            gamma = 1.0;
        } else {
            gamma = (disc.D() * static_cast<float_t>(num_children) + lost_mass)
                  / denom;
        }
        gamma = std::clamp(gamma, MinProb, MaxProb);
        node.bow = static_cast<float>(-std::log(gamma));

        for (std::size_t down_idx = node.down; down_idx < next.down; ++down_idx) {
            float_t discounted = disc.Discount(down_level[down_idx].cnt);
            float_t alpha = discounted / denom;

            TokenID w = down_level[down_idx].id;

            float_t p_lower;
            if (parent_lvl == 0) {
                p_lower = std::exp(-node_levels_[0][0].pro);
            } else if (parent_lvl == 1) {
                p_lower = GetPro(1, &w);
            } else {
                std::array<TokenID, 2> q{node.id, w};
                p_lower = GetPro(2, q.data());
            }

            float_t p_interp = alpha + gamma * p_lower;
            p_interp = std::clamp(p_interp, MinProb, MaxProb);
            down_level[down_idx].pro = static_cast<float>(-std::log(p_interp));
        }
    }
}

} // namespace sime