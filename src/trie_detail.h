#pragma once

#include "trie.h"

#include <cstddef>
#include <vector>

namespace trie {

inline std::vector<uint8_t> ScanAlphabet(const ArrayUnit* array, std::size_t size) {
    bool seen[256] = {};
    for (std::size_t i = 0; i < size; ++i) {
        uint8_t l = array[i].label;
        if (l != 0) seen[l] = true;
    }
    std::vector<uint8_t> result;
    for (int c = 1; c <= 255; ++c) {
        if (seen[c]) result.push_back(static_cast<uint8_t>(c));
    }
    return result;
}

} // namespace trie