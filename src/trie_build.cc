#include "trie.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_set>

namespace trie {
// --- Builder ---

void DoubleArray::Builder::Run(const std::vector<std::string>& keys,
                               const std::vector<uint32_t>& values) {
    // Build intermediate trie.
    auto root = std::make_unique<TrieNode>();
    for (std::size_t i = 0; i < keys.size(); ++i) {
        TrieNode* cur = root.get();
        for (char c : keys[i]) {
            auto ch = static_cast<uint8_t>(c);
            auto& child = cur->children[ch];
            if (!child) child = std::make_unique<TrieNode>();
            cur = child.get();
        }
        cur->eow = true;
        cur->value = values[i];
    }

    // Convert to double array.
    units_.resize(1024);
    used_.resize(1024, false);
    used_[0] = true;
    units_[0].label = 0;
    if (!root->children.empty() || root->eow)
        ConvertNode(root.get(), 0);

    // Trim trailing unused.
    while (!units_.empty() && !used_.back()) {
        units_.pop_back();
        used_.pop_back();
    }
}

std::unique_ptr<ArrayUnit[]> DoubleArray::Builder::GetResult(std::size_t& size) {
    size = units_.size();
    auto result = std::make_unique<ArrayUnit[]>(size);
    for (std::size_t i = 0; i < size; ++i) result[i] = units_[i];
    return result;
}

void DoubleArray::Builder::ConvertNode(TrieNode* node, std::size_t pos) {
    std::vector<uint8_t> labels;
    std::vector<TrieNode*> children;
    for (const auto& [ch, child] : node->children) {
        labels.push_back(ch);
        children.push_back(child.get());
    }
    if (node->eow) {
        labels.push_back(0);
        children.push_back(nullptr);
    }
    if (labels.empty()) return;

    uint32_t base = SetupChildren(labels, pos, node);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != 0) {
            ConvertNode(children[i], base ^ labels[i]);
        }
    }
}

uint32_t DoubleArray::Builder::SetupChildren(
    const std::vector<uint8_t>& labels, std::size_t pos, TrieNode* node) {
    uint32_t base = FindFreeBase(labels);
    units_[pos].index = static_cast<uint32_t>(pos ^ base);

    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::size_t p = base ^ labels[i];
        EnsureSize(p + 1);
        used_[p] = true;
        units_[p].label = labels[i];
        units_[p].parent = static_cast<uint32_t>(pos);
        if (labels[i] == 0) {
            units_[p].value = static_cast<int32_t>(node->value);
            units_[p].eow = true;
            units_[pos].eow = true;
        }
    }
    return base;
}

uint32_t DoubleArray::Builder::FindFreeBase(const std::vector<uint8_t>& labels) {
    uint32_t start = (prev_base_ > 256) ? prev_base_ - 256 : 1;
    for (uint32_t base = start; ; ++base) {
        bool ok = true;
        for (auto l : labels) {
            std::size_t p = base ^ l;
            EnsureSize(p + 1);
            if (used_[p]) { ok = false; break; }
        }
        if (ok) {
            prev_base_ = base;
            return base;
        }
    }
}

void DoubleArray::Builder::EnsureSize(std::size_t n) {
    if (n > units_.size()) {
        std::size_t ns = std::max(n, units_.size() * 2);
        units_.resize(ns);
        used_.resize(ns, false);
    }
}

} // namespace trie
