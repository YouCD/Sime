#include "sime.h"

#include "ustr.h"

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sime {

std::string Sime::AbbreviatePieces(const char* full_pieces,
                                   std::string_view input) {
    // Parse syllables from full_pieces.
    struct Syl { const char* begin; std::size_t len; };
    std::vector<Syl> syls;
    for (const char* p = full_pieces; *p; ) {
        const char* s = p;
        while (*p && *p != '\'') ++p;
        if (p > s) syls.push_back({s, static_cast<std::size_t>(p - s)});
        if (*p == '\'') ++p;
    }
    if (syls.empty()) return full_pieces;

    // Check if input char matches a syllable letter.
    auto charMatch = [](char ic, char sc) -> bool {
        if (ic == sc) return true;
        if (ic >= '2' && ic <= '9') {
            const char* letters = Dict::NumToLettersLower(
                static_cast<uint8_t>(ic));
            for (const char* l = letters; *l; ++l) {
                if (*l == sc) return true;
            }
        }
        return false;
    };

    // Find the max prefix length of syllable syl that matches input
    // starting at ipos. Returns 0 if first char doesn't match.
    auto maxMatch = [&](const Syl& syl, std::size_t ipos) -> std::size_t {
        std::size_t matched = 0;
        for (std::size_t k = 0; k < syl.len && ipos + k < input.size()
                 && input[ipos + k] != '\''; ++k) {
            if (!charMatch(input[ipos + k], syl.begin[k])) break;
            ++matched;
        }
        return matched;
    };

    // Recursive search: for each syllable, try lengths 1..max (prefer
    // shorter = more abbreviated). Return true if all syllables can be
    // covered consuming all input non-separator chars.
    std::vector<std::size_t> chosen(syls.size(), 0);

    std::function<bool(std::size_t, std::size_t)> solve =
        [&](std::size_t si, std::size_t ipos) -> bool {
        // Skip input separators.
        while (ipos < input.size() && input[ipos] == '\'') ++ipos;
        if (si == syls.size()) return ipos >= input.size();
        if (ipos >= input.size()) return false;
        std::size_t mm = maxMatch(syls[si], ipos);
        if (mm == 0) return false;
        for (std::size_t len = 1; len <= mm; ++len) {
            chosen[si] = len;
            // Skip any input separator after the consumed digits.
            std::size_t next_ipos = ipos + len;
            if (solve(si + 1, next_ipos)) return true;
        }
        return false;
    };

    if (!solve(0, 0)) {
        // Fallback: greedy (original behavior) — shouldn't normally happen.
        return full_pieces;
    }

    // Build result from chosen lengths.
    std::string result;
    for (std::size_t i = 0; i < syls.size(); ++i) {
        if (i > 0) result += '\'';
        result.append(syls[i].begin, chosen[i]);
    }
    return result;
}

std::string Sime::ExtractUnits(const std::vector<Link>& path,
                               std::string_view input) {
    std::string py;
    for (const auto& link : path) {
        if (link.id == NotToken) continue;
        if (!link.pieces || link.pieces[0] == '\0') continue;
        if (!py.empty()) py += '\'';
        py += AbbreviatePieces(link.pieces,
                               input.substr(link.start, link.end - link.start));
    }
    return py;
}

std::string Sime::ExtractText(const std::vector<Link>& path,
                              std::string_view input) const {
    std::u32string u32;
    for (const auto& link : path) {
        if (link.id == NotToken) {
            // A ▁ (space) boundary link renders as a single space; apostrophe
            // links render nothing. Suppress a duplicate space when the
            // previous token already ended in one.
            if (link.start < input.size() &&
                input[link.start] == static_cast<char>(0xE2) &&
                (u32.empty() || u32.back() != U' ')) {
                u32.push_back(U' ');
            }
            continue;
        }
        std::u32string t = ToText(link);
        // SentencePiece-style pieces may carry a leading ▁; don't double the
        // space we just emitted for the boundary link.
        if (!t.empty() && t[0] == 0x2581 && !u32.empty() && u32.back() == U' ') {
            t.erase(t.begin());
        }
        u32 += t;
    }
    return TextFromU32(u32);
}

std::string Sime::TextFromU32(std::u32string& u32) {
    for (auto& ch : u32) {
        if (ch == 0x2581) ch = U' ';
    }
    return ustr::FromU32(u32);
}

std::vector<TokenID> Sime::ExtractTokens(
    const std::vector<Link>& path) const {
    std::vector<TokenID> ids;
    for (const auto& link : path) {
        if (link.id == NotToken) continue;
        ids.push_back(link.id);
    }
    return ids;
}

void Sime::PruneNode(std::vector<Link>& edges,
                     std::unordered_map<TokenID, float_t>* score_cache) const {
    if (edges.size() <= NodeSize) return;

    // Group by span = end - start (bounded by max word length ≤ ~12).
    std::size_t max_span = 0;
    for (const auto& e : edges) {
        std::size_t span = e.end - e.start;
        if (span > max_span) max_span = span;
    }
    std::vector<std::vector<std::size_t>> groups(max_span + 1);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        groups[edges[i].end - edges[i].start].push_back(i);
    }

    auto get_score = [&](TokenID id) -> float_t {
        if (id == NotToken) return 0.0;
        if (score_cache) {
            auto it = score_cache->find(id);
            if (it != score_cache->end()) return it->second;
            Scorer::Pos ppos{};
            Scorer::Pos pnext{};
            float_t s = scorer_.ScoreMove(ppos, id, pnext);
            score_cache->emplace(id, s);
            return s;
        }
        Scorer::Pos ppos{};
        Scorer::Pos pnext{};
        return scorer_.ScoreMove(ppos, id, pnext);
    };

    std::vector<Link> pruned;
    pruned.reserve(edges.size());

    for (auto& indices : groups) {
        if (indices.empty()) continue;
        if (indices.size() <= NodeSize) {
            for (auto idx : indices) pruned.push_back(edges[idx]);
            continue;
        }

        // Single-key sort: estimate first-pass beam cost as
        //   unigram_score + edge.penalty
        // edge.penalty already encodes the English / expansion penalties
        // set by ComputeEdgePenalties.
        std::vector<std::pair<float_t, std::size_t>> scored;
        scored.reserve(indices.size());
        for (auto idx : indices) {
            float_t cost = get_score(edges[idx].id) + edges[idx].penalty;
            scored.emplace_back(cost, idx);
        }
        std::partial_sort(scored.begin(),
                          scored.begin() + static_cast<std::ptrdiff_t>(NodeSize),
                          scored.end());

        for (std::size_t i = 0; i < NodeSize; ++i) {
            pruned.push_back(edges[scored[i].second]);
        }
    }

    edges = std::move(pruned);
}

State Sime::InitialState(const std::vector<TokenID>& context) const {
    Scorer::Pos pos = scorer_.StartPos();
    TokenID last = NotToken;
    const std::size_t usable =
        static_cast<std::size_t>(std::max(scorer_.Num() - 1, 1));
    const std::size_t begin = context.size() > usable
                                  ? context.size() - usable
                                  : 0;
    for (std::size_t i = begin; i < context.size(); ++i) {
        TokenID token = context[i];
        if (token == NotToken) {
            continue;
        }
        Scorer::Pos next{};
        scorer_.ScoreMove(pos, token, next);
        scorer_.Back(next);
        pos = next;
        last = token;
    }
    return State(0.0, 0, pos, nullptr, last);
}

void Sime::Process(std::vector<Node>& net) const {
    for (std::size_t col = 0; col < net.size(); ++col) {
        auto& column = net[col];
        for (auto it = column.states.begin(); it != column.states.end(); ++it) {
            const auto& value = *it;
            for (const auto& word : column.es) {
                Scorer::Pos cur_pos = value.pos;
                Scorer::Pos next_pos{};
                float_t step = scorer_.ScoreMove(cur_pos, word.id, next_pos);
                scorer_.Back(next_pos);
                cur_pos = next_pos;

                float_t user_adjust = 0.0;
                if (user_sentence_enabled_) {
                    user_adjust = user_sentence_.CostAdjustment(
                        value.backtrace_token, word.id, step);
                }
                float_t next_cost =
                    value.score + step + word.penalty + user_adjust;
                State next(next_cost, word.end, cur_pos, &value, word.id,
                           word.pieces);
                net[word.end].states.Insert(next);
            }
        }
    }
}

std::vector<Sime::Link> Sime::Backtrace(
    const State& tail_state,
    std::size_t end) {
    std::vector<Link> path;
    const State* state = &tail_state;
    while (state != nullptr && state->backtrace_state != nullptr) {
        const State* prev = state->backtrace_state;
        path.push_back({prev->now, state->now,
                        state->backtrace_token,
                        state->backtrace_pieces});
        state = prev;
    }
    std::reverse(path.begin(), path.end());
    if (!path.empty() && path.back().end == end) {
        path.pop_back();
    }
    return path;
}

std::u32string Sime::ToText(const Link& n) const {
    if (n.id == NotToken) {
        return {};
    }
    std::u32string buffer;
    const char32_t* chars = dict_.TokenAt(n.id);
    if (chars) {
        for (std::size_t i = 0; chars[i] != 0 && i < 64; ++i) {
            buffer.push_back(chars[i]);
        }
    }
    return buffer;
}

} // namespace sime