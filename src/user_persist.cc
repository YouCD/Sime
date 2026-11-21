#include "user.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace sime {

namespace {

constexpr const char* kHeader = "SIME_USER_SENTENCE_V6";
constexpr const char* kHeaderV5 = "SIME_USER_SENTENCE_V5";
constexpr const char* kHeaderV4 = "SIME_USER_SENTENCE_V4";
constexpr std::string_view kVocabPrefix = "VOCAB ";
constexpr std::string_view kSeedMarker = "SEED";

bool AcceptHeader(const std::string& line) {
    return line == kHeader || line == kHeaderV5 || line == kHeaderV4;
}

// Split a selection/seed id column into numeric fields plus an optional
// trailing "@<tick>" timestamp. Returns the tick (0 when absent, i.e.
// legacy → caller treats as now) and fills `fields` with the id tokens.
std::uint32_t SplitTick(std::string_view ids_part,
                        std::vector<std::string>& fields) {
    fields.clear();
    std::istringstream ss{std::string(ids_part)};
    std::string field;
    while (ss >> field) {
        fields.push_back(std::move(field));
    }
    std::uint32_t tick = 0;
    if (!fields.empty() && !fields.back().empty() && fields.back()[0] == '@') {
        tick = static_cast<std::uint32_t>(
            std::strtoul(fields.back().c_str() + 1, nullptr, 10));
        fields.pop_back();
    }
    return tick;
}

} // namespace

bool UserSentence::Load(const std::filesystem::path& path,
                       std::string_view vocab_sig) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    return LoadText(in, vocab_sig);
}

bool UserSentence::Save(const std::filesystem::path& path,
                       std::string_view vocab_sig,
                       const TokenTextFn& token_text) const {
    if (auto parent = path.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return false;
        }
    }

    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp);
        if (!out.is_open()) {
            return false;
        }
        if (!SaveText(out, vocab_sig, token_text)) {
            out.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

bool UserSentence::LoadText(std::istream& in, std::string_view vocab_sig) {
    std::string line;
    if (!std::getline(in, line) || !AcceptHeader(line)) {
        return false;
    }

    if (!std::getline(in, line)) {
        return false;
    }
    if (line.size() < kVocabPrefix.size() ||
        std::string_view(line).substr(0, kVocabPrefix.size()) != kVocabPrefix) {
        return false;
    }
    std::string_view stored_sig =
        std::string_view(line).substr(kVocabPrefix.size());
    if (!vocab_sig.empty() && stored_sig != vocab_sig) {
        return false;
    }

    seed_unigram_.clear();
    seed_total_ = 0;
    seed_tick_.clear();

    std::deque<Entry> loaded;
    Entry current;
    bool in_seed = false;
    bool para_full = false;
    const std::uint32_t now = Now();
    auto flush = [&]() {
        if (!current.selections.empty()) {
            if (!para_full) {
                loaded.push_back(std::move(current));
                if (loaded.size() >= max_sentence_count_) {
                    para_full = true;
                }
            }
            current = Entry{};
        }
    };

    while (std::getline(in, line)) {
        if (line == kSeedMarker) {
            flush();
            in_seed = true;
            continue;
        }
        if (in_seed) {
            // Seed line: "<id> <count> @<tick>[\t<annotation>]".
            // Annotation is informational only; absent @tick → now.
            auto tab = line.find('\t');
            std::string_view ids_part = (tab == std::string::npos)
                                            ? std::string_view(line)
                                            : std::string_view(line).substr(0, tab);
            std::vector<std::string> fields;
            std::uint32_t tick = SplitTick(ids_part, fields);
            if (tick == 0) tick = now;
            if (fields.size() >= 2) {
                std::uint64_t raw_id = std::strtoull(fields[0].c_str(), nullptr, 10);
                std::uint64_t raw_count = std::strtoull(fields[1].c_str(), nullptr, 10);
                if (raw_id <= std::numeric_limits<TokenID>::max() &&
                    raw_count > 0) {
                    SeedUnigram(static_cast<TokenID>(raw_id),
                                static_cast<std::uint32_t>(raw_count), tick);
                }
            }
            continue;
        }
        if (line.empty()) {
            flush();
            continue;
        }

        // Per-selection line: "<id1> <id2> ... @<tick>[\t<annotation>]"
        // The annotation is informational only — we ignore it on load.
        // Absent @tick (legacy V5/V4) → now, so old history stays fresh.
        auto tab = line.find('\t');
        std::string_view ids_part = (tab == std::string::npos)
                                        ? std::string_view(line)
                                        : std::string_view(line).substr(0, tab);

        std::vector<std::string> fields;
        std::uint32_t tick = SplitTick(ids_part, fields);
        if (tick == 0) tick = now;
        Sentence tokens;
        for (const auto& f : fields) {
            std::uint64_t raw = std::strtoull(f.c_str(), nullptr, 10);
            if (raw > std::numeric_limits<TokenID>::max()) {
                tokens.clear();
                break;
            }
            tokens.push_back(static_cast<TokenID>(raw));
        }
        tokens = CleanSentence(tokens);
        if (tokens.empty()) {
            continue;
        }
        current.selections.push_back({std::move(tokens), tick});
    }
    flush();

    sentences_ = std::move(loaded);
    Rebuild();
    return true;
}

bool UserSentence::LoadAndMigrate(const std::filesystem::path& path,
                                  const TokenizeFn& tokenize) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::string line;
    if (!std::getline(in, line) || !AcceptHeader(line)) {
        return false;
    }
    // Skip the stored VOCAB line — caller's vocab_sig will be written
    // on the next save.
    if (!std::getline(in, line) ||
        line.size() < kVocabPrefix.size() ||
        std::string_view(line).substr(0, kVocabPrefix.size()) != kVocabPrefix) {
        return false;
    }

    seed_unigram_.clear();
    seed_total_ = 0;
    seed_tick_.clear();

    std::deque<Entry> migrated;
    Entry current;
    bool in_seed = false;
    const std::uint32_t now = Now();

    auto flush_paragraph = [&]() {
        if (!current.selections.empty()) {
            migrated.push_back(std::move(current));
            current = Entry{};
        }
    };

    while (std::getline(in, line)) {
        if (line == kSeedMarker) {
            flush_paragraph();
            in_seed = true;
            continue;
        }
        if (in_seed) {
            // Seed line: "<old_id> <count> @<tick>\t<text>". Old id is
            // useless under the new vocab; re-tokenize the text but keep
            // the stored tick so migration preserves recency. Only
            // single-token words migrate cleanly (Rime entries are
            // word-level).
            auto tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::vector<std::string> fields;
            std::uint32_t tick = SplitTick(
                std::string_view(line).substr(0, tab), fields);
            if (tick == 0) tick = now;
            if (fields.size() < 2) {
                continue;
            }
            std::uint64_t raw_count = std::strtoull(fields[1].c_str(), nullptr, 10);
            if (raw_count == 0) {
                continue;
            }
            std::string text = line.substr(tab + 1);
            text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
            if (text.empty()) {
                continue;
            }
            auto cuts = tokenize(text);
            if (cuts.size() == 1 && cuts[0].first != NotToken &&
                cuts[0].first != static_cast<TokenID>(-1)) {
                SeedUnigram(cuts[0].first,
                            static_cast<std::uint32_t>(raw_count), tick);
            }
            continue;
        }
        if (line.empty()) {
            flush_paragraph();
            if (migrated.size() >= max_sentence_count_) {
                break;
            }
            continue;
        }
        // Take only the text column (after tab); old IDs are useless
        // under the new vocab. Strip the per-token spaces we wrote at
        // save time so Cutter sees an unbroken run. Keep the @tick from
        // the id column so migration preserves recency.
        auto tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        std::vector<std::string> fields;
        std::uint32_t tick = SplitTick(
            std::string_view(line).substr(0, tab), fields);
        if (tick == 0) tick = now;
        std::string text = line.substr(tab + 1);
        text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
        if (text.empty()) {
            continue;
        }

        // Per-line tokenize: each original selection becomes one new
        // Selection in the migrated file, preserving the user's pick
        // grouping. If any token in the slice is OOV under the new
        // vocab, drop the whole selection.
        auto cuts = tokenize(text);
        Sentence sel_tokens;
        bool has_unk = false;
        for (const auto& [id, txt] : cuts) {
            if (id == NotToken || id == static_cast<TokenID>(-1)) {
                has_unk = true;
                break;
            }
            sel_tokens.push_back(id);
        }
        if (has_unk || sel_tokens.empty()) {
            continue;
        }
        current.selections.push_back({std::move(sel_tokens), tick});
    }
    flush_paragraph();

    sentences_ = std::move(migrated);
    Rebuild();
    return true;
}

bool UserSentence::SaveText(std::ostream& out,
                           std::string_view vocab_sig,
                           const TokenTextFn& token_text) const {
    out << kHeader << '\n';
    out << kVocabPrefix << vocab_sig << '\n';
    bool first_entry = true;
    for (const auto& entry : sentences_) {
        if (!first_entry) {
            out << '\n';
        }
        first_entry = false;
        for (const auto& sel : entry.selections) {
            for (std::size_t i = 0; i < sel.tokens.size(); ++i) {
                if (i != 0) {
                    out << ' ';
                }
                out << sel.tokens[i];
            }
            out << " @" << sel.tick;
            if (token_text) {
                bool any = false;
                std::string annotation;
                for (std::size_t i = 0; i < sel.tokens.size(); ++i) {
                    auto t = token_text(sel.tokens[i]);
                    if (i != 0) {
                        annotation.push_back(' ');
                    }
                    annotation.append(t);
                    if (!t.empty()) {
                        any = true;
                    }
                }
                if (any) {
                    out << '\t' << annotation;
                }
            }
            out << '\n';
        }
    }
    if (!seed_unigram_.empty()) {
        out << kSeedMarker << '\n';
        std::vector<std::pair<TokenID, std::uint32_t>> seeds(
            seed_unigram_.begin(), seed_unigram_.end());
        std::sort(seeds.begin(), seeds.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [id, count] : seeds) {
            out << id << ' ' << count;
            auto tit = seed_tick_.find(id);
            out << " @" << (tit != seed_tick_.end() ? tit->second : Now());
            if (token_text) {
                auto t = token_text(id);
                if (!t.empty()) {
                    out << '\t' << t;
                }
            }
            out << '\n';
        }
    }
    return static_cast<bool>(out);
}

} // namespace sime