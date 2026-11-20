// sime-import: seed Sime user-sentence unigram frequencies from Rime
// history. Reads either a raw Rime userdb (LevelDB directory) directly,
// or a TSV export (pinyin \t text \t commits \t tick).
//
//   sime-import --dict <sime.dict> --cnt <sime.cnt>
//               --rime <rime_ice.userdb> [--rime <another.userdb> ...]
//               --out <sentences.txt>
//
//   sime-import --dict <sime.dict> --cnt <sime.cnt>
//               --in <export.tsv> --out <sentences.txt>
//
// Each entry's text is segmented with the Sime Cutter; entries that map
// to exactly one known word-level token contribute their commit count as
// a unigram seed (no bigram is fabricated). Multi-token / OOV entries are
// skipped. If --out already exists it is loaded first (migrating vocab if
// needed) so seeds merge into existing history rather than replacing it.
// Multiple --rime dirs are all read; overlapping words accumulate.

#include "cut.h"
#include "dict.h"
#include "score.h"
#include "sime.h"

#include <leveldb/db.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::string> SplitTab(const std::string& line) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        auto pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

struct Stats {
    std::size_t imported = 0;
    std::size_t skipped = 0;
    std::uint64_t total_commits = 0;
};

// Parse a Rime userdb value "c=<commits> d=<depth> t=<tick>". Returns the
// commit count, or 0 when the record carries no usable frequency.
std::uint32_t RimeCommits(const std::string& value) {
    auto pos = value.find("c=");
    if (pos == std::string::npos) {
        return 0;
    }
    return static_cast<std::uint32_t>(
        std::strtoul(value.c_str() + pos + 2, nullptr, 10));
}

// Resolve a default model directory holding sime.dict + sime.cnt.
// Order: the runtime install dir used by the fcitx5 plugin
// (/usr/share/sime), then the pipeline build output relative to the
// executable (build/../pipeline/output-new), then relative to cwd.
// Returns empty when none exists.
std::filesystem::path ResolveModelDir(const char* argv0) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> candidates;
    candidates.emplace_back("/usr/share/sime");
    fs::path exe = fs::absolute(fs::path(argv0), ec);
    if (!ec) {
        // exe = <repo>/build/sime-import → <repo>/pipeline/output-new
        candidates.push_back(exe.parent_path().parent_path() /
                             "pipeline" / "output-new");
    }
    candidates.push_back(fs::current_path(ec) / "pipeline" / "output-new");
    for (const auto& dir : candidates) {
        if (fs::exists(dir / "sime.dict", ec) &&
            fs::exists(dir / "sime.cnt", ec)) {
            return dir;
        }
    }
    return {};
}

// Default output path mirrors the fcitx5 plugin's userSentencePath():
// $XDG_DATA_HOME/fcitx5/sime/sentences.txt, falling back to
// $HOME/.local/share/fcitx5/sime/sentences.txt.
std::filesystem::path ResolveOutPath() {
    namespace fs = std::filesystem;
    fs::path base;
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) {
        base = fs::path(xdg);
    } else if (const char* home = std::getenv("HOME"); home && *home) {
        base = fs::path(home) / ".local/share";
    }
    if (!base.empty()) {
        return base / "fcitx5" / "sime" / "sentences.txt";
    }
    return "sentences.txt";
}

} // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path dict_path, cnt_path, in_path, out_path;
    std::vector<std::filesystem::path> rime_paths;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* a, const char* b) {
            return std::strcmp(argv[i], a) == 0 || std::strcmp(argv[i], b) == 0;
        };
        if (match("--dict", "-d") && i + 1 < argc) {
            dict_path = argv[++i];
        } else if (match("--cnt", "-c") && i + 1 < argc) {
            cnt_path = argv[++i];
        } else if (match("--in", "-i") && i + 1 < argc) {
            in_path = argv[++i];
        } else if (match("--rime", "-r") && i + 1 < argc) {
            rime_paths.emplace_back(argv[++i]);
        } else if (match("--out", "-o") && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::cerr << "Usage: sime-import [--dict <sime.dict>] [--cnt <sime.cnt>] "
                         "(--rime <userdb>... | --in <tsv>) [--out <sentences.txt>]\n"
                         "  --dict/--cnt default to /usr/share/sime or "
                         "pipeline/output-new when omitted.\n"
                         "  --out defaults to "
                         "$XDG_DATA_HOME/fcitx5/sime/sentences.txt.\n";
            return 1;
        }
    }
    if (in_path.empty() && rime_paths.empty()) {
        std::cerr << "At least one of --rime/--in is required\n";
        return 1;
    }
    if (out_path.empty()) {
        out_path = ResolveOutPath();
    }
    std::cerr << "Out: " << out_path << "\n";

    // Default model location when --dict/--cnt are not given explicitly.
    if (dict_path.empty() || cnt_path.empty()) {
        auto dir = ResolveModelDir(argv[0]);
        if (dir.empty()) {
            std::cerr << "No default model found (looked in /usr/share/sime "
                         "and pipeline/output-new); specify --dict and --cnt\n";
            return 1;
        }
        if (dict_path.empty()) dict_path = dir / "sime.dict";
        if (cnt_path.empty()) cnt_path = dir / "sime.cnt";
    }
    std::cerr << "Dict: " << dict_path << "\nModel: " << cnt_path << "\n";

    sime::Dict dict;
    if (!dict.Load(dict_path)) {
        std::cerr << "Failed to load dict: " << dict_path << "\n";
        return 1;
    }
    sime::Scorer scorer;
    if (!scorer.Load(cnt_path)) {
        std::cerr << "Failed to load scorer: " << cnt_path << "\n";
        return 1;
    }
    sime::Cutter cutter(dict, scorer);

    sime::Sime engine(dict_path, cnt_path);
    if (!engine.Ready()) {
        std::cerr << "Failed to init engine\n";
        return 1;
    }
    engine.SetUserSentenceEnabled(true);

    // Merge into existing history if present (best effort).
    std::error_code ec;
    if (std::filesystem::exists(out_path, ec)) {
        if (engine.LoadUserSentence(out_path)) {
            std::cerr << "Merged into existing " << out_path << "\n";
        }
    }

    Stats stats;
    auto seed = [&](const std::string& text, std::uint32_t commits) {
        if (text.empty() || commits == 0) {
            ++stats.skipped;
            return;
        }
        auto cuts = cutter.Cut(text);
        if (cuts.size() == 1 && !cuts[0].is_unk) {
            engine.SeedUserUnigram(cuts[0].id, commits);
            ++stats.imported;
            stats.total_commits += commits;
        } else {
            ++stats.skipped;
        }
    };

    // Raw Rime userdb (LevelDB) input.
    for (const auto& rp : rime_paths) {
        leveldb::DB* db = nullptr;
        leveldb::Options opts;
        opts.create_if_missing = false;
        auto st = leveldb::DB::Open(opts, rp.string(), &db);
        if (!st.ok()) {
            std::cerr << "Failed to open userdb " << rp << ": "
                      << st.ToString() << "\n";
            return 1;
        }
        std::unique_ptr<leveldb::DB> db_guard(db);
        std::unique_ptr<leveldb::Iterator> it(
            db->NewIterator(leveldb::ReadOptions()));
        std::size_t records = 0;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();
            std::string value = it->value().ToString();
            auto tab = key.find('\t');
            if (tab == std::string::npos) {
                continue;  // metadata key (db_name / version / tick ...)
            }
            std::uint32_t commits = RimeCommits(value);
            if (commits == 0) {
                continue;
            }
            ++records;
            seed(key.substr(tab + 1), commits);
        }
        std::cerr << "Read " << records << " records from " << rp << "\n";
    }

    // TSV export input.
    if (!in_path.empty()) {
        std::ifstream in(in_path);
        if (!in.is_open()) {
            std::cerr << "Failed to open input: " << in_path << "\n";
            return 1;
        }
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (first) {
                first = false;
                if (line.rfind("pinyin\t", 0) == 0) continue;  // header
            }
            auto fields = SplitTab(line);
            if (fields.size() < 3) {
                ++stats.skipped;
                continue;
            }
            std::uint32_t commits = 0;
            try {
                commits = static_cast<std::uint32_t>(std::stoul(fields[2]));
            } catch (...) {
                ++stats.skipped;
                continue;
            }
            seed(fields[1], commits);
        }
    }

    if (!engine.SaveUserSentence(out_path)) {
        std::cerr << "Failed to save: " << out_path << "\n";
        return 1;
    }
    std::cerr << "Imported " << stats.imported << " words ("
              << stats.total_commits << " commits), skipped " << stats.skipped
              << " -> " << out_path << "\n";
    return 0;
}