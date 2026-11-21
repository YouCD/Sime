#include "pinyin_correct.h"

#include "dict.h"

#include <algorithm>
#include <unordered_map>

namespace sime::pinyin_correct {

namespace {

void AddIfValid(std::vector<std::string>& out, std::string_view input,
                std::string candidate) {
    if (candidate == input) return;
    if (Dict::PinyinSyllableSet().contains(candidate))
        out.push_back(std::move(candidate));
}

std::string_view KeyboardNeighbors(char c) {
    static const std::unordered_map<char, std::string_view> map = {
        {'q', "was"}, {'w', "qeasd"}, {'e', "wrsdf"}, {'r', "etdfg"},
        {'t', "ryfgh"}, {'y', "tughj"}, {'u', "yihjk"}, {'i', "uojkl"},
        {'o', "ipkl"}, {'p', "ol"},
        {'a', "qwszx"}, {'s', "awdezx"}, {'d', "sefrxc"}, {'f', "drgtcv"},
        {'g', "fthyvb"}, {'h', "gyjubn"}, {'j', "hukinm"}, {'k', "jilom"},
        {'l', "kop"},
        {'z', "asx"}, {'x', "zsdc"}, {'c', "xdfv"}, {'v', "cfgb"},
        {'b', "vghn"}, {'n', "bhjm"}, {'m', "njk"},
    };
    auto it = map.find(c);
    return it == map.end() ? std::string_view{} : it->second;
}

void Dedup(std::vector<std::string>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

void AddIfValidT9(std::vector<std::string>& out, std::string_view input,
                  std::string candidate) {
    if (candidate == input) return;
    if (Dict::IsKnownT9Syllable(candidate)) out.push_back(std::move(candidate));
}

// Orthogonal neighbours on the phone keypad (1 carries no letters):
//   2 3
//   4 5 6
//   7 8 9
std::string_view KeypadNeighbors(char c) {
    switch (c) {
        case '2': return "35";
        case '3': return "26";
        case '4': return "57";
        case '5': return "2468";
        case '6': return "359";
        case '7': return "48";
        case '8': return "579";
        case '9': return "68";
        default: return {};
    }
}

} // namespace

std::vector<std::string> FuzzyVariants(std::string_view syllable) {
    std::vector<std::string> out;
    const std::string s(syllable);
    if (s.size() < 2) return out;

    auto starts = [&](std::string_view p) {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    };
    auto ends = [&](std::string_view p) {
        return s.size() >= p.size()
            && s.compare(s.size() - p.size(), p.size(), p) == 0;
    };

    // Flat / retroflex initials: zh <-> z, ch <-> c, sh <-> s.
    if (starts("zh")) AddIfValid(out, s, "z" + s.substr(2));
    else if (starts("z")) AddIfValid(out, s, "zh" + s.substr(1));
    if (starts("ch")) AddIfValid(out, s, "c" + s.substr(2));
    else if (starts("c")) AddIfValid(out, s, "ch" + s.substr(1));
    if (starts("sh")) AddIfValid(out, s, "s" + s.substr(2));
    else if (starts("s")) AddIfValid(out, s, "sh" + s.substr(1));

    // Nasal / lateral initial: n <-> l.
    if (s[0] == 'n') AddIfValid(out, s, "l" + s.substr(1));
    else if (s[0] == 'l') AddIfValid(out, s, "n" + s.substr(1));

    // Front / back nasal finals: ang <-> an, eng <-> en, ing <-> in.
    if (ends("ang")) AddIfValid(out, s, s.substr(0, s.size() - 1));
    else if (ends("an")) AddIfValid(out, s, s + "g");
    if (ends("eng")) AddIfValid(out, s, s.substr(0, s.size() - 1));
    else if (ends("en")) AddIfValid(out, s, s + "g");
    if (ends("ing")) AddIfValid(out, s, s.substr(0, s.size() - 1));
    else if (ends("in")) AddIfValid(out, s, s + "g");

    Dedup(out);
    return out;
}

std::vector<std::string> TypoVariants(std::string_view segment) {
    const std::string s(segment);
    if (s.size() < 2 || s.size() > 6) return {};

    static thread_local std::unordered_map<std::string, std::vector<std::string>>
        cache;
    auto it = cache.find(s);
    if (it != cache.end()) return it->second;

    std::vector<std::string> out;
    const std::size_t n = s.size();

    // Adjacent transposition (jei -> jie).
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::string c = s;
        std::swap(c[i], c[i + 1]);
        AddIfValid(out, s, std::move(c));
    }
    // QWERTY-adjacent substitution.
    for (std::size_t i = 0; i < n; ++i) {
        for (char nb : KeyboardNeighbors(s[i])) {
            std::string c = s;
            c[i] = nb;
            AddIfValid(out, s, std::move(c));
        }
    }
    // Single deletion.
    for (std::size_t i = 0; i < n; ++i) {
        std::string c = s;
        c.erase(i, 1);
        AddIfValid(out, s, std::move(c));
    }
    // Single insertion.
    for (std::size_t i = 0; i <= n; ++i) {
        for (char letter = 'a'; letter <= 'z'; ++letter) {
            std::string c = s;
            c.insert(i, 1, letter);
            AddIfValid(out, s, std::move(c));
        }
    }

    Dedup(out);
    if (cache.size() > 8192) cache.clear();
    cache.emplace(s, out);
    return out;
}

std::vector<std::string> T9TypoVariants(std::string_view digits) {
    const std::string s(digits);
    if (s.size() < 2 || s.size() > 6) return {};

    static thread_local std::unordered_map<std::string, std::vector<std::string>>
        cache;
    auto it = cache.find(s);
    if (it != cache.end()) return it->second;

    std::vector<std::string> out;
    const std::size_t n = s.size();

    // Adjacent transposition.
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::string c = s;
        std::swap(c[i], c[i + 1]);
        AddIfValidT9(out, s, std::move(c));
    }
    // Keypad-adjacent single-digit substitution (length-preserving).
    for (std::size_t i = 0; i < n; ++i) {
        for (char nb : KeypadNeighbors(s[i])) {
            std::string c = s;
            c[i] = nb;
            AddIfValidT9(out, s, std::move(c));
        }
    }

    Dedup(out);
    if (cache.size() > 8192) cache.clear();
    cache.emplace(s, out);
    return out;
}

} // namespace sime::pinyin_correct