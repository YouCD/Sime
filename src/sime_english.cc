#include "sime_detail.h"

#include "dict.h"

#include <algorithm>
#include <string>
#include <string_view>

namespace sime::detail {

namespace {

// Curated common short English commands / tech words (≤4 letters). A
// full-cover English exact match of one of these is confident developer
// intent, so it gets EnglishWhitelistPenalty instead of the heavy
// short-English penalty. Sorted for binary_search. Deliberately excludes
// valid pinyin syllables (ping / lan / wan / man / long / su / du) and
// protected 2-letter coincidences (nm / us / be / up / key) so CN 简拼
// and full-pinyin paths are not displaced.
constexpr std::string_view kCommonEnglish[] = {
    "adb", "ant", "api", "apt", "ar", "as", "awk", "aws",
    "bash", "bc", "bios", "brew", "buck",
    "cat", "cc", "cd", "cli", "code", "cors", "cpu", "cron", "csrf",
    "css", "curl",
    "dd", "ddos", "deb", "df", "dhcp", "diff", "dns", "dom", "dpkg",
    "echo", "ed", "env", "etcd", "ex",
    "find", "fish", "fsck", "ftp",
    "gcc", "gdb", "ghc", "git", "go", "grep", "grub", "gui", "gzip",
    "halt", "hdmi", "head", "helm", "html", "htop", "http",
    "ide", "init", "ipv4", "ipv6",
    "java", "jq", "json", "jwt",
    "kill", "ksh", "kvm",
    "less", "lldb", "llvm", "ln", "ls", "lsof", "lua", "lxc", "lxd",
    "make", "mkfs", "mvc", "mvvm",
    "nat", "nice", "nl", "node", "npm", "nvme",
    "od", "oop",
    "pci", "pcie", "perf", "perl", "pip", "psql", "pwa", "pwd",
    "qemu",
    "raid", "ram", "rar", "rest", "rpc", "rpm", "ruby", "rust",
    "sata", "sbt", "scp", "sd", "sed", "seq", "sh", "snap", "sort",
    "spa", "sql", "ssd", "ssh", "sudo", "svn",
    "tail", "tar", "tcp", "tmux", "top", "tr", "tree",
    "udp", "uefi", "uniq", "uri", "url", "usb",
    "venv", "vga", "vim", "vpn",
    "wc", "wget", "who", "wifi",
    "xfs", "xss",
    "yaml", "yarn", "yum",
    "zip", "zsh", "zstd",
};

} // namespace

bool IsCommonEnglishWord(std::string_view word) {
    constexpr std::size_t n =
        sizeof(kCommonEnglish) / sizeof(kCommonEnglish[0]);
    return std::binary_search(kCommonEnglish, kCommonEnglish + n, word);
}

namespace {

// Two-letter shell / dev commands. Deliberately excludes "du" (a valid
// full pinyin syllable) and non-command coincidences (nm / us / be / up).
// Most entries also live in kCommonEnglish (strong whitelist bonus); this
// set additionally covers ps / cp / mv / rm / vi with a *gentle* lift so
// they surface without crushing CN 简拼 rivals. Sorted for binary_search.
constexpr std::string_view kTwoLetterCommands[] = {
    "cd", "cp", "dd", "df", "jq", "ln", "ls",
    "mv", "ps", "rm", "sh", "vi", "wc",
};

} // namespace

bool IsTwoLetterCommand(std::string_view word) {
    if (word.size() != 2) return false;
    constexpr std::size_t n =
        sizeof(kTwoLetterCommands) / sizeof(kTwoLetterCommands[0]);
    return std::binary_search(kTwoLetterCommands, kTwoLetterCommands + n, word);
}

float_t ComputeEnglishPenalty(std::string_view span, bool has_cn_rival,
                              bool has_exact_cn_rival, bool whitelist_on,
                              float_t light, float_t heavy, float_t bonus) {
    // Mid-sentence English span (not full-cover).
    if (whitelist_on && IsCommonEnglishWord(span)) {
        // Confident dev word: only an *exact* CN reading blocks it. A
        // speculative abbreviation/expansion rival (grep → 热盘) must not
        // suppress a curated command.
        return has_exact_cn_rival ? heavy : bonus;
    }
    // Other words relax only when there is no CN-track rival at all AND the
    // span is not itself a known pinyin syllable — otherwise women / song /
    // long / ding would be wrongly lifted over their Chinese readings.
    if (has_cn_rival || Dict::IsKnownPinyin(std::string(span))) return heavy;
    if (span.size() >= 4) return light;
    return heavy;
}

} // namespace sime::detail