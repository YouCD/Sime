#!/usr/bin/env python3
"""Linux 命令行训练语料生成器（轻量注入，词表/GRU 不变）。

源（用户确认）：
  - tldr-pages/tldr      规范常用选项示例（主力）
  - TellinaTool/nl2bash  真实命令搭配（all.cm）
  - 126 .zsh_history     个人高频（按 sqrt 频次加权）

输出预切词（空格分隔 token）喂 sime-count：OOV 自动断窗、标点断上下文，
故价值集中在「命令→子命令/BSD选项/参数词」的无标点 bigram
（git commit / docker ps / ps aux / apt update …）。flag 的领先 '-' 拆成
独立标点 token（-ef → - ef），使 命令→- bigram 成立。

只发射词表内 token 构成的搭配；不扩词表、不动 GRU（§9.6/9.7 轻量原则）。
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
SRC = HERE / "sources"

# shell 操作符两侧补空格，使其成为独立 token（OOV → sime-count 断窗，
# 保留管道/序列边界，避免跨段伪造 bigram）。
_OP_RE = re.compile(r"([|;&(){}<>])")
_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def load_vocab(path):
    return set(l.strip() for l in open(path, encoding="utf-8") if l.strip())


def split_dashes(tok):
    i = 0
    while i < len(tok) and tok[i] == "-":
        i += 1
    parts = ["-"] * i
    rest = tok[i:]
    if rest:
        parts.append(rest)
    return parts


def normalize(cmd):
    cmd = _PLACEHOLDER_RE.sub(" ", cmd)
    cmd = cmd.lower()
    cmd = _OP_RE.sub(r" \1 ", cmd)
    out = []
    for tok in cmd.split():
        out.extend(split_dashes(tok))
    return " ".join(out)


def first_word(line):
    toks = line.split()
    return toks[0] if toks else ""


def iter_tldr(tldr_dir):
    for sub in ("common", "linux"):
        d = tldr_dir / "pages" / sub
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*.md")):
            text = md.read_text(encoding="utf-8", errors="ignore")
            for m in _BACKTICK_RE.finditer(text):
                yield m.group(1)


def iter_nl2bash(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line:
            yield line


def iter_zsh(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"^: \d+:\d+;(.*)$", line.rstrip("\n"))
        cmd = m.group(1) if m else line.rstrip("\n")
        cmd = cmd.strip()
        if cmd:
            yield cmd


def collect(lines, vocab, min_tokens=2):
    """规范化 + 过滤：首词须在词表，且至少 min_tokens 个 token（聚焦 bigram）。"""
    kept = []
    for raw in lines:
        norm = normalize(raw)
        toks = norm.split()
        if len(toks) < min_tokens:
            continue
        if first_word(norm) not in vocab:
            continue
        kept.append(norm)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(PIPE / "output-news3-tech" / "sime.token.dict.txt"))
    ap.add_argument("--tldr", default=str(SRC / "tldr"))
    ap.add_argument("--nl2bash", default=str(SRC / "nl2bash" / "data" / "bash" / "all.cm"))
    ap.add_argument("--zsh", default="/tmp/zsh_history_126")
    ap.add_argument("--tldr-weight", type=int, default=3)
    ap.add_argument("--nl2bash-weight", type=int, default=2)
    ap.add_argument("--zsh-cap", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "linux.cut.txt"))
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    print(f"vocab: {len(vocab)}")

    tldr = collect(iter_tldr(Path(args.tldr)), vocab)
    nl = collect(iter_nl2bash(args.nl2bash), vocab)
    zsh_raw = list(iter_zsh(args.zsh))
    zsh = collect(zsh_raw, vocab)
    print(f"kept lines: tldr={len(tldr)} nl2bash={len(nl)} zsh={len(zsh)}")

    # zsh 按规范化命令的 sqrt 频次加权（封顶），反映个人高频又不压垮通用 LM。
    zsh_freq = Counter(zsh)
    zsh_weighted = []
    for line, freq in zsh_freq.items():
        w = max(1, min(args.zsh_cap, round(math.sqrt(freq))))
        zsh_weighted.extend([line] * w)

    corpus = []
    corpus.extend(tldr * args.tldr_weight)
    corpus.extend(nl * args.nl2bash_weight)
    corpus.extend(zsh_weighted)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")
    print(f"corpus lines: {len(corpus)} "
          f"(tldr×{args.tldr_weight}={len(tldr)*args.tldr_weight}, "
          f"nl2bash×{args.nl2bash_weight}={len(nl)*args.nl2bash_weight}, "
          f"zsh加权={len(zsh_weighted)}) -> {args.out}")

    # 预览高频命令→次词 bigram（诊断预测效果）。
    bg = Counter()
    for line in set(tldr + nl + zsh):
        toks = line.split()
        for a, b in zip(toks, toks[1:]):
            if a in vocab and b in vocab and b != "-":
                bg[(a, b)] += 1
    print("top command bigrams (sample):")
    for (a, b), n in bg.most_common(25):
        print(f"  {n:4d}  {a} -> {b}")


if __name__ == "__main__":
    main()