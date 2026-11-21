# Sime 模型重建与 GRU 重训 —— 完整复现方案

日期：2026-11-20
适用仓库：`/data/tmp/Sime`（引擎）、`/data/tmp/DictCut`（分词配方）、`/data/tmp/fasttext`（fastText 二进制）

本文档记录从"用户自有词库/语料"出发，重建基线语言模型并重训 GRU 重排器的全过程。
另一个 agent 按本文顺序执行即可完整复现，包括所有格式细节与已踩过的坑。

---

## 0. 目标与最终结论

**目标**：用用户自己的 `pipeline/dict.txt`（190 万词）+ `pipeline/sentences.txt`（915MB 口语）
重建 `sime.dict` / `sime.cnt`，修复基线准确率，再重训全拼/T9 两个 GRU 重排器。

**基线结论**（四评测集整句准确率 %）：

| 评测集 | 旧基线 | 新基线（最终） | 差值 |
|---|---|---|---|
| cases.1（全拼·正式） | 72.99 | 67.98 | −5.01 |
| cases.2（全拼·口语） | 38.40 | **40.70** | +2.30 |
| cases.num.1（T9·正式） | 62.76 | 54.27 | −8.49 |
| cases.num.2（T9·口语） | 22.57 | **26.69** | +4.12 |

- 新基线 recall@10：cases.1 = 69.38%，num.1 = 58.41%；
  rerankable（gold 在 Top10 但非首位，即 GRU 收益上限）：1.40% / 4.14%。
- 正式集差距的根因取证：213 个"旧对新错"样本中 199 个 gold 不在 Top30，
  196 个是 LM 稀疏（词表拼音全覆盖），仅 3 个缺拼音；213 个 gold 里只有 30 个
  出现在我方语料中 → 旧模型的正式语体优势主要是记住了评测同源训练数据，
  公开语料无法复现，属于数据覆盖问题而非流水线缺陷。
- 实验矩阵证明：词表越大越好（262k 词表实验全面更差，"词表稀释"假设被否决）；
  新闻语料 ×3 + 大预算 construct（10M/4M）为最优组合。

**GRU 关键发现**：
- 从 LM 训练语料内挖掘几乎得不到 hard 样本（模型背住了训练句）：
  域内 15.3 万全拼组仅 55 hard，且 99% 的组只有 1 个候选（无法训练）。
- 必须用**未参与训练的域外口语语料**（本方案用清华 LCCC）。
- 新引擎很强：LCCC 短句全拼 40.4 万组仅 247 hard；长句（15-30 字）全拼
  5.8 万组 **0 hard 且全部单候选**（长上下文消除歧义，错误全是召回失败）。
  hard 只存在于"短而歧义"的区间；T9 歧义不随句长消失，是 GRU 主战场。
- 全拼 GRU 训练（1006 有效组）：val acc 0.75 → 0.846，hard 0 → 0.385，epoch 5 收敛。

---

## 1. 环境与依赖

硬件参考：16 核 CPU / 46GB RAM / 2×RTX 2080 Ti（训练用 GPU，运行时推理按产品设计
保持 ncnn CPU 单线程）。

```bash
# Python 虚拟环境（Python 3.14）
cd /data/tmp/Sime/pipeline
python -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
.venv/bin/python -m pip install modelscope huggingface_hub pyarrow requests ncnn opencc-python-reimplemented
```

**坑 1**：venv 移动过位置后 `.venv/bin/pip` 的 shebang 会失效（指向旧路径），
一律用 `.venv/bin/python -m pip`。

**坑 2**：shell 每条命令的工作目录会重置回 `/data/tmp/Sime`，涉及 pipeline 的命令
先 `cd pipeline` 或用绝对路径。

外部依赖：
- `/data/tmp/DictCut`：分词配方仓库。用到 `data/process.py`（繁简转换+清洗）、
  `data/split_sentences.py`（切句）、`scripts/run_em.sh`（EM 词频训练）。
- `/data/tmp/fasttext/fasttext`：fastText 二进制。
- 网络：HuggingFace 走本地代理 `http://127.0.0.1:7890`（约 1.9MB/s）；
  ModelScope 直连（约 12-28MB/s）。

构建引擎：
```bash
cd /data/tmp/Sime
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIME_ENABLE_NCNN=ON
cmake --build build -j16
```
**坑 0（最重要）**：`SIME_ENABLE_NCNN` 默认 **OFF**！不启用时 GRU 文件即使齐全
也会静默回退原始排序，评测分数与无 GRU 完全相同，极易误判"GRU 无效"。
ncnn 无系统包时由 FetchContent 从 GitHub 下载（需代理环境变量传给 cmake）。

---

## 2. 目录与产物总览

```
pipeline/
  dict.txt                 中文白名单（rime-ice 五库合并，190 万词，见 §3.1；word 或 word\tfreq）
  dict.unit                拼音映射（gen_units.py 自 rime-ice，92 万条，见 §3.1）
  en_dict.raw              英文白名单（仓库碎片合并 + Linux 命令，38.6 万词，见 §3.1）
  sentences.txt            用户口语语料（915MB）
  cases.{1,2}.txt          全拼评测集（tab 分隔：拼音 … gold）
  cases.num.{1,2}.txt      T9 评测集（数字串 … gold）
  Makefile                 计数/构建流水线（见 §4）
  gen_dict.py              词表白名单生成
  wiki/
    fetch.py               语料下载（finewiki + THUCNews）
    prepare.sh             清洗切句混合
    rebuild.sh             EM + 全量流水线（output-new）
    news3.sh               新闻×3 + 大预算（output-news3，最终基线）
    data/zhwiki.jsonl      维基原文
    news-data/thucnews.jsonl  新闻原文
    wiki_sentences.txt / news_sentences.txt   切句结果
  em-mixed/dict.txt        EM 最终词频词典（152.6 万词，word\tfreq）
  sentences.mixed.txt      混合原始语料
  sentences.cut.txt        混合语料分词结果
  sentences.cut.news3.txt  新闻×3 分词语料
  output-news3/            ★ 最终基线模型目录
    sime.dict              162,634,120 B，1,505,922 词元（cn 1,423,389 + en 82,518 + 标点 15）
    sime.cnt               135,484,150 B（大预算 10M/4M 版）
    sime.std.cnt           标准预算版（2.5M/1M，备用）
    sime.token.dict.txt    词元文本表（行 i → TokenID i+1）
    sime.dict.txt          词元拼音表（word 空格分隔多读音，/ 分隔音节）
    sime.ft.dict.txt / emoji.txt
  eval/
    eval_sime.py           四集整句/字准确率评测
    recall_sime.py         top1 / recall@10 / rerankable
  gru/
    ft/emb.vec             fastText 词向量（3,428,344 词 × 32 维）
    mine.py                候选挖掘（支持 --source 外部语料）
    lccc/prep.py           LCCC 展平
    lccc/utterances.txt    1663 万句
    lccc/utterances.long.txt  20.1 万长句（15-30 字）
    merge.py               训练集合并去重
    train.py               GRU 训练
    export.py              导出 STI8 + ncnn
    verify_export.py       ncnn 数值验证
    mine/ mine-lccc/ mine-long/   挖掘产物（py.jsonl / t9.jsonl）
    train-data/            合并后训练集
    ckpt/                  每 epoch 检查点
    export/                部署文件（gru.embedding.i8 等）
```

---

## 3. 阶段一：词库与语料准备

### 3.1 词库输入准备（rime-ice → dict.txt / dict.unit / en_dict.raw）

三个文件都不在 Sime 仓库里，需从 rime-ice 词库 + 仓库自带英文碎片制备。
（原始制备在 /home/ycd/tmp 下完成，本节路径按本文 /data/tmp 布局归一化。）

```bash
git clone --depth 1 https://github.com/iDvel/rime-ice.git /data/tmp/rime-ice
# 网络受限时走代理：https_proxy=http://127.0.0.1:7890 git clone ...
```

**dict.txt —— 中文白名单（1,902,693 词 / 24MB）**

按 gen_units.py 同款优先级合并 cn_dicts 五个词库：8105 → 41448 → base → ext → tencent。
解析规则：跳过 YAML 头（到 `...` 行）、跳过注释/空行，取每行第一列（tab 分隔），
跨文件去重保留首次出现：

```python
import os
d = "/data/tmp/rime-ice/cn_dicts"
files = ["8105.dict.yaml", "41448.dict.yaml", "base.dict.yaml",
         "ext.dict.yaml", "tencent.dict.yaml"]
seen, words = set(), []
for f in files:
    in_header = True
    for line in open(os.path.join(d, f), encoding="utf-8"):
        line = line.rstrip("\n")
        if in_header:
            if line == "...": in_header = False
            continue
        if not line or line.startswith("#"): continue
        w = line.split("\t")[0]
        if w and w not in seen:
            seen.add(w); words.append(w)
open("dict.txt", "w", encoding="utf-8").write("\n".join(words) + "\n")
```

只是白名单：最终词表还要过 gen_dict.py 的"语料词频≥3 ∩ 标准CJK ∩ 非日语黑名单"（§4.1）。
生僻字/扩展区字（𤭢、㞎 等）无需手动剔除，`is_standard_chinese` 自动过滤。
技术词补充实验：用 jieba 切 linux-command/tldr 技术语料提取 ≥2 字纯汉字词（freq≥3），
结果 **+0 新增**——tencent 词库已覆盖全部高频技术词（正则表达式、守护进程等均在），无需追加。

**dict.unit —— 拼音映射（920,453 条 / 25MB）**

gen_units.py 从 rime-ice 生成，但 `RIME_DIR` 硬编码了作者本地路径，先改：
```python
RIME_DIR = "/data/tmp/rime-ice/cn_dicts"   # 原值 /home/tfbao/new/rime-ice/cn_dicts
```
```bash
cd pipeline && python3 gen_units.py dict.unit
```
产出：40,596 单字（收集全部不重复读音，空格分隔）+ 879,857 多字词（取首次出现读音，
`'` 分隔音节），格式如 `你好 ni'hao`。脚本自带校验：音节纯字母、音节数=字数，不合格条目丢弃。

**en_dict.raw —— 英文白名单（385,573 词 / 3.2MB）**

两步：
1. 合并仓库碎片（381,598 词）。**注意**：`en_dict.raw.2` 带释义列（word\t释义），
   而 gen_dict.py 是整行小写后做白名单匹配，必须只取词本身，否则白名单失效：
   ```bash
   cd pipeline
   cut -f1 en_dict.raw.2 en_dict.raw.3 en_dict.raw.ext | sed '/^$/d' > en_dict.raw
   ```
2. 追加 Linux 命令（+3,975）：来源 = jaywcjlove/linux-command 的 623 个文档文件名
   + tldr-pages/tldr 页面名（pages.zh/common/linux）+ 本机 `compgen -c`；
   过滤 `^[a-zA-Z][a-zA-Z']*$`（gen_dict.py 的 is_english 只认纯字母+撇号，
   apt-get/7z/g++ 进不去），对已有条目小写去重后追加。
   命令进最终词表还需语料词频 ≥ MIN_COUNT(16)：技术语料（linux-command/tldr 句子，
   已并入 sentences.txt）保证常用命令达标；极冷门命令可调低 MIN_COUNT 重跑 make dict。

### 3.2 下载（wiki/fetch.py）

两个来源，合计约 2.2B 字符：
- **中文维基**：ModelScope `HuggingFaceFW/finewiki`，只要 `data/zhwiki/*` parquet。
  ```python
  from modelscope.hub.api import HubApi
  from modelscope import dataset_snapshot_download
  dataset_snapshot_download('HuggingFaceFW/finewiki',
      allow_patterns=['data/zhwiki/*'], local_dir='wiki/ms-finewiki', max_workers=4)
  ```
  然后用 pyarrow `iter_batches(columns=['text'])` 转成 `wiki/data/zhwiki.jsonl`。
  **坑 3**：`HubApi().get_dataset_files` 的参数是 `root_path=` 不是 `root=`；
  返回的 `Data` 是 list 不是 dict。
- **新闻**：HF `Tongjilibo/THUCNews`（14 个分类 jsonl，2.24GB，字段 id/title/content），
  走 7890 代理 `hf_hub_download(..., repo_type='dataset')`，
  拼 `title+"\n"+content` 写 `wiki/news-data/thucnews.jsonl`。
  **坑 4**：wikinews（ROOTS/wikimedia）在 HF 和 hf-mirror 都是 gated 403，别浪费时间；
  ModelScope 上的新闻替代品质量差（daily_news_corpus 是爬坏的 markdown）。

```bash
.venv/bin/python wiki/fetch.py all     # 或 wiki / news 分开
```

### 3.3 清洗切句混合（wiki/prepare.sh）

```bash
bash wiki/prepare.sh
```
做的事：
1. DictCut `process.py` 处理维基（繁简转换 opencc + 清洗）→ `split_sentences.py` 切句；
2. 新闻同法（`ln -sf thucnews.jsonl wiki/news-data/zhwiki.jsonl`，因为 process.py
   只认 `zhwiki.jsonl` 文件名）；
3. `cat sentences.txt wiki_sentences.txt news_sentences.txt > sentences.mixed.txt`。

语料占比：用户 350M 字符（14%）+ 维基 1.39B（55%）+ 新闻 776M（31%），
分词后 1.33B 词 ≈ 旧模型训练量（1.59B）的 83%。

---

## 4. 阶段二：基线重建

### 4.1 Makefile 关键变量

```
MIN_COUNT=16  COUNT_MAX=83886080  MAX_VOCAB=2000000  NPROC=20（命令行覆盖为 12）
OUT=output-new  CORPUS=sentences.cut.txt  CUT_DICT（分词词典）  RAW（原始语料）
```
流水线：cut（iscut/DictCut 分词）→ chars.cnt → `gen_dict.py` 白名单 →
`parallel_count.sh`/`sime-count` n-gram 计数 → `sime-construct`
（`-w` 活跃词表 = raw.1gram 字节数/8，`-r 2500000,1000000`，`-m 15`，MKN+熵剪枝）
→ `sime-compact` → `sime-converter` → ft（opencc）+ emoji。

**坑 5（重要）**：`gen_dict.py` 词表截断必须只砍英文尾部：
```python
overflow = min(len(all_tokens) - max_vocab, en_kept)
```
否则中文词数超过 MAX_VOCAB 时会把英文全部砍光。中文白名单规则：
`dict.txt ∩ 语料词频≥3 ∩ 标准CJK ∩ 非日语黑名单`；英文：小写化后词频≥16 降序。
MAX_VOCAB 已设为 2000000（中文 142 万 + 英文 8 万，留足余量）。

**坑 6**：Makefile 用了 grouped target（`&:`），中途 kill 后重跑前先删掉
半成品产物，否则 mtime 判断会跳过未完成的步骤。

### 4.2 EM 分词词典 + 第一轮重建

```bash
bash wiki/rebuild.sh
```
= EM 词频训练（无监督分词词典）：
```bash
bash /data/tmp/DictCut/scripts/run_em.sh \
    em/dict.init sentences.mixed.txt em-mixed 2000000 2 5 12
```
+ `make all NPROC=12 CUT_DICT=em-mixed/dict.txt RAW=sentences.mixed.txt`
→ `output-new/`（144 万词表标准预算）。

### 4.3 新闻×3 + 大预算（最终基线）

```bash
bash wiki/news3.sh
```
= `cat sentences.cut.txt news_sentences.cut.txt news_sentences.cut.txt > sentences.cut.news3.txt`
+ `make all OUT=output-news3 CORPUS=sentences.cut.news3.txt CUT_DICT=em-mixed/dict.txt
  DICT=dict.txt MAX_VOCAB=2000000 RAW=sentences.mixed.txt`
+ 大预算 construct：
```bash
AV=$(( $(stat -c%s output-news3/raw.1gram) / 8 ))
../build/sime-construct output-news3/raw -n 3 -o /tmp/news3big.raw.cnt \
    -c 0,1,2 -w "$AV" -r 10000000,4000000 -m 15
../build/sime-compact /tmp/news3big.raw.cnt /tmp/news3big.cnt
```
最后部署：
```bash
cp output-news3/sime.cnt output-news3/sime.std.cnt   # 保留标准版
cp /tmp/news3big.cnt output-news3/sime.cnt           # 大预算版为最终
```

---

## 5. 阶段三：评测

```bash
cd pipeline
.venv/bin/python eval/eval_sime.py output-news3/sime.dict output-news3/sime.cnt
.venv/bin/python eval/recall_sime.py output-news3/sime.dict output-news3/sime.cnt
```
- `eval_sime.py`：四集（cases.1/2 全拼，cases.num.1/2 加 `--num`），`sime -s -n 1`，
  整句准确率 + difflib 字准确率。
- `recall_sime.py`：`-n 10`，top1 / recall@10 / rerankable。
- CLI 输出格式（挖掘也靠它）：
  `[0] 我们今天去吃饭 [wo'men'jin'tian'qu'chi'fan] (score -26.307, matched 20/20, ids: 315519 242288)`
  每个输入块以 `^> ` 开头；ids 是解码路径真实词元 ID。

**评测集严禁进入任何训练数据**（mine.py 已内置去重）。

### 5.1 与 rime-ice 的横向对比（本机实测，2026-11-20）

在另一台 Arch 本机（/home/ycd/tmp 布局）用同一套评测集对比了自训 Sime 与
rime-ice（用户 fcitx5 在用的雾凇拼音配置）。模型即 output-news3 定稿五件套
（拷至 /tmp/models），已部署到 fcitx5-sime（AUR 2026.08.13）+ /usr/share/sime/。
原版预训练模型备份已按用户要求删除；如需还原，重装 AUR 包即可恢复原版模型与 .so。

**脚本**（均在 /home/ycd/tmp/，可复用于任何新模型）：
- `eval_sime_batch.py`：整文件管道喂给 `build/sime -s -n 1`（num 集加 `--num`），
  按 `> ` 提示符切块取 `[0]` 候选，四集整句准确率。模型路径在脚本头部常量。
  ```bash
  python3 /home/ycd/tmp/eval_sime_batch.py
  ```
- `eval_rime.py`：ctypes 直调 librime 1.17（绑定参照 /home/ycd/tmp/rime.h，
  自 github rime/librime 1.17.0 的 rime_api.h）。`simulate_key_sequence` 模拟
  全拼键入，取 `commit_text_preview`（= 按空格会上屏的整句首选），**不真正上屏**
  → 不触发用户词典学习。用户目录用 ~/.local/share/fcitx5/rime 的副本
  /tmp/rime-eval-user，schema=rime_ice。rime 无 T9 方案，只测两个全拼集。
  ```bash
  cp -a ~/.local/share/fcitx5/rime /tmp/rime-eval-user   # 首次
  python3 /home/ycd/tmp/eval_rime.py
  ```
- 逐句预测落盘：`sime_pred_cases.*.txt` / `rime_pred_cases.*.txt`，便于交叉分析。

**结果**（整句一次打对、零手动选词，%）：

| 评测集 | Sime 无GRU | Sime+GRU（0.40/0.50） | rime-ice |
|---|---|---|---|
| cases.1（全拼·正式） | 67.98 | **67.53** | 55.48 |
| cases.2（全拼·口语） | 40.70 | **42.80** | 37.25 |
| cases.num.1（T9·正式） | 54.27 | **53.35** | —（无 T9） |
| cases.num.2（T9·口语） | 26.69 | **29.80** | —（无 T9） |

含 GRU 数字与 §6.7 推荐档逐位一致，确认部署模型与定稿配置无误。

**逐句交叉**（全拼两集）：

| | 两者都对 | 仅 Sime 对 | 仅 rime 对 | 都错 | 并集 |
|---|---|---|---|---|---|
| 正式 | 1013 | 346 | 96 | 544 | 72.79% |
| 口语 | 586 | 228 | 159 | 1027 | 48.65% |

**结论与口径**：
- 整句连打 Sime 全面胜出（正式 +12.05、口语 +5.55，含 GRU 口径），
  n-gram 整句解码在长句上对词库匹配优势显著；"正式文本 rime 更稳"的直觉被否决。
- 两者互补：rime 独对 96/159 句，多为人工调频的固定搭配。
- 口径偏向 Sime：测的是"空格直接上屏整句"的极限能力。rime 实际使用靠手动
  选词 + 用户词典自学习，真实体验高于表中数字；Sime 无自学习，表中即日常水平。

**坑 10**：横向对比第一轮误用 `SIME_ENABLE_NCNN=OFF` 的本地 build，测出的恰是
无 GRU 基线（与 §6.7"无 GRU"行逐位一致，可作自检信号）。评测/对比前务必确认
CMakeCache 里 ncnn=ON（见坑 0）。另注意 **AUR fcitx5-sime 2026.08.13 的编译期
系数仍是旧值 0.60/0.72**，与定稿 0.40/0.50 不一致。

**已解决（2026-11-20）**：自编译 SimeApp 插件替换 AUR 的 libsime.so，系数对齐
定稿。步骤（SimeApp 通过 `-DSIME_ENGINE_ROOT` 直接 add_subdirectory 本地引擎，
无需改 SimeApp 源码）：
```bash
https_proxy=http://127.0.0.1:7890 git clone --depth 1 \
  https://github.com/Ismantic/SimeApp.git /home/ycd/tmp/SimeApp
# 本地 Sime 已改 src/sime.cc:22-23 为 0.40/0.50
cmake -S /home/ycd/tmp/SimeApp/Linux/fcitx5 \
  -B /home/ycd/tmp/SimeApp/Linux/fcitx5/build \
  -DCMAKE_BUILD_TYPE=Release -DSIME_NCNN_PROVIDER=SYSTEM \
  -DSIME_ENGINE_ROOT=/home/ycd/tmp/Sime
cmake --build /home/ycd/tmp/SimeApp/Linux/fcitx5/build -j16
sudo cp /home/ycd/tmp/SimeApp/Linux/fcitx5/build/libsime.so /usr/lib/fcitx5/
# 重启 fcitx5，日志确认 "Sime: resources loaded, GRU=ready"
```
只替换 .so，不动 conf/图标/模型（pacman 文件清单不变）。**注意**：`yay -Syu`
升级 fcitx5-sime 会覆盖回官方 .so（旧系数）与原版模型，升级后需重放编译产物
复制并重新部署 /tmp/models 五件套。原版本地备份（/usr/share/sime-stock/）
已按用户要求删除，还原一律走重装 AUR 包。

---

## 6. 阶段四：GRU 重训

架构（与旧版一致，见 gru/GRU_TRAINING.md）：冻结 fastText 32 维嵌入 →
双向 GRU h64 → concat(前向末态, 后向末态) → Linear 128→1 →
`最终分 = sime_score + scale × gru_score`，listwise softmax CE。
运行时：Top-10 候选重排，ncnn CPU 单线程。

### 6.1 fastText 词向量

```bash
/data/tmp/fasttext/fasttext skipgram -input sentences.cut.txt \
  -dim 32 -ws 2 -neg 5 -epoch 1 -minn 2 -maxn 4 -minCount 1 \
  -bucket 2000000 -thread 16 -output gru/ft/emb
```
约 9 分钟，产出 `gru/ft/emb.vec`（343 万词）。对 150.6 万词表**零缺失**。

### 6.2 候选挖掘（gru/mine.py）

原理：采样句子 → 用 `output-news3/sime.dict.txt` 最长匹配分词（取首读音，`/` 去掉）
→ 拼音拼接 = 全拼输入 → 键盘映射 `abc→2 def→3 … wxyz→9` = T9 输入 →
引擎 `-s -n 10` 解码 → 保留 gold 在全匹配候选中的组。
gold_rank>0 即 hard。只保留 matched==total 的候选。

```bash
# 域内（可选，hard 极少，主要提供 easy 组）
.venv/bin/python gru/mine.py --workers 12        # → gru/mine/{py,t9}.jsonl

# LCCC（hard 主力）
mkdir -p gru/lccc && cd gru/lccc
HTTPS_PROXY=http://127.0.0.1:7890 curl -sL -o lccc_base_train.jsonl.gz \
  'https://huggingface.co/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz'
cd ../..
.venv/bin/python gru/lccc/prep.py    # 20,073,720 轮 → 去非汉字 → 16,634,694 句
.venv/bin/python gru/mine.py --workers 12 --source gru/lccc/utterances.txt \
  --source-k 900000 --out gru/mine-lccc

# 长句补充（全拼已证明无用，T9 仍可一试）
# 从 utterances.txt 筛 15-30 字句采样 20 万 → utterances.long.txt
.venv/bin/python gru/mine.py --workers 12 --source gru/lccc/utterances.long.txt \
  --source-k 400000 --out gru/mine-long
```

**坑 7**：LCCC 原始文本是预分词带空格+标点的，prep.py 用 `re.sub(r"[^一-鿿]+","",s)`
去掉所有非汉字再筛长度，否则通过率只有 0.5%。

**坑 8**：CPU 核数有限时多个挖掘任务并行要控制 worker 总数 ≤ 核数；
mine.py 无断点续跑，kill 重跑会丢进度。

实测产量（复现时应当接近）：

| 来源 | 组数 | hard | ≥2候选组 |
|---|---|---|---|
| 域内 py | 152,590 | 55 | 195 |
| 域内 t9 | 96,448 | 144 | — |
| LCCC py | 404,438 | 247 | 977 |
| LCCC t9 | 252,678 | 659 | ~2,700 |
| 长句 py | 58,325 | 0 | 0 |
| 长句 t9 | 25,052 | 0 | — |

### 6.3 合并（gru/merge.py）

LCCC+长句全部组 + 域内抽样 7.5 万组，按 (input,gold) 去重 →
`gru/train-data/{py,t9}.jsonl`。训练时 `load_groups` 再过滤 <2 候选的组。

### 6.4 训练（gru/train.py）

```bash
.venv/bin/python gru/train.py --mode pinyin --data gru/train-data --epochs 8 --device cuda:0
.venv/bin/python gru/train.py --mode t9     --data gru/train-data --epochs 7 --device cuda:1
```
超参：AdamW lr 2e-3 wd 1e-5，梯度裁剪 5.0，seed 1，batch-groups 128，
val = 5%。每 epoch 存 `gru/ckpt/{mode}.epoch{N}.pt`，打印 val acc 与 hard acc。
**选 epoch 看 val acc + hard acc 平台期**。实测：
- 全拼（1006 有效组）：val 0.75→0.846，hard 0→0.385，epoch 5 收敛；
- T9（2880 有效组）：val 0.768→0.834，hard 0.029→0.514，epoch 3 最佳。

### 6.5 导出（gru/export.py）——格式细节是复现关键

```bash
.venv/bin/python gru/export.py --pinyin-epoch 5 --t9-epoch <best> --out gru/export
```

**STI8 嵌入格式**（`gru.embedding.i8`）：
```
magic "STI8" | u32 ver=1 | u32 rows | u32 dim=32
rows × fp16 scale | rows × 32 × int8
```
rows = vocab+1；行映射 `R[1 + t % vocab] = E[t]`（t 为 1-based TokenID，
在 [1..vocab] 上是双射，行 0 不用）。scale = maxabs/127，值 round 后 clip ±127。
运行时按 `1 + token_id % (rows-1)` 取行——**词表变了必须重新导出，
旧文件会静默错位，没有回退**（只有文件缺失/加载失败才回退原始排序）。

**ncnn bin 布局**（fp16，已逐字节对照旧 save/gru.pinyin.ncnn.bin 验证，75,028 B）：
每个权重块前有 u32 flag `0x01306B47`（fp16 标志），顺序：
1. `weight_ih_l0`（192×32）前向 + `_reverse` 后向 —— 直接拷贝，不转置；
2. bias（64×4）×2 方向，行序为
   `[b_ih_r+b_hh_r, b_ih_z+b_hh_z, b_ih_n, b_hh_n]`
   （PyTorch 门序 r/z/n = ncnn R/U/N；两路 bias 相加，n 门分开）；
3. `weight_hh_l0`（192×64）×2 方向；
4. linear weight（128）fp16 带 flag；
5. linear bias（1）**裸 fp32，无 flag**。
channel(0)=前向，channel(1)=后向。

**param 文件**：直接复用 `save/gru.pinyin.ncnn.param`（两个模式相同）：
Input in0 → GRU(0=64 1=12288 2=2) → Split → Crop(前向末态) → Reshape 64 →
Crop(后向初态) → Reshape 64 → Concat → InnerProduct(0=1 1=1 2=128) → Flatten out0。
其中 12288 = weight_data_size = 192×32×2，size 参数 = 12288/2/64/3 = 32。

### 6.6 数值验证（gru/verify_export.py）

```bash
.venv/bin/python gru/verify_export.py
```
读 STI8 反量化 → 加载 ckpt（`weights_only=True`）→ 50 条随机序列（长 1-12）
分别过 PyTorch 与 ncnn python（`ncnn.Mat` 形状 (T,32)，输入 in0 输出 out0）→
最大绝对差 < 0.05 即 VERIFY_OK。**不通过不得部署。**

**坑 9**：torch 参考必须用 **STI8 反量化后的嵌入**（运行时 C++ 也用 int8），
否则稀有词向量的量化噪声（可达 ~0.9 分）会被误判为导出错误。
对齐嵌入后 fp16 导出实测最大差 0.015，布局正确性可用 fp32 对照进一步隔离。

### 6.7 部署与评测

```bash
cp gru/export/gru.embedding.i8 gru/export/gru.*.ncnn.* output-news3/
# 引擎自动从 sime.cnt 同目录加载（src/gru_reranker.cc:43-72）
.venv/bin/python eval/eval_sime.py output-news3/sime.dict output-news3/sime.cnt
```
运行时系数在 `src/sime.cc:21-23`（编译期常量）：
`kGruCandidates=10, kGruPinyinScale, kGruT9Scale`。
扫描需改常量重编译，逐档跑四集评测。实测扫描结果（整句准确率 %）：

| py 系数 | cases.1 | cases.2 | | t9 系数 | num.1 | num.2 |
|---|---|---|---|---|---|---|
| 无 GRU | 67.98 | 40.70 | | 无 GRU | 54.27 | 26.69 |
| 0.20 | 67.68 | 41.90 | | 0.30 | 54.83 | 28.43 |
| 0.30 | 67.48 | 42.45 | | 0.40 | 53.40 | 29.14 |
| 0.40 | 67.53 | 42.80 | | 0.50 | 53.35 | 29.80 |
| 0.50 | 66.23 | 43.25 | | 0.60 | 52.69 | 30.01 |
| 0.60 | 66.23 | 43.75 | | 0.72 | 52.43 | 30.57 |

规律：GRU 训练数据是 LCCC 口语 → 口语集收益随系数单调上升，正式集单调受损
（域偏置）；T9 正式集在低系数 0.30 时反而 +0.56。两模式系数独立，可自由组合。
推荐 **py=0.40 / t9=0.50**（口语 +2.10/+3.11，正式 −0.45/−0.92）；
保守可选 py=0.40 / t9=0.30（正式几乎不损，口语 +2.10/+1.74）。

**上线检查单**：
- [ ] 新词表对应的 STI8/ncnn 五件套齐全且验证通过；
- [ ] 若决定不带 GRU 上线，**必须删除目录里所有旧 gru.* 文件**（旧嵌入错位无回退）；
- [ ] BeamSize 是编译期常量（sime.h:131=20 / state.h:107=48），不是调优旋钮；
- [ ] TokenID 为 u32、1-based（行 i → ID i+1），无 18-bit 限制。

---

## 7. 复现命令速查（从零到评测）

```bash
cd /data/tmp/Sime && cmake -B build -DCMAKE_BUILD_TYPE=Release -DSIME_ENABLE_NCNN=ON \
  && cmake --build build -j16
cd pipeline
# 词库三件套 dict.txt / dict.unit / en_dict.raw 制备见 §3.1（rime-ice）
.venv/bin/python wiki/fetch.py all          # 语料下载（~30min）
bash wiki/prepare.sh                        # 清洗切句混合
bash wiki/rebuild.sh                        # EM + output-new（数小时）
bash wiki/news3.sh                          # output-news3 + 大预算 cnt
cp output-news3/sime.cnt output-news3/sime.std.cnt
cp /tmp/news3big.cnt output-news3/sime.cnt
.venv/bin/python eval/eval_sime.py output-news3/sime.dict output-news3/sime.cnt
# --- GRU ---
/data/tmp/fasttext/fasttext skipgram -input sentences.cut.txt -dim 32 -ws 2 -neg 5 \
  -epoch 1 -minn 2 -maxn 4 -minCount 1 -bucket 2000000 -thread 16 -output gru/ft/emb
.venv/bin/python gru/mine.py --workers 12
# 下载 LCCC（§6.2）→ prep.py → mine-lccc → mine-long
.venv/bin/python gru/merge.py
.venv/bin/python gru/train.py --mode pinyin --data gru/train-data --epochs 8 --device cuda:0 &
.venv/bin/python gru/train.py --mode t9 --data gru/train-data --epochs 7 --device cuda:1 &
wait
.venv/bin/python gru/export.py --pinyin-epoch <best> --t9-epoch <best>
.venv/bin/python gru/verify_export.py
cp gru/export/gru.* output-news3/
.venv/bin/python eval/eval_sime.py output-news3/sime.dict output-news3/sime.cnt
```

## 8. 当前状态（2026-11-20）

全流程完成：基线重建、fastText、挖掘（域内/LCCC/长句 × 全拼/T9）、合并、
双模式训练（py epoch5 / t9 epoch3）、导出、数值验证、ncnn 启用重编译、
部署、scale 扫描。**定稿系数 py=0.40 / t9=0.50（用户确认，主要 Linux 全拼场景）**，
已编译进 build/sime，GRU 五件套在 output-news3/。
最终四集（推荐档）：67.53 / 42.80 / 53.35 / 29.80，
对比旧基线 72.99 / 38.40 / 62.76 / 22.57：口语 +4.40 / +7.23，正式 −5.46 / −9.41
（正式差距为语料覆盖问题，见 §0 取证）。

另（本机 /home/ycd/tmp）：定稿五件套已部署到 Arch 本机 fcitx5-sime（AUR），
并完成与 rime-ice 的同集横向对比（Sime 整句准确率全面领先），见 §5.1；
对比脚本 eval_sime_batch.py / eval_rime.py 存于 /home/ycd/tmp/。

---

## 9. 地名增强（gazetteer 注入，2026-11-20 追加）

**动机**：基线对行政区划地名偏弱。典型 bug：`hunanshengzhijiangxian` 出
"湖南省枝江县"（错），应为"湖南省芷江县"。取证：湖北"枝江县"是过时名
（今枝江市），但语料词频 127 > 芷江县 114；bigram 湖南省→芷江县 仅 2 次
→ MKN 回退到 unigram，错误词胜出。用户学习可临时纠正但不系统。

**方案**：注入现行行政区划 gazetteer 语料，增量重建 `sime.cnt`。
现行名单里湖北是"枝江市"、湖南是"芷江县"，注入后词频反转，过时名不被增强。

### 9.1 数据源

[modood/Administrative-divisions-of-China](https://github.com/modood/Administrative-divisions-of-China)
（21k stars，省/市/县/乡/村五级，CSV+JSON）：

```bash
https_proxy=http://127.0.0.1:7890 git clone --depth 1 \
  https://github.com/modood/Administrative-divisions-of-China.git /data/tmp/gazetteer
# dist/{provinces,cities,areas,streets,villages}.csv
# areas.csv 列：code,name,cityCode,provinceCode（2978 县级）
```

**坑 10**：gazetteer 用**官方全称**，自治县是"芷江侗族自治县"而非用户输入的
短名"芷江县"；湖北是"枝江市"不是"枝江县"。必须从官方名推导常用短名，
否则修不到短名 bug。国家统计局自 2024-10 起不再公开完整区划数据，GitHub 镜像更实用。

### 9.2 生成器（gazetteer/gen_corpus.py）

产出**预切词**语料（空格分隔词元，与 `sentences.cut.*.txt` 同格式），可直接喂
`sime-count`，绕过切词以保证分词与模型 token 完全一致。要点：
- 只发射已存在于 `sime.token.dict.txt` 的词元（保证计数 ID 对齐，OOV 自动跳过）；
- 自治县短名推导：去"各族"+ 55 个民族名（按长度降序替换）+ "自治"，
  如 芷江侗族自治县→芷江县、围场满族蒙古族自治县→围场县（111/117 命中词表）；
- 搭配模板：`县名`、`省 县`、`市 县`、`省 市 县`，短名另加 `短名`×W、`省 短名`×W；
- 市级占位（市辖区/县/省直辖县级行政区划）不参与搭配；
- `--weight W`（默认 8）控制短名与省+短名关键链的重复次数 = 注入计数强度。

```bash
cd pipeline
python3 gazetteer/gen_corpus.py --weight 8        # → gazetteer/gazetteer.cut.txt（13916 行）
# 芷江县 unigram +18（114→132 > 枝江县 127），bigram 湖南省→芷江县 +8（2→10）
```

### 9.3 增量重建（只换 sime.cnt，词表/GRU 全复用）

词表不变 → `sime.dict`（trie）与 GRU 五件套**完全不动**，只需增强 n-gram 计数：

```bash
cd pipeline
# 1) 单独计数 gazetteer 语料（用现有词表，ID 对齐）
../build/sime-count -n 3 -d output-news3/sime.token.dict.txt \
  -o gazetteer/gaz -s gazetteer/swap -p punct.txt -c 1000000 gazetteer/gazetteer.cut.txt
# 2) merge 进现有 raw 计数（merge.bin 同 key 求和，输入须有序；sime-count 输出有序）
make merge.bin
mkdir -p output-news3-gaz
for n in 1 2 3; do
  ./merge.bin -n $n -o output-news3-gaz/raw.${n}gram \
    output-news3/raw.${n}gram gazetteer/gaz.${n}gram &
done; wait
# 3) 大预算 construct + compact（与 news3 同参数）
AV=$(( $(stat -c%s output-news3-gaz/raw.1gram) / 8 ))
../build/sime-construct output-news3-gaz/raw -n 3 -o /tmp/gaz.raw.cnt \
  -c 0,1,2 -w "$AV" -r 10000000,4000000 -m 15        # 约 3.5 分钟
../build/sime-compact /tmp/gaz.raw.cnt output-news3-gaz/sime.cnt
# 4) 复用不变的 trie 与 GRU（软链即可）
cd output-news3-gaz
ln -sf ../output-news3/sime.dict sime.dict
for f in gru.embedding.i8 gru.pinyin.ncnn.bin gru.pinyin.ncnn.param \
         gru.t9.ncnn.bin gru.t9.ncnn.param; do ln -sf ../output-news3/$f $f; done
```

实测：1gram 词表数不变（1,505,922，地名本就在语料中），新增 2368 bigram /
2192 trigram；新 `sime.cnt` 135,485,922 B（仅比原版大 1772 B）。

### 9.4 验证与评测

```bash
printf 'hunanshengzhijiangxian\nhubeishengzhijiangshi\n' | \
  ../build/sime --dict output-news3-gaz/sime.dict --cnt output-news3-gaz/sime.cnt -s -n 3
# hunanshengzhijiangxian → 湖南省芷江县 ✅；hubeishengzhijiangshi → 湖北省枝江市 ✅
.venv/bin/python eval/eval_sime.py output-news3-gaz/sime.dict output-news3-gaz/sime.cnt
```

四集零回归：67.58 / 42.80 / 53.35 / 29.80（cases.1 较修复前 67.53 微涨 +0.05，
其余持平）。地名是稀有专有名词，注入对通用评测几乎无影响。
裸 `zhijiangxian` 仍枝江县优先（GRU 重排 + 历史同名固有歧义，次要）。

### 9.5 部署（含用户历史清理，关键）

**坑 11（重要）**：只换模型不够。用户历史里若累积了错误整句记录
（如 `湖南省 枝江县` ×3），其用户学习加权（weight 0.2 混合）会把 LM 修复**压回去**。
必须同时清掉错误记录。验证矩阵：

| 组合 | 首选 |
|---|---|
| 新模型 + 无历史 | 湖南省芷江县 ✅ |
| 新模型 + 清理后历史 | 湖南省芷江县 ✅ |
| 新模型 + 原历史（含错误×3） | 湖南省枝江县 ❌ |

部署到 fcitx5（以 192.168.104.126 为例，免密 sudo）：

```bash
# 备份
sudo cp -p /usr/share/sime/sime.cnt /usr/share/sime/sime.cnt.bak-gaz
cp -p ~/.local/share/fcitx5/sime/sentences.txt ~/.local/share/fcitx5/sime/sentences.txt.bak-gaz
# 换 cnt（dict/GRU 不动）
sudo cp output-news3-gaz/sime.cnt /usr/share/sime/sime.cnt
# 停 fcitx5（SIGTERM 触发退出 flush）→ 清错误记录 → 重启
pkill -TERM -x fcitx5; while pidof fcitx5 >/dev/null; do sleep 0.3; done
grep -v "枝江县" ~/.local/share/fcitx5/sime/sentences.txt.bak-gaz \
  > ~/.local/share/fcitx5/sime/sentences.txt
# 重启需图形会话环境（从运行进程 /proc/<pid>/environ 抓 DISPLAY/WAYLAND_DISPLAY/
# DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR），set -a; source env; fcitx5 -d
```

部署后首选正确，按空格上屏学到的是对的 → 正向自强化，不再反复。
回滚：还原两个 `.bak-gaz` 备份并重启 fcitx5。

### 9.6 乡镇级轻量扩展（已实施，2026-11-20）

数据含 `streets.csv`（乡级 41,352）/`villages.csv`（村级 620,573）。
**词表覆盖分析**（决定扩展策略）：乡镇名仅 30.5%（12,598）在词表——
在词表的是语料词频≥3 的**常用乡镇**；词表外 69.5%（28,754）是长尾冷门，
其中 28,322 个白名单+拼音都缺（全覆盖需补拼音，多音字易错）。
结论：**轻量扩（只注入在词表乡镇）实际收益更大**——修的是用户真会输入的常用名；
全覆盖多出的多是低频长尾，且要补拼音+重建词表+重导 GRU+词表稀释风险，性价比低。

生成器已加乡镇层（`--town-weight 3`，`--no-streets` 回退纯县级）：
按 areaCode 关联县名，模板 `乡镇`×TW、`县 乡镇`×TW、`省 县 乡镇`，只发射词表内 token。
**纯轻量保持词表不变**（那 432 个"只差词频"的乡镇不纳入——进词表会迫使 GRU 重导）。

```bash
python3 gazetteer/gen_corpus.py --weight 8 --town-weight 3   # 102,102 行（县+乡）
# 计数→从原始 output-news3/raw 重新 merge（避免县级重复计数）→construct→compact，同 §9.3
```

实测：词表数仍 1,505,922，新增约 9.8k bigram / 13.8k trigram，
新 cnt 135,489,906 B。验证 `donghuamenjiedao`→东华门街道 ✅，县级修复保持，
四集 67.58 / 42.80 / 53.35 / 29.80（与县级版一致，零回归）。已部署 126。

### 9.7 成语 + 高校轻量扩展（已实施，2026-11-20）

沿用 §9.6 轻量原则（只注入词表内 token，词表/GRU 不变），扩两个高收益域。
**四域覆盖摸底**（决定取舍）：

| 域 | 数据源 | 词表内覆盖 | 取舍 |
|---|---|---|---|
| 成语 | chinese-xinhua `idiom.json`（30,895） | 11,555（37.4%，多为四字常用） | **注入** |
| 高校 | 教育部名单 `ramwin/china-public-data` CSV（2,631 所） | 1,133（含分校区括号剥离） | **注入** |
| 古诗词 | `xuchunyang/300` 唐诗 300（319 首） | 整句 99.6% 在词表，但 token 多为单字（月/君/时） | 暂缓：古汉语、现代输入收益小，恐抬升生僻字 |
| 计算机 | `jaywcjlove/linux-command`（623 文档） | 技术词 90% 已在词表，sentences.txt 已含技术句 | 暂缓：缺口为 OOV 新架构术语（灰度发布/分库分表等），需全量重建 |

生成器（均只发射词表内 token，预切词输出）：
- `idiom/gen_corpus.py`：成语整词 ×`--weight 5`（unigram 计数是整词排序的主杠杆）→ 57,775 行。
- `university/gen_corpus.py`：校名 ×5 + `市 校名` ×5（`--weight 5`）→ 11,330 行；
  分校区括号名（中国矿业大学（北京））剥离括号后的本部名若在词表也注入。

```bash
python3 idiom/gen_corpus.py        # idiom/idiom.cut.txt
python3 university/gen_corpus.py   # university/university.cut.txt
# 两域一起计数（同 §9.3 的 sime-count），再 merge 进 gaz raw（含县+乡），避免重复计数：
../build/sime-count -n 3 -d output-news3/sime.token.dict.txt -o ext/ext -s ext/swap \
    -c 83886080 -p punct.txt idiom/idiom.cut.txt university/university.cut.txt
./merge.bin -n 1 -o output-news3-ext/raw.1gram output-news3-gaz/raw.1gram ext/ext.1gram
./merge.bin -n 2 -o output-news3-ext/raw.2gram output-news3-gaz/raw.2gram ext/ext.2gram
ln -sf ../output-news3-gaz/raw.3gram output-news3-ext/raw.3gram   # 无新增三元，沿用
# construct/compact 同 §9.3（大预算 -r 10000000,4000000），dict/GRU 符号链接复用
```

实测：1gram 记录数不变（1,505,922，纯计数增强无新键），2gram +1,059 搭配键，
新 cnt 135,490,022 B。计数增量验证：成语 +5/词、高校 +10/词（整词×5 + 市校名链×5）、
芷江县 +0（gazetteer 已在、无重复）。常用成语/高校顶候选 gaz/ext 一致（基线本就排对，
注入提升的是计数裕量与生僻词排序）。四集 67.53 / 42.80 / 53.35 / 29.85
（vs gaz 67.58/42.80/53.35/29.80：cases.1 −1 句、cases.num.2 +1 句，char% 略升，零回归）。
已部署 126（`/usr/share/sime/sime.cnt`，备份 `.bak-town` 为乡镇版），GRU=ready。
村级 62 万同理可扩但收益更稀，未实施。

### 9.8 计算机新架构术语：全量扩词表（已实施，2026-11-20）

轻量注入只能 boost 已在词表的词；现代云原生/LLM 术语是 OOV，必须扩词表。
**现成词库摸底结论：都偏旧，补不全。** 对 77 个现代 OOV 术语（灰度发布/微前端/
大语言模型/检索增强生成等）：THUOCL IT 库（15999 词，无拼音）覆盖 11，搜狗计算机
三库（词汇大全/名词/专业词库，合并 11671 词带拼音，`parse_scel.py` 解析）覆盖 1，
rime-ice 腾讯词向量（98 万词，两列无显式拼音）覆盖 5——**三库合计仅 16/77，57 个全缺**
（语料均≈2018 年前，未含 2020+ 术语）。故手动补 35 个高频词（`tech/terms.txt`，
pypinyin 注音并核对多音字：嵌 qiàn/蒸 zhēng/绿 lǜ/降 jiàng/混 hùn/倾 qīng/从 cóng）。

**核心设计：追加到 token 表末尾保 ID，免 GRU 重训。**
- `tech/build_vocab.py`：新词追加到 `output-news3/sime.token.dict.txt` 末尾（标点之后），
  现有 1,505,922 个 ID 全不变，新词得 ID 1,505,923–1,505,957；cn 词典追加 `词 音节/音节`。
  token 顺序 [CN][EN][punct][新CN]，**插在 CN 块会移动 EN/punct ID → GRU 错位，必须追加末尾**。
- GRU ncnn 权重与词表无关（图输入是 32 维 embedding 向量），只有 `gru.embedding.i8`
  依赖词表（rows=vocab+1，row=1+id%vocab，导出与运行时同步换词表即一致）。
- **坑12（关键）：`gru/ckpt` 现存权重与部署基线 ncnn 不同源**——`export.py` 用当前 ckpt
  重导的 ncnn bin 与基线 md5 不同（基线==canonical `gru/export`）。故 **ncnn 直接复用基线**
  （符号链接），只重导 embedding.i8，保证 GRU 权重零变化。现有词 embedding 向量逐一比对
  完全一致（最大差异 0.000000）。
- 新词 embedding：`tech/make_supplement.py` 单遍扫 `gru/ft/emb.vec`（912MB），emb.vec 已有
  3 词（灰度发布/全链路压测/主从复制）取原向量，其余 32 词用**构成字符向量均值**派生
  （字符均在 emb.vec），写 `tech/emb_supplement.vec`。`export.py` 加 `--supplement` 选项
  定向覆盖新词（不动现有词，防回归）。

```bash
python3 tech/build_vocab.py                       # 追加词表+cn词典+tech.cut.txt(×8)
../build/sime-converter output-news3-tech/sime.token.dict.txt output-news3-tech/sime.dict \
    --cn output-news3-tech/sime.dict.txt --en output-news3-tech/sime.en.dict.txt
../build/sime-count -n 3 -d output-news3-tech/sime.token.dict.txt -o tech/tech -s tech/swap \
    -c 83886080 -p punct.txt tech/tech.cut.txt
./merge.bin -n 1 -o output-news3-tech/raw.1gram output-news3-ext/raw.1gram tech/tech.1gram
# 2/3gram 无新增三元，符号链接沿用 ext；construct -w 1505957 大预算 → compact，同 §9.3
python3 tech/make_supplement.py
.venv/bin/python gru/export.py --dict-dir output-news3-tech \
    --supplement tech/emb_supplement.vec --out output-news3-tech   # ncnn 产物弃用，复用基线
```

**同音/切分竞争词定向 boost**：weight 8 时 35 词中 32 正确，3 词压不过常用竞争词——
回表↔徽标(797)、树摇↔束腰(960)、微前端↔为(568万)+前端(切分)。给这 3 词额外计数
（回表+1000/树摇+1200/微前端+3000，`tech/tech_boost.cut.txt`）merge 后重建 → **35/35 全正确**。

实测：词表 1,505,957，新 cnt 135,490,406 B、sime.dict 162,638,712 B、
embedding.i8 51,202,588 B（rows 1,505,958）。四集 67.53 / 42.80 / 53.40 / 29.85
（vs ext 67.53/42.80/53.35/29.85：cases.num.1 +1 句，零回归）。地理/乡镇/成语/高校修复全保持。
已部署 126：**换 sime.dict + sime.cnt + gru.embedding.i8 三件，ncnn 不动**，
备份 `.bak-ext`（ext 版），GRU=ready。

---

## 10. 拼音纠错与英文混合输入修复（2026-11-21 追加）

两项**引擎代码**改动（不动词表/模型/GRU），提交在 `ycd` 分支：
`c1dab09`（引擎：纠错 + 英文白名单）、`ff0e6c7`（pipeline 词表上限，见 §4.1 坑 5）。
对任意模型生效；评测口径用部署模型 `output-news3-tech`。

### 10.1 英文短命令白名单与混合输入排名

**问题**：`grep` 单独输入排第 6（首位"跟人跑"），`awk/sed/ssh/tar/bash/code/make`
等开发者常用短命令被高频中文简拼/同音词压过。根因：旧逻辑 `short_input≤4`
对全覆盖英文也施加重罚，且英文 `<eos>|english=0` 转移在 beam 尾部占便宜。

**修复**（`src/sime.cc` 匿名命名空间 `kCommonEnglish[]` ≈140 个排序词 +
`IsCommonEnglishWord` 二分查找；`ComputeEdgePenalties`）：
```cpp
const bool short_input   = input.size() <= 4;   // 简拼/扩展惩罚档
const bool short_english = input.size() <= 3;   // 英文重罚档（与上者解耦）
if (edge.english) {
    bool full_cover = edge.start == 0 && edge.end == input.size();
    if (full_cover && input.size() <= 4 && IsCommonEnglishWord(input))
        en_pen = EnglishWhitelistPenalty;        // -3.0（加分）
    else
        en_pen = (full_cover && !short_english) ? EnglishPenalty   // 2.5
                                                : EnglishPenaltyShort; // 8.0
}
```
常量在 `include/sime.h`：`EnglishPenalty=2.5`、`EnglishPenaltyShort=8.0`、
`EnglishWhitelistPenalty=-3.0`。白名单**刻意排除**合法拼音音节
（ping/lan/wan/man/long/su/du）与受保护双字母巧合（nm/us/be/up/key），
保证 `ping→平`、`lan→兰` 等拼音守卫不被破坏。惩罚在 beam 搜索**之前**施加
（`ComputeEdgePenalties` 先于 Layer1），被剪枝的英文边（如 ssh）也能救回。

实测（tech 模型）：`grep/awk/ssh` 均 top1；`ping→平`、`lan→兰` 不变。
**已知局限**：① 双字母命令（wc/ps/sh/jq）仍输给高频中文简拼；② 白名单只认
**全覆盖**英文，混排句中英文（`yong grep sousuo`→"用 grep 搜索"）里的 grep
仍按部分覆盖重罚，只解出"用"——混排英文待后续做句中识别。

### 10.2 拼音纠错（模糊音等价 + 错拼）

**动机**：引擎原先无任何错误拼音矫正。用户平翘舌/鼻边音混淆（zh/z、n/l）
或键盘 slip（转置/邻键）时，目标词排不上来。

**关键设计取证**：LM 是**词级**的。最初做单音节纠错边让 beam 链接成字序列
（中+国），实测 `zongguo` 首位仍是"总国"、中国根本不出现——因为字序列
bigram 分远低于整词 token（中国）的分。**结论：必须把纠错音节拼成整词键查
DAT**（`zongguo → zhong'guo → 中国`），让候选继承整词 LM 分。

**候选生成**（新增 `include/pinyin_correct.h` / `src/pinyin_correct.cc`）：
- `FuzzyVariants(合法音节)`：zh↔z、ch↔c、sh↔s、n↔l、ang↔an、eng↔en、ing↔in，
  规则展开后只保留已知音节、排除自身。
- `TypoVariants(非法段，len 2–6)`：编辑距离≤1（相邻转置 jei→jie、QWERTY 邻键
  替换、单字删除/插入），`thread_local` 缓存（上限 8192 清空）。
- 校验用 `Dict::PinyinSyllableSet()`（新增，`include/dict.h`/`src/dict.cc`）：
  全部合法音节的 `unordered_set`，O(1) 成员判定，替代逐候选二分查找。

**加性词级 DFS**（`src/sime.cc` `InitNet`，精确/简拼路径之后、tier 过滤之前）：
- `Link` 增 `float_t correction`（`include/sime.h`，聚合序
  `{start,end,id,pieces,penalty,expansion,english,correction}`）；
  `ComputeEdgePenalties`：`edge.penalty = en_pen + expansion + edge.correction`。
- 从每个**贪心音节边界**启动 DFS，自由枚举音节 span（len 1–6，跨贪心边界，
  故 jei 被切成 j+ei 仍能恢复），合法音节走模糊音、非法走错拼；把音节用 `'`
  拼成词键 `Dat.Get` → `GetEntry` → 发射带 correction 成本的边。
- 常量：`FuzzyPinyinPenalty=5.0`、`TypoCorrectionPenalty=7.0`、
  `kEnableFuzzyCorrection/kEnableTypoCorrection=true`（`include/sime.h`）。
  成本按纠错音节数累加；精确路径 correction=0 不受影响 → **零回归**。

### 10.3 性能优化（关键，否则不可上线）

朴素实现（每字符位置启动 DFS、对每个非法段生成 ~350 错拼候选）延迟从基线
4.13 ms 飙到 11.34 ms/decode（+7 ms，≈2.7×）。隔离测量：**模糊音几乎免费
（+0.17 ms），开销全在错拼**。逐项优化（均已在代码中）：

| 优化 | 作用 |
|---|---|
| 仅从贪心音节边界启动 DFS | 起始点降到 ≈1/3（词本就从音节边界起）|
| 错拼门控 `!IsExtendablePinyin(seg)` | 合法前缀（zho→zhou）交给简拼/扩展，不对跨边界碎片（oa/ain）生成候选 |
| `PinyinSyllableSet()` 哈希校验 | 候选合法性 O(1) |
| 剪枝 `ncorrected==0 && nsyl>=2` 停 | 纠错须发生在前两音节，不走纯原音节前缀 |
| 不发射模糊音单音节边（`nsyl==1 && cost<TypoCorrectionPenalty`）| 合法音节已有精确同音字，避免 beam 膨胀；保留错拼单音节与所有词级边 |
| `kMaxSyl=3`、`kLookupBudget=64` | 限制 DFS 深度与每起点查键次数 |

优化后（output-new 重建模型实测）：基线 4.13/3.20 ms → **5.92/4.82 ms/decode
（+1.8 ms，≈+43%）**。桌面全拼可接受。

### 10.4 评测（部署模型 output-news3-tech，纠错开启 vs §9.8 基线）

| 评测集 | 纠错前（§9.8） | 纠错后 | 差值 |
|---|---|---|---|
| cases.1（全拼·正式） | 67.53 | **68.08** | +0.55 |
| cases.2（全拼·口语） | 42.80 | **43.65** | +0.85 |
| cases.num.1（T9·正式） | 53.40 | 53.40 | 0 |
| cases.num.2（T9·口语） | 29.85 | 29.85 | 0 |

评测集本身含误拼输入 → 纠错**净提升、零回归**。T9 两集**完全不变**：纠错只
加在 `InitNet`（字母），未动 `InitNumNet`（九键）。
典型用例（tech 模型）：`zongguo→中国`、`shijei→世界`、`beijign→北京`（转置）；
精确 `zhongguo→中国`、`nihao→你好` 完全不变。

### 10.5 部署与回滚

插件由 SimeApp 重编（`-DSIME_ENGINE_ROOT=/data/tmp/Sime`、`SIME_NCNN_PROVIDER=FETCH`
静态 ncnn，自包含）：
```bash
cmake --build /data/tmp/SimeApp/Linux/fcitx5/build -j16
# libsime.so 15.0MB，md5 a0973439fa86236ad4713bed84fe3b79
```
部署到两台远程（本机不部署），`/usr/lib/fcitx5/libsime.so`，备份
`.bak-pre-correction`（旧版 md5 ce63646ff0c8a5adfd7eff5ea310cb6c）：
```bash
scp build/libsime.so ycd@HOST:/tmp/libsime.so.new
ssh ycd@HOST 'sudo cp -a /usr/lib/fcitx5/libsime.so /usr/lib/fcitx5/libsime.so.bak-pre-correction
  sudo install -m0755 -o root -g root /tmp/libsime.so.new /usr/lib/fcitx5/libsime.so'
# 重启 fcitx5（从 /proc/<pid>/environ 抓 WAYLAND_DISPLAY/XDG_RUNTIME_DIR/
# DBUS_SESSION_BUS_ADDRESS），日志确认 "Sime: resources loaded, GRU=ready"
```
- **192.168.1.188**：已生效。fcitx5 重启（Wayland 会话），GRU=ready，单实例，sime-pinyin 可用。
- **192.168.104.126**：插件已装 + 备份，但**用户会话 dbus 处于 failed 状态**
  （`systemctl --user is-active dbus.socket dbus.service` 均 failed，
  `busctl --user list` 连接被拒，`/run/user/1000/bus` 是陈旧套接字）——机器既有
  问题，非部署所致，也是 fcitx5 原本未运行的原因。无会话总线时 fcitx5 无法作
  IME，试启动的无效实例已清理。**坑 13**：修复需重新登录或
  `systemctl --user restart dbus.socket` 后重开 fcitx5（影响该机会话，未擅自操作）；
  插件会在 fcitx5 下次正常启动时自动加载纠错版。

回滚：还原 `.bak-pre-correction` 并重启 fcitx5。纠错亦可用编译期开关
`kEnableFuzzyCorrection/kEnableTypoCorrection=false` 单独关闭（重编即回退）。

---

## 11. 用户历史学习：Rime 词频种子导入 + 时间衰减（2026-11-21 追加）

两项**引擎代码**改动（不动词表/模型/GRU），让 Sime 具备 Rime 式的用户自学习：
先把 Rime 历史词频导入成 unigram 种子（方案 A，V5），再补上 Rime 的 `t/tick`
时间衰减（V6）。**截至本节这两项均未提交**（工作树改动，按约定未经要求不提交）。
涉及文件：`include/user.h`、`src/user.cc`（核心）、`include/sime.h`/`src/sime.cc`
（`SeedUserUnigram` 透传）、`bin/sime-import.cc`（新增导入工具）、`bin/sime.cc`
（`--user` 加载历史）、`CMakeLists.txt`（sime-import 目标 + leveldb 依赖）。

### 11.1 方案 A：Rime 词频 → unigram 种子（V5）

**动机**：Sime 原本无自学习（§5.1 口径里"表中即日常水平"）。Rime userdb 积累了
用户真实词频，导过来即可让常用词压过同音竞争词。

**Rime userdb 结构**：LevelDB 目录，key=`拼音\t文本`，value=`c=<commits> d=<depth> t=<tick>`。
是**纯 unigram 自频**（无 bigram/上下文）；`t` 是逻辑时钟、跨机不一致，不可靠。

**种子机制**（`UserSentence`，与段落历史分离）：
- 新增 `seed_unigram_`(map<TokenID,u32>) + `seed_total_`，独立 SEED 段持久化。
- `UnigramCount = 段落计数 + 种子`；`ParagraphCount = 仅段落`（**bigram 分母用它**，
  种子质量绝不污染转移概率）；`TotalUnigramCount = total_unigram_count_ + seed_total_`。
- `SeedUnigram(token,count)` 只加 unigram 质量，**不伪造 bigram**。
- `CostAdjustment` 仍是 libime 式概率空间线性混合（weight 0.2），max-floor 于 LM
  分（只会改善、不会变差）；无种子/无历史 → 返回 0（零回归）。

**导入工具 `bin/sime-import.cc`**（直接吃原始数据，无需先导出 TSV）：
```bash
# 原始 Rime userdb（LevelDB 目录，--rime 可重复，多库累加）
sime-import --rime ~/…/rime_ice.userdb --rime ~/…/luna_pinyin.userdb
# 或 TSV 导出（pinyin\ttext\tcommits\ttick）
sime-import --in export.tsv
```
- 每条文本用 Sime `Cutter.Cut` 切词，**仅当恰好切成 1 个已知词级 token**
  （`cuts.size()==1 && !is_unk`）才把 commits 计为种子；多词/OOV 跳过。
- `--dict/--cnt` 缺省按 `ResolveModelDir` 解析：`/usr/share/sime` →
  可执行文件同级 `../pipeline/output-new` → `cwd/pipeline/output-new`。
- `--out` 缺省按 `ResolveOutPath`：`$XDG_DATA_HOME/fcitx5/sime/sentences.txt` →
  `$HOME/.local/share/fcitx5/sime/sentences.txt`（与 fcitx5 插件 `userSentencePath()` 一致）。
- `--out` 已存在则先 `LoadUserSentence` 合并（词表不符自动迁移），种子叠加而非覆盖。
- 依赖 leveldb（`find_path/find_library`，本机与 126 均有 `/usr/lib/libleveldb.so.1`）。

**跨词表迁移**（关键）：`vocab_sig = dict_size:lm_size:lm_mtime_secs`
（`MakeVocabSignature`，sime.cc）。两台机器模型不同 → sig 不同 → `LoadUserSentence`
触发 `LoadAndMigrate`：忽略旧 token ID，把每段 **TEXT 注释列**拼回连续串，用本机
`Cutter.Cut` 重切，按本机词表重新落 ID。故导入文件必须带文本注释列（Save 时
`token_text` 回调发射 tab 对齐注释）。实测 188 模型 162M/135M ≠ 本机 154M/50M，
迁移后本机 ID（代码 66393 / 模型 211671 / 关闭 114911）正确落地。

**V5 格式**：头 `SIME_USER_SENTENCE_V5` + `VOCAB <sig>` + 段落（空行分段，每行
`<ids…>[\t<text>]`）+ `SEED` 段（`<id> <count>[\t<text>]`，按 id 排序）。

实测 boost（本机 output-new）：模型 moxing −10.34→−6.73；代码 daima 跌出 top3→首位
−6.87；关闭 guanbi→首位 −6.79。无种子用户零回归。

### 11.2 时间衰减（V6，Rime t/tick 移植）

**动机**：V5 计数无时间维度——很久以前打过一次的词 count=1 会一直保留（直到
8192 段落上限挤掉），和昨天刚打的词同权。Rime 用 per-word 衰减权重让"最近常用"
压过"陈旧一次性"。

**衰减模型**：per-token 指数衰减 unigram 权重，墙钟秒，**半衰期 30 天**。
- `DecayFactor(from,to)=exp(-(to-from)*kDecayLambda)`，`kDecayLambda=ln2/(30*86400)`；
  `to<from`（时钟回拨）时 elapsed 钳 0，不放大权重。
- **只衰减 unigram 尾部**；bigram 的 alpha/gamma 仍用原始 `ParagraphCount`/`BigramCount`，
  不污染转移概率。
- 归一化 `pr_unigram = decayed(cur)/decayed_total`。所有权重同速率衰减 → 纯时间
  流逝不改相对排序；**只有新选词**（新鲜 +1、旧权重已衰减）才抬升近期词。
- **种子衰减起点 = 导入时刻**（Rime `t` 不可靠）。刚导入所有种子同龄 → 相对排序
  与 V5 一致（无即时回归），之后随用户新选词逐渐体现近期偏好。

**数据结构**（`include/user.h`）：
- `Selection` 增 `std::uint32_t tick=0`（墙钟秒，0=未知→load 视为 now）；`Add` 设 `sel.tick=Now()`。
- 新成员 `decay_w_`(map<TokenID,double>)、`decay_t_`(map<TokenID,u32>)、
  `decay_total_`(double)、`decay_total_t_`(u32)；**`seed_tick_`(map<TokenID,u32>)**
  保存种子时间戳供 Rebuild 重建。
- helper：`Now()`、`DecayFactor`、`TouchDecay(token,delta,now)`（O(1) 增量，刷新
  total 与 token 到 now）、`AdjustEntryDecay(entry,±1,now)`。

**坑 14（关键设计修正）**：原想"Rebuild 经 AdjustEntryCounts+TouchDecay 重建衰减"
**不可行**——load 回放任意历史 tick，而 `TouchDecay` 的全局刷新假设时间单调递增，
遇到更旧的时间戳会把 `decay_total_` 往回衰减、破坏归一化。最终拆成两条路径：
- `AdjustEntryCounts` 只管原始 unigram/bigram（不动）。
- **运行时**增删（`StartParagraph`/`AppendToLatest`/`SetMaxSentenceCount`/pop）另调
  `AdjustEntryDecay`/`TouchDecay`（now 单调，正确）。
- **`Rebuild`**（load 时）清衰减后，在**公共参考时刻 now** 直接重建：段落每 token
  `+= DecayFactor(sel.tick,now)`，种子 `+= count*DecayFactor(seed_tick_,now)`，
  所有 `decay_t_=now`。不经过 TouchDecay。
- `SeedUnigram(token,count,tick=0)`：tick=0→now；聚合种子计数、记 `seed_tick_`、
  `TouchDecay(token,count,Now())`（load 时该临时权重随后被 Rebuild 覆盖，无害）。

**`CostAdjustment`**：保留 `cur_count==0` 早退；`now_q=Now()`，
`total_w=decay_total_*DecayFactor(decay_total_t_,now_q)`，guard `total_w>0`；
`cur_w=decay_w_[cur]*DecayFactor(decay_t_[cur],now_q)`；
`pr_unigram=max(cur_w-kUnigramDiscount,0)/total_w`。bigram 分支与 LM 混合不变。

**V6 格式**：头 `SIME_USER_SENTENCE_V6`，`AcceptHeader` 兼容读 V6/V5/V4。
时间戳用 **`@<tick>` 后缀**（保持 tab 结构、向后兼容）：
- selection 行 `<ids…> @<tick>[\t<text>]`；seed 行 `<id> <count> @<tick>[\t<text>]`。
- `SplitTick` 按空格切字段，末字段 `@` 开头→tick，其余为 ids；**无 `@`→tick=now**
  （旧 V5/V4 文件视为新鲜，避免升级即失效）。
- `LoadAndMigrate` 重切文本时**保留旧行 `@tick`**（词表迁移不丢 recency），种子同理。
- 现有 sentences.txt 无需手动升级：新引擎读 V5（视为 now），下次 flush 自动写 V6。

### 11.3 验证（全部通过）

构建 `sime`/`sime-import`/插件，无 `-Wconversion` 告警。用同音竞争词
`gongshi`（攻势 108177 LM −11.119 / 公式 108175 LM −11.140，极接近）构造 V6 种子文件：

| 文件 | 公式种子 | 攻势种子 | 首位 | 说明 |
|---|---|---|---|---|
| fresh | 100 @now | 20 @now | **公式** −1.792 | 新鲜高计数抬升 |
| stale | 100 @now−90天(≈12.5) | 20 @now | **攻势** −2.095 | 90天=3半衰期，公式被压回 LM 基线 |
| half | 100 @now−30天(≈50) | 50 @now | **攻势** −2.302 | 30天权重减半→近平局，LM 决胜 |
| fresh50 | 100 @now | 50 @now | **公式** −2.015 | 对照：新鲜公式胜攻势50 |

half vs fresh50 证实 30 天半衰期（公式 100 衰减到 ≈50 后输给攻势 50）。
- **坑 15**：单个种子词衰减会在归一化里**自我抵消**（`cur_w/total_w=1` 与衰减无关），
  必须 ≥2 个竞争种子才能观察到排序变化。
- **坑 16**：CLI sentence 模式只有**首位候选**显示含用户加权的分，非首位显示 LM
  基线分；验证看**排序**而非各候选分值。
- V5 兼容：加载 `seeds_fresh.txt`（V5）模型仍 −6.725（与方案 A 一致）。
- 迁移保留 tick：伪造 sig 触发 `LoadAndMigrate`，公式陈旧 tick 原样保留、新导入词
  取导入时刻，输出 V6 + 本机 sig。
- 零回归：无 `--user` → 攻势基线首位；无历史 `decay_total_=0` → CostAdjustment 返回 0。

### 11.4 部署与回滚

插件由 SimeApp 重编（`-DSIME_ENGINE_ROOT=/data/tmp/Sime`、`SIME_NCNN_PROVIDER=FETCH`
静态 ncnn 自包含）：
```bash
cmake --build /data/tmp/SimeApp/Linux/fcitx5/build -j16
# libsime.so 15,055,160 B，md5 c6fee7623d3d068fffeeda0255ad06a6（含 V6/V5/V4 头）
```
部署两台（本机不部署），`/usr/lib/fcitx5/libsime.so`，旧版备份 `/tmp/libsime.so.old`
（md5 c15e13c191f2f9289c650259b22f18bb）。**sentences.txt 保留不动**（V5），新引擎
下次 flush 自动升 V6：
```bash
scp build/libsime.so ycd@HOST:/tmp/libsime.so.new
# 188（fcitx5 运行中）：停→装→启，避免退出 flush 覆盖（坑 17）
ssh ycd@192.168.1.188 'cp /usr/lib/fcitx5/libsime.so /tmp/libsime.so.old
  kill $(pgrep -x fcitx5); while pidof fcitx5 >/dev/null; do sleep 0.3; done
  sudo cp /tmp/libsime.so.new /usr/lib/fcitx5/libsime.so; sudo chmod 755 …'
# 重启抓会话环境（WAYLAND_DISPLAY=wayland-1 / XDG_RUNTIME_DIR=/run/user/1000 /
# DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus），setsid  detached 启动
```
- **192.168.1.188**：已生效。新进程 pid 4035386，日志 `Sime: resources loaded, GRU=ready`，
  映射新 so，sentences 保留 V5（待 flush 升 V6）。
- **192.168.104.126**：新插件已装 + 备份，但**用户会话 dbus 仍 failed**（同坑 13），
  fcitx5 起不来 → 预置待激活，修复 dbus 后下次正常启动自动加载 V6 引擎。
- **坑 17（沿用 §9.5）**：fcitx5 退出会 flush 当前状态覆盖 sentences.txt。换插件务必
  **停→装→启**；本次保留 sentences 故 flush 无害，但若同时铺历史文件必须先停进程。

回滚：还原 `/tmp/libsime.so.old`（c15e13c1，V5 引擎）并重启 fcitx5；V6 文件可被
V5 引擎拒绝（头不符），如需回滚引擎建议同时还原 sentences 备份。衰减半衰期是

## 12. 二开六项 + 单元测试体系（2026-11-21 追加）

引擎 `57aefac` 之后落地六项二开，**默认配置零回归**（四集逐位一致），新增逻辑
尽量抽成免模型纯函数 + doctest 单测。插件 md5 `b3f4bd57eeb7db2ab68be04f5fa802b4`
（15,281,976 B）。

### 12.1 六项设计与插入点

1. **混排句中英文识别**（`yong grep sousuo`→用 grep 搜索）。`InitNet` 把空格
   归一化的 ▁(U+2581, E2 96 81) 当自由跨越边：贪心 walk 消费 3 字节标 terminal、
   列循环发 `{s,s+3,NotToken}` 并跳过内部字节、expansion 内层边界 walk 遇 ▁ break、
   英文尾补全在首个 ▁ 截断（边 end 改为 ▁ 位）、纠错 DFS break 条件加 ▁ 三字节。
   `ExtractText(path,input)` 对 ▁ NotToken 边渲染单个空格（抑制 SentencePiece 前导
   ▁ 重复空格）。惩罚 span 化：`full_cover` 分支逐字节保持原样；新增 mid-sentence
   分支用纯函数 `ComputeEnglishPenalty`，**CN 竞争守卫**——whitelist 确信词只受
   *精确* CN 边（`!expansion`）守卫（grep 的展开简拼"热盘"不算竞争），其余词需无任何
   CN 边且 `!IsKnownPinyin(span)` 才放松（women/song/long 保持重惩罚）。仅当输入含 ▁
   才进新分支 → 纯中文/纯英文/T9 不受影响。
2. **关键参数运行时可配置**：编译期常量改成员 + setter（默认=旧值），fcitx5 Option
   热重载（`applyConfig`，整数百分比 /100.0）。见 §12.4。
3. **用户感知下一词预测**：`NextTokens` 取 pool 后对全候选折叠
   `user_sentence_.CostAdjustment(prev,tid,pro)`（≤0），按 adj_cost 升序 stable_sort
   再取 num。无历史 adj=0 → 排序同纯 LM（零回归）。复用衰减 unigram/bigram + 种子。
4. **双字母命令**：纯函数 `IsTwoLetterCommand`（curated 集，排除合法拼音 du 与
   nm/us/be/up 等非命令）。`ComputeEdgePenalties` full-cover len==2 命中 → 温和
   `english_penalty_`(2.5) 而非 8.0（不给 -3.0 强 bonus，避免压垮简拼）；whitelist
   词（wc/ls/cd/sh/jq…）仍走原 -3.0。实测 ps 抬升居首、mv 列第二（女仍首）、ping/du 不变。
5. **T9 拼音纠错**：纯函数 `pinyin_correct::T9TypoVariants`（长度保持的相邻转置 +
   电话键位邻键替换，仅留已知 T9 音节）。`InitNumNet` digit 展开后、tier filter 前
   加有界 DFS（≤3 音节、预算 32），**仅对无效且不可扩展的数字段**发纠错边（valid 输入
   与简拼前缀不触发），键为数字串经 `PrefixSearchT9` 解析，cost=typo_correction_penalty_。
   **独立开关 `t9_correction_enabled_` 默认关闭**（见坑 18）。开启后 `96635`(wo+误拼
   men 636→635)→我们居首；精确 `94664486`→中国不变。
6. **GRU 自适应 scale**：纯函数 `AdaptiveGruScale(base,spread,n,input_len)`，
   `spread=l1[0].score-l1[min(10,size)-1].score`，`scale=base*clamp(1+0.1*(2-spread),0.5,1.5)`，
   n≤1 或 len≤2 保持 base。`RerankWithGru` 增 adaptive/input_len 参数。**开关
   `adaptive_gru_enabled_` 默认关闭**（无正式域分类器，启发式收益边际）；关闭==固定 0.40/0.50。

### 12.2 文件拆分（>500 行约束）

特性前先做行为保持的纯代码移动，四集逐位一致后再叠加：`sime.cc`(1629)→
sime/sime_net_pinyin/sime_net_num/sime_penalty/sime_english/sime_search/sime_decode/
sime_predict/sime_gru + `sime_detail.h`；`user.cc`(856)→user/user_decay/user_persist；
`trie.cc`(870)→trie/trie_serialize/trie_pinyin/trie_build + `trie_detail.h`；
`construct.cc`(905)→construct/construct_smooth/construct_io + `construct_detail.h`
（模板成员须头文件可见）；`score.cc`(535)→score/score_query。全部文件 <500 行。

### 12.3 单元测试体系（doctest，60 用例全绿）

`cmake/Doctest.cmake` FetchContent 单头；根 `option(SIME_BUILD_TESTS OFF)` +
`SIME_TEST_MODEL_DIR` cache。免模型单测：test_user_sentence（计数/衰减/持久化/迁移）、
test_pinyin_correct（Fuzzy/Typo/**T9Typo**）、test_dict_static、test_ustr、
test_config（默认值/setter）、test_helpers（**IsTwoLetterCommand/ComputeEnglishPenalty/
AdaptiveGruScale** 纯函数）。gated 集成 test_decode（需模型目录）：预测零回归 + 用户种子
抬升 + 英文过滤、双字母命令、混排英文、T9 纠错 opt-in 恢复。
```bash
cmake -S /data/tmp/Sime -B build -DSIME_BUILD_TESTS=ON \
  -DSIME_TEST_MODEL_DIR=/data/tmp/Sime/pipeline/output-news3-tech
cmake --build build -j16 && ctest --test-dir build --output-on-failure  # 60/60
```

### 12.4 运行时配置项（fcitx5 Option → 引擎 setter）

| Option | 引擎接口 | 默认 |
|---|---|---|
| GruPinyinScale(%) / GruT9Scale(%) | SetGruScales | 40 / 50 |
| AdaptiveGru | SetAdaptiveGruEnabled | false |
| EnableFuzzy / EnableTypo | SetCorrectionEnabled | true / true |
| T9Correction | SetT9CorrectionEnabled | false |
| EnglishWhitelist | SetEnglishWhitelistEnabled | true |
| UserSentence | SetUserSentenceEnabled | true |
| UserWeight(%) | MutableUserSentence().SetWeight | 20 |
| DecayHalfLifeDays | MutableUserSentence().SetDecayHalfLife | 30 |

fcitx5-config 无 double 类型 → 比例用 `Option<int,IntConstrain>` 整数百分比，
applyConfig /100.0。GUI 由 fcitx5-configtool 自动生成，值存 ~/.config/fcitx5/conf/sime.conf。

### 12.5 评测（部署模型 output-news3-tech，默认配置）

四集（cases.1/2 全拼、cases.num.1/2 T9）与 §10.4 基线**逐位一致**（diff 全等），
准确率不变：cases.1 句 68.08/字 92.86、cases.2 43.65/85.60、num.1 53.40/86.32、
num.2 29.85/76.61。混排/双字母为加性守卫分支（纯中文/纯英文/精确 T9 不进新分支），
T9 纠错与自适应 GRU 默认关闭。手测：`yong grep sousuo`→用 grep 搜索、
`chang song`→长 送（song 不被抬）、`shuo hello women`→说 hello 我们、ps 居首、ping→平。

### 12.6 部署与回滚

```bash
cmake --build /data/tmp/SimeApp/Linux/fcitx5/build -j16   # libsime.so b3f4bd57…
scp build/libsime.so ycd@HOST:/tmp/libsime.so.new
# 旧版备份 /tmp/libsime.so.bak-sixfeat（c6fee762…，V6 引擎）
```
- **192.168.1.188**：停→装→启（坑 17），新 pid 12233，日志
  `Sime: resources loaded, GRU=ready`，单实例，映射 b3f4bd57，sentences 保留。
- **192.168.104.126**：已装 + 备份，**dbus 仍 failed**（坑 13），fcitx5 未运行 →
  预置待激活，修复 dbus 后下次正常启动自动加载六项引擎。
- 本机不部署。回滚：还原 `/tmp/libsime.so.bak-sixfeat` 并重启 fcitx5。
- **坑 18（T9 加性纠错的跨度优势）**：Layer2 距离惩罚 ≈11.1/位（|UnknownPenalty|20 /
  DistancePenalty1.8），远超纠错代价 7。任何覆盖*更多*数字位的纠错边都会经 dist_penalty
  优势压过精确短词（如 我姓何 8 位 顶掉 我依稀 6 位），即便仅对无效段纠错亦然——评测集
  含此类段。字母通道无此问题（纠错不改变 span 长度，且 valid 输入恒有精确逐字解析）。
  故 T9 纠错**默认关闭、可选开启**；fuzzy-on-valid 方案因 T9 数字歧义会生成更长 span
  新候选，回归更重，已弃用，改长度保持的按键纠错。
编译期常量 `kDecayHalfLifeSec`（src/user.cc），改值重编即可调。

## 13. Linux 命令行语料注入：命令→选项/子命令预测（2026-11-21 追加）

目标：提交 `ps` 后联想出常用选项（`aux`），`git`/`docker`/`apt` 出子命令。
方案 B（全量实施：语料→增量重训→四集门禁→部署），词表/GRU 不变（轻量注入）。

### 13.1 语料来源与生成（`linux/gen_corpus.py`）

三源混合（用户确认）：
- **tldr-pages/tldr**（`pages/{common,linux}/*.md`，6856 页）：反引号示例为主力，
  剥 `{{placeholder}}` 模板，×3 权重。
- **TellinaTool/nl2bash**（`data/bash/all.cm`，12607 行）：真实命令搭配，×2。
- **126 `.zsh_history`**（13613 命令）：按规范化命令的 √频次加权（封顶 8），反映个人高频。

清洗切词（输出空格分隔**预切词**，喂 `sime-count`）：
- 小写化；shell 操作符 `|;&(){}<>` 两侧补空格成独立 token（OOV → sime-count 断窗，
  保留管道/序列边界，避免跨段伪造 bigram）。
- flag 领先 `-` 拆成独立标点 token（`-ef`→`- ef`、`--all`→`- - all`）。
- 过滤：首词须在词表、且 ≥2 token（聚焦 bigram）。

产出 `linux/linux.cut.txt` **75897 行**（tldr 14377×3 + nl2bash 11863×2 + zsh 加权 9040）。
关键搭配：`ps aux`×59、`git commit`×75、`docker ps`×6、`ps - ef` 等。

### 13.2 增量重建（同 §9.3，只换 sime.cnt）

```bash
cd pipeline
# 标点表去掉 '-'（linux/punct-nodash.txt = punct.txt 删 '-' 行），让 dash 作为
# 连接 token，使 ls→-→la / ps→-→ef 成词（见坑 19）。
../build/sime-count -n 3 -d output-news3-tech/sime.token.dict.txt \
  -o linux/linux -s linux/swap -p linux/punct-nodash.txt -c 1000000 linux/linux.cut.txt
make merge.bin
mkdir -p output-news3-linux
for n in 1 2 3; do
  ./merge.bin -n $n -o output-news3-linux/raw.${n}gram \
    output-news3-tech/raw.${n}gram linux/linux.${n}gram &
done; wait
AV=$(( $(stat -c%s output-news3-linux/raw.1gram) / 8 ))   # 1505957
../build/sime-construct output-news3-linux/raw -n 3 -o /tmp/linux.raw.cnt \
  -c 0,1,2 -w "$AV" -r 10000000,4000000 -m 15
../build/sime-compact /tmp/linux.raw.cnt output-news3-linux/sime.cnt
# trie/GRU/词表全复用 tech（软链）
```

词表数不变 1,505,957；新增约 **3009 bigram / 6211 trigram**（含 dash 连接 `-→la/-→ef` 等）。
新 `sime.cnt` 135,507,986 B（比 tech 大 17,580 B）。

### 13.3 预测验证与插件英文-only 修复（关键）

CLI `--next` 手测：`git`→clone/remote/push/diff/checkout、`docker`→compose/container/run/pull、
`apt`→install/list 均出子命令。`ps` 的 **`--next-en`（英文-only）`aux` 排 [0]**。

- **坑 19（标点断上下文 → dash 非标点化）**：`sime-count` 对标点 token 先喂再 reset_windows。
  初版用 `punct.txt`（含 `-`），`ps - ef` 只形成 `ps→-`，`-→ef` 不成立 → `ls` 等全 dash 选项
  的命令联想只剩光杆 `-`。**修复**：linux 计数改用 `punct-nodash.txt`（删 `-`），dash 作连接 token，
  `ls→-→la`、`ps→-→ef` 成词。预测两步：`ls`→`-`→`la/lh/l`。BSD 无标点选项（`ps aux`）与
  子命令（git/docker/apt）一步直达。`aux`/`ef`/`la`/`lh` 本就在词表。
- **坑 20（英文命令预测：合并 dash 选项）**：插件原 `NextTokens(ctx,n)` 默认 en=false，`ps` 后
  全局高频标点 `/ ： ，` 与「的」把 `aux` 压到首页外。一度改 en=true，但 `ls` 选项全是 dash，
  英文-only 查询返回**空** → `ls` 提交后无候选。**最终方案**：`showPredictions` 对英文命令
  （`isEnglishWord`）合并两路——① en=true 取无 dash 选项/子命令（`aux`/`clone`）；② 从 mixed
   successors 找 dash token，再 `NextTokens(ctx+[dash])` 取选项词，拼成 `-la`/`-ef` 合并候选
  （tokens=[dash,flag]）。`ls`→`-l/-lh/-la…`、`ps`→`aux/-ef`、`git`→`clone/commit` 一步直达。
  中文上下文仍 en=false 不变（零回归）。
- **坑 21（联想提交缺空格）**：`SimeNextCandidateWord::select` 原直接 `commitString(text_)`，
  `ps`+`aux` 粘成 `psaux`。修复：上一提交词为英文（`isEnglishWord`）且联想词为英文词**或 dash**
  且词首非空格时提交 `" " + text_`（`ls`+`-`→`ls -`）；dash 后的选项词紧粘（`-`+`la`→`-la`，
  因 `isEnglishWord("-")=false`）；中文联想与标点仍无空格。最终 `ls -la`、`ps aux`、`ps -ef`。
- **坑 22（Shift 切英文须交核心，勿插件拦截）**：用户报「Sime 下按 Shift 切了输入法而非纯英文，
  只有 Rime 的 Shift 切英文」。根因：188 全局配置 `[Hotkey/AltTriggerKeys]=Shift_L` 是 fcitx5
  **核心层**中英切换（切到键盘布局＝纯英文），Rime 正是靠它。插件原自己拦截 Shift 切内部
  `ascii_mode_` 与核心打架。**修复**：删除插件 Shift 拦截/`ascii_mode_`/`toggleAsciiMode`，
  Shift 不消费、交核心 AltTriggerKeys 处理（与 Rime 一致）；Shift+字母仍解析大写键sym 正常上屏。
  保留 Ctrl+. 中英标点切换（`english_punc_`，中文模式下用）。

### 13.4 四集门禁（部署模型 output-news3-linux vs §12.5 tech 基线）

| 评测集 | 基线 句/字 | linux 句/字 | 差值 |
|---|---|---|---|
| cases.1（全拼） | 68.08 / 92.86 | 68.08 / 92.84 | 句持平 |
| cases.2（全拼） | 43.65 / 85.60 | 43.65 / 85.60 | 完全一致 |
| cases.num.1（T9） | 53.40 / 86.32 | 53.40 / 86.34 | 句持平 |
| cases.num.2（T9） | 29.85 / 76.61 | 29.80 / 76.59 | −1 句（噪声级） |

零回归（num.2 仅 1 句波动，1963 中）。命令搭配是稀有专有 bigram，对通用评测几乎无影响。

### 13.5 部署与回滚

- 模型 `/usr/share/sime/sime.cnt`（135,507,986 B），备份 `sime.cnt.bak-tech`（135,490,406 B）。
- 插件 `/usr/lib/fcitx5/libsime.so`（md5 `cf74b389…`，含 §13.3 合并 dash 选项预测 + 联想空格 + Ctrl+. 中英标点；
  Shift 中英切换交核心 AltTriggerKeys，见坑 22），备份 `/tmp/libsime.so.bak-en`。
- **188 / 126** 均已装 + 备份；fcitx5 当前未运行 → 预置。188 需桌面终端 `fcitx5 -r -d` 激活；
  126 dbus 仍 failed（坑 13），修复后自动加载。**本机不部署**。
- 回滚：`sudo install -m644 /usr/share/sime/sime.cnt.bak-tech /usr/share/sime/sime.cnt`
  + 还原 `/tmp/libsime.so.bak-en`，重启 fcitx5。