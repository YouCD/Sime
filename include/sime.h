#pragma once

#include "common.h"
#include "score.h"
#include "state.h"
#include "dict.h"
#include "user.h"
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace sime {

class GruReranker;

struct DecodeResult {
    std::string text;        // UTF-8 display text (▁ prefix stripped)
    std::string units;       // segmented pinyin (e.g. "ni'hao")
    std::vector<TokenID> tokens;  // token IDs for LM context
    float_t score = 0.0;    // larger is better (negative log probability negated)
    std::size_t cnt = 0;     // bytes of input consumed
};

class Sime {
public:
    Sime();
    Sime(const std::filesystem::path& dict_path,
         const std::filesystem::path& model_path);
    ~Sime();

    bool Ready() const { return ready_; }
    bool GruReady() const;
    int ContextSize() const { return scorer_.Num() - 1; }

    // Cache trim hooks for the long-running daemon use case (Android
    // service / Linux fcitx5). Called periodically inside decode entry
    // points to bound the trie sep_cache content size; ResetCaches()
    // is for hard release on memory pressure (e.g. Android
    // onTrimMemory).
    void ResetCaches() const;

    void SetUserSentenceEnabled(bool enabled);
    bool UserSentenceEnabled() const { return user_sentence_enabled_; }
    UserSentence& MutableUserSentence() { return user_sentence_; }
    const UserSentence& GetUserSentence() const { return user_sentence_; }
    // Loads from disk. Returns false on missing/corrupt/vocab-mismatch
    // files; the in-memory state is left empty in those cases. The
    // caller may delete the stale file when this returns false.
    bool LoadUserSentence(const std::filesystem::path& path);
    bool SaveUserSentence(const std::filesystem::path& path) const;
    // Opaque tag identifying the LM vocabulary the engine is currently
    // bound to. Stored alongside user sentences so they get dropped
    // when the LM is regenerated and TokenIDs no longer mean the same
    // thing. Empty until/unless an LM was successfully loaded.
    const std::string& VocabSignature() const { return vocab_sig_; }
    void LearnUserSentence(const std::vector<TokenID>& context,
                           const std::vector<TokenID>& sentence);
    // Inject an imported unigram frequency seed (e.g. from a Rime
    // userdb) for a known token id. Adds unigram mass only — no bigram
    // is fabricated. No-op when the engine is not ready or id is
    // NotToken. Persisted via SaveUserSentence's SEED section.
    void SeedUserUnigram(TokenID id, std::uint32_t count);
    // UTF-8 text for a single token id, sourced from the dict's
    // mmap'd token table. Empty for NotToken or out-of-range ids.
    std::string TokenText(TokenID id) const;

    // Runtime tuning. Defaults equal the historical compile-time values,
    // so an unconfigured engine behaves exactly as before. Setters are
    // cheap value assignments meant to be called from the fcitx5 config
    // hot-reload path; they take effect on the next decode.
    void SetEnglishPenalties(float_t normal, float_t short_input,
                             float_t whitelist) {
        english_penalty_ = normal;
        english_penalty_short_ = short_input;
        english_whitelist_penalty_ = whitelist;
    }
    void SetEnglishWhitelistEnabled(bool enabled) {
        english_whitelist_enabled_ = enabled;
    }
    bool EnglishWhitelistEnabled() const { return english_whitelist_enabled_; }
    void SetCorrectionEnabled(bool fuzzy, bool typo) {
        enable_fuzzy_correction_ = fuzzy;
        enable_typo_correction_ = typo;
    }
    void SetCorrectionPenalties(float_t fuzzy, float_t typo) {
        fuzzy_pinyin_penalty_ = fuzzy;
        typo_correction_penalty_ = typo;
    }
    bool FuzzyCorrectionEnabled() const { return enable_fuzzy_correction_; }
    bool TypoCorrectionEnabled() const { return enable_typo_correction_; }
    // T9 key-typo correction is opt-in (default off): additive corrected edges
    // can span more digits than the best exact Layer-2 word and win via the
    // distance penalty, perturbing correct-input rankings. Off keeps the T9
    // eval bit-identical; on recovers genuine mistypes.
    void SetT9CorrectionEnabled(bool enabled) { t9_correction_enabled_ = enabled; }
    bool T9CorrectionEnabled() const { return t9_correction_enabled_; }
    void SetGruScales(float_t pinyin, float_t t9) {
        gru_pinyin_scale_ = pinyin;
        gru_t9_scale_ = t9;
    }
    // Adaptive GRU domain bias is opt-in (default off): the spread heuristic
    // is marginal without a formal-domain classifier. Off == fixed scales.
    void SetAdaptiveGruEnabled(bool enabled) { adaptive_gru_enabled_ = enabled; }
    bool AdaptiveGruEnabled() const { return adaptive_gru_enabled_; }
    float_t GruPinyinScale() const { return gru_pinyin_scale_; }
    float_t GruT9Scale() const { return gru_t9_scale_; }

    // Decode
    std::vector<DecodeResult> DecodeStr(std::string_view input,
                                        std::size_t num = 5) const;
    std::vector<DecodeResult> DecodeSentence(std::string_view input,
                                             std::size_t extra = 0) const;
    std::vector<DecodeResult> DecodeSentence(
        std::string_view input,
        const std::vector<TokenID>& context,
        std::size_t extra = 0) const;
    // Prediction: given confirmed token IDs as context, suggest next words.
    // When `en` is true, only English tokens are returned (for the English
    // IME's prediction slot); Chinese tokens are filtered out.
    std::vector<DecodeResult> NextTokens(
        const std::vector<TokenID>& context,
        std::size_t num = 10,
        bool en = false) const;

    // Prefix completion: return tokens starting with `prefix`, sorted by
    // unigram score. Default searches both English and pinyin DATs (mixed
    // mode). When `en` is true, only the English DAT is searched.
    std::vector<DecodeResult> GetTokens(
        std::string_view prefix,
        std::size_t num = 10,
        bool en = false) const;

    // Num-key decode (T9/nine-key).
    // `start` is the confirmed prefix (letters, possibly with `'`
    // separators). Supports both pinyin and English prefixes.
    std::vector<DecodeResult> DecodeNumStr(
        std::string_view nums,
        std::string_view start = {},
        std::size_t num = 18) const;
    // Layer 1: full sentence N-best covering start + nums. Returns
    // 1 + `extra` sentences (top sentence is always included; `extra`
    // additional alternatives are appended).
    // Layer 2: word/char alternatives anchored at the first digit
    // column. Both layers are scored against the
    // LM context produced by the prefix `start`.
    std::vector<DecodeResult> DecodeNumSentence(
        std::string_view nums,
        std::string_view start = {},
        std::size_t extra = 0) const;
    std::vector<DecodeResult> DecodeNumSentence(
        std::string_view nums,
        std::string_view start,
        const std::vector<TokenID>& context,
        std::size_t extra = 0) const;
private:
    // Lattice types
    struct Link {
        std::size_t start = 0;
        std::size_t end = 0;
        TokenID id = 0;
        const char* pieces = nullptr;  // piece path, e.g. "ni'hao"
        float_t penalty = 0;           // syllable mismatch penalty
        bool expansion = false;        // tail-expansion edge (lower priority in PruneNode)
        bool english = false;          // edge from English DAT (lower priority than full pinyin)
        float_t correction = 0;        // fuzzy-pinyin / typo correction cost (0 = exact path)
    };

    struct Node {
        std::vector<Link> es;
        NetStates states;
    };

    // Search parameters
    static constexpr std::size_t NodeSize = 60;
    static constexpr std::size_t BeamSize = 20;
    static constexpr float_t DistancePenalty = 1.8;
    // Flat per-edge penalty for expansion edges (abbreviations / tail
    // completions). A single value — not multiplied by syllable mismatch
    // count — so longer abbreviations aren't punished more than shorter
    // ones (mirrors English exact). Short input (≤4) gets a smaller
    // penalty so n-initial abbreviations like 87 → 他说 can beat short
    // English exact (e.g. up); long input gets a larger penalty so
    // abbreviation expansions don't crowd out the natural full-pinyin
    // path in long sentences.
    static constexpr float_t ExpansionPenalty = 5.5;
    static constexpr float_t ExpansionPenaltyShort = 4.0;
    // Flat penalty for English edges so Chinese full-pinyin matches rank
    // higher in beam search and Layer 2, while common English words
    // (iphone, hello) still surface when there's no strong Chinese rival.
    // Runtime-configurable; defaults = historical compile-time values.
    float_t english_penalty_ = 2.5;
    // Short-input English penalty: very short inputs (≤3 letters/digits)
    // where an English exact word coincidentally spell-matches the input
    // (e.g. "nm" / "us" / "be" / "men") need a stronger nudge so CN 简拼
    // candidates can win. Length 4+ full-cover English uses english_penalty_.
    float_t english_penalty_short_ = 8.0;
    // Bonus (negative penalty) for full-cover English edges whose text is
    // a curated common short command / tech word (≤4 letters, e.g. awk /
    // sed / ssh / tar / bash / code). Confident English intent for a
    // developer, so they must beat CN 简拼 / 同音 rivals. Scoped to ≤4
    // letters so longer English keeps english_penalty_.
    float_t english_whitelist_penalty_ = -3.0;
    bool english_whitelist_enabled_ = true;
    // Pinyin correction (additive edges; exact path untouched). Fuzzy
    // pronunciation (zh/z, n/l, ang/an ...) costs less than a keyboard typo
    // (edit distance 1) because it is a systematic, higher-confidence
    // reinterpretation. Both sit above english_penalty_ so a correct exact
    // pinyin match always outranks a corrected rival.
    float_t fuzzy_pinyin_penalty_ = 5.0;
    float_t typo_correction_penalty_ = 7.0;
    bool enable_fuzzy_correction_ = true;
    bool enable_typo_correction_ = true;
    bool t9_correction_enabled_ = false;
    // GRU rerank scales (full-pinyin / T9). Runtime-configurable.
    float_t gru_pinyin_scale_ = 0.40;
    float_t gru_t9_scale_ = 0.50;
    bool adaptive_gru_enabled_ = false;

    // Lattice building
    void InitNet(std::string_view input,
                    std::vector<Node>& net,
                    bool expansion = true) const;
    void ComputeEdgePenalties(std::vector<Node>& net,
                              std::string_view input) const;
    void PruneNode(std::vector<Link>& edges,
                   std::unordered_map<TokenID, float_t>* score_cache = nullptr) const;

    // Beam search
    State InitialState(const std::vector<TokenID>& context = {}) const;
    void Process(std::vector<Node>& net) const;
    static std::vector<Link> Backtrace(const State& tail_state,
                                       std::size_t end);

    std::u32string ToText(const Link& n) const;
    std::string ExtractText(const std::vector<Link>& path,
                            std::string_view input) const;
    static std::string ExtractUnits(const std::vector<Link>& path,
                                    std::string_view input);
    static std::string AbbreviatePieces(const char* full_pieces,
                                        std::string_view input);
    std::vector<TokenID> ExtractTokens(const std::vector<Link>& path) const;
    static std::string TextFromU32(std::u32string& u32);

    // Num-key lattice
    void InitNumNet(std::string_view start,
                     std::string_view nums,
                     std::vector<Node>& net,
                     bool expansion = true) const;

    // Periodic soft trim of trie sep_cache, called from decode entries.
    // Drops cached sep lists every kSepCacheTrimInterval decodes so the
    // daemon's memory footprint stays bounded. Soft (no allocator
    // churn): ~free in latency.
    void MaybeTrimCaches() const;
    static constexpr std::size_t kSepCacheTrimInterval = 5000;
    mutable std::size_t decode_count_ = 0;
    // Dict::scratch_ and the trie separator caches are intentionally reused
    // between calls. Serialize public decode/cache entry points so const
    // callers may safely share one Sime instance across threads.
    mutable std::mutex decode_mutex_;

    // Resources
    Dict dict_;
    Scorer scorer_;
    UserSentence user_sentence_;
    std::string vocab_sig_;
    bool user_sentence_enabled_ = false;
    bool ready_ = false;
    std::unique_ptr<GruReranker> gru_;
};

} // namespace sime
