#include "sime.h"

#include "cut.h"
#include "gru_reranker.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sime {

namespace {

// Vocabulary signature used to detect stale user-sentence files when the
// LM is regenerated (new TokenIDs). LM file size + mtime is a cheap proxy
// for "was the LM regenerated"; when it changes we drop the old file.
std::string MakeVocabSignature(const std::filesystem::path& dict_path,
                               const std::filesystem::path& lm_path) {
    std::error_code ec;
    auto dict_size = std::filesystem::file_size(dict_path, ec);
    if (ec) return {};
    auto lm_size = std::filesystem::file_size(lm_path, ec);
    if (ec) return {};
    auto lm_mtime = std::filesystem::last_write_time(lm_path, ec);
    if (ec) return {};
    auto lm_mtime_secs =
        std::chrono::duration_cast<std::chrono::seconds>(
            lm_mtime.time_since_epoch())
            .count();
    std::ostringstream ss;
    ss << dict_size << ':' << lm_size << ':' << lm_mtime_secs;
    return ss.str();
}

} // namespace

void Sime::MaybeTrimCaches() const {
    if (++decode_count_ % kSepCacheTrimInterval == 0) {
        dict_.ClearSepCaches();
    }
}

void Sime::ResetCaches() const {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    dict_.ResetSepCaches();
}

Sime::Sime() = default;

Sime::Sime(const std::filesystem::path& dict_path,
                         const std::filesystem::path& model_path) {
    if (!dict_.Load(dict_path)) {
        return;
    }
    if (!scorer_.Load(model_path)) {
        dict_.Clear();
        return;
    }
    vocab_sig_ = MakeVocabSignature(dict_path, model_path);
    gru_ = std::make_unique<GruReranker>();
    gru_->Load(model_path.parent_path());
    ready_ = true;
}

Sime::~Sime() = default;

bool Sime::GruReady() const {
    return gru_ && gru_->Ready();
}

void Sime::SetUserSentenceEnabled(bool enabled) {
    user_sentence_enabled_ = enabled;
}

bool Sime::LoadUserSentence(const std::filesystem::path& path) {
    if (user_sentence_.Load(path, vocab_sig_)) {
        return true;  // fast path: vocab unchanged
    }

    // Vocab mismatch (LM regenerated, token IDs shifted). Try to
    // migrate: re-tokenize each paragraph's text under the new
    // vocabulary via Cutter, then persist immediately so the file is
    // consistent and next load takes the fast path.
    if (!ready_) return false;
    Cutter cutter(dict_, scorer_);
    auto tokenize = [&cutter](std::string_view text) {
        std::vector<std::pair<TokenID, std::string>> out;
        auto cuts = cutter.Cut(text);
        out.reserve(cuts.size());
        for (auto& ct : cuts) {
            TokenID id = ct.is_unk ? NotToken : ct.id;
            out.emplace_back(id, std::move(ct.text));
        }
        return out;
    };
    if (!user_sentence_.LoadAndMigrate(path, tokenize)) {
        return false;
    }
    // Re-write the file with the new vocab_sig and re-tokenized IDs so
    // subsequent loads skip migration.
    SaveUserSentence(path);
    return true;
}

bool Sime::SaveUserSentence(const std::filesystem::path& path) const {
    return user_sentence_.Save(path, vocab_sig_,
        [this](TokenID id) { return TokenText(id); });
}

void Sime::LearnUserSentence(const std::vector<TokenID>& context,
                             const std::vector<TokenID>& sentence) {
    if (!user_sentence_enabled_) {
        return;
    }
    user_sentence_.Add(context, sentence);
}

void Sime::SeedUserUnigram(TokenID id, std::uint32_t count) {
    if (!ready_ || id == NotToken) {
        return;
    }
    user_sentence_.SeedUnigram(id, count);
}

std::string Sime::TokenText(TokenID id) const {
    if (id == NotToken || !ready_) {
        return {};
    }
    const char32_t* p = dict_.TokenAt(id);
    if (!p) {
        return {};
    }
    std::u32string u32;
    while (*p) {
        u32.push_back(*p);
        ++p;
    }
    return TextFromU32(u32);
}

} // namespace sime