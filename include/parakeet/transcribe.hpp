#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/io/safetensors.hpp>

#include "parakeet/audio.hpp"
#include "parakeet/audio_io.hpp"
#include "parakeet/config.hpp"
#include "parakeet/ctc.hpp"
#include "parakeet/phrase_boost.hpp"
#include "parakeet/tdt.hpp"
#include "parakeet/tdt_ctc.hpp"
#include "parakeet/timestamp.hpp"
#include "parakeet/vocab.hpp"

namespace parakeet {

// ─── Transcription Result ───────────────────────────────────────────────────

struct TranscribeResult {
    std::string text;           // Decoded text
    std::vector<int> token_ids; // Raw token IDs before detokenization

    // Optional timestamps (populated when timestamps=true)
    std::vector<TimestampedToken> timestamped_tokens;
    std::vector<WordTimestamp> word_timestamps;
};

// ─── High-Level Transcription API ───────────────────────────────────────────

enum class Decoder { CTC, TDT };

// ─── Transcription Options ──────────────────────────────────────────────────

struct TranscribeOptions {
    Decoder decoder = Decoder::TDT;
    bool timestamps = false;
    std::vector<std::string> boost_phrases;
    float boost_score = 5.0f;
};

/// Transcribe an audio file to text using a ParakeetTDTCTC model.
/// Supports WAV, FLAC, MP3, OGG. Automatically resamples to 16kHz.
///
/// This is the simplest entry point — loads audio, preprocesses, encodes,
/// decodes, and detokenizes in one call.
///
///   parakeet::Transcriber t("model.safetensors", "vocab.txt");
///   auto result = t.transcribe("audio.wav");
///   std::cout << result.text << std::endl;
///
class Transcriber {
  public:
    /// Construct from a safetensors weights file and SentencePiece vocab file.
    /// Uses the 110M TDT-CTC config by default.
    Transcriber(const std::string &weights_path, const std::string &vocab_path,
                const TDTCTCConfig &config = make_110m_config())
        : config_(config), model_(config) {
        auto weights = axiom::io::safetensors::load(weights_path);
        model_.load_state_dict(weights, "", false);
        tokenizer_.load(vocab_path);
    }

    /// Move model to GPU (Metal). Call once after construction.
    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    /// Transcribe an audio file (WAV, FLAC, MP3, OGG).
    TranscribeResult transcribe(const std::string &audio_path,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        auto audio = read_audio(audio_path);
        return transcribe(audio.samples, decoder, timestamps);
    }

    /// Transcribe from raw float32 samples (16kHz mono).
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        TranscribeOptions opts;
        opts.decoder = decoder;
        opts.timestamps = timestamps;
        return transcribe(samples, opts);
    }

    /// Transcribe an audio file with options (boost phrases, etc.).
    TranscribeResult transcribe(const std::string &audio_path,
                                const TranscribeOptions &opts) {
        auto audio = read_audio(audio_path);
        return transcribe(audio.samples, opts);
    }

    /// Transcribe from raw float32 samples with options.
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                const TranscribeOptions &opts) {
        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto features = preprocess_audio(samples, audio_cfg);
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        // Build trie if boost phrases provided
        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        TranscribeResult result;

        if (opts.timestamps) {
            if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                auto all_ts = use_boost
                                  ? ctc_greedy_decode_with_timestamps_boosted(
                                        cpu_lp, trie, opts.boost_score)
                                  : ctc_greedy_decode_with_timestamps(cpu_lp);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else {
                auto all_ts = use_boost
                                  ? tdt_greedy_decode_with_timestamps_boosted(
                                        model_, encoder_out, config_.durations,
                                        trie, opts.boost_score)
                                  : tdt_greedy_decode_with_timestamps(
                                        model_, encoder_out, config_.durations);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            }

            if (tokenizer_.loaded()) {
                result.text = tokenizer_.decode(result.token_ids);
                result.word_timestamps = group_timestamps(
                    result.timestamped_tokens, tokenizer_.pieces());
            }
        } else {
            std::vector<std::vector<int>> all_tokens;
            if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                all_tokens = use_boost ? ctc_greedy_decode_boosted(
                                             cpu_lp, trie, opts.boost_score)
                                       : ctc_greedy_decode(cpu_lp);
            } else {
                all_tokens = use_boost
                                 ? tdt_greedy_decode_boosted(
                                       model_, encoder_out, config_.durations,
                                       trie, opts.boost_score)
                                 : tdt_greedy_decode(model_, encoder_out,
                                                     config_.durations);
            }

            if (!all_tokens.empty()) {
                result.token_ids = all_tokens[0];
                if (tokenizer_.loaded()) {
                    result.text = tokenizer_.decode(result.token_ids);
                }
            }
        }

        return result;
    }

    /// Access the underlying model for advanced use.
    ParakeetTDTCTC &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    TDTCTCConfig config_;
    ParakeetTDTCTC model_;
    Tokenizer tokenizer_;
    bool use_gpu_ = false;
};

// ─── TDT-Only Transcriber (for 600M multilingual etc.) ─────────────────────

/// Transcribe using a TDT-only model (no CTC head).
///
///   parakeet::TDTTranscriber t("model.safetensors", "vocab.txt",
///                               parakeet::make_tdt_600m_config());
///   auto result = t.transcribe("audio.wav");
///
class TDTTranscriber {
  public:
    TDTTranscriber(const std::string &weights_path,
                   const std::string &vocab_path,
                   const TDTConfig &config = make_tdt_600m_config())
        : config_(config), model_(config) {
        auto weights = axiom::io::safetensors::load(weights_path);
        model_.load_state_dict(weights, "", false);
        tokenizer_.load(vocab_path);
    }

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    TranscribeResult transcribe(const std::string &audio_path,
                                bool timestamps = false) {
        auto audio = read_audio(audio_path);
        return transcribe(audio.samples, timestamps);
    }

    TranscribeResult transcribe(const axiom::Tensor &samples,
                                bool timestamps = false) {
        TranscribeOptions opts;
        opts.decoder = Decoder::TDT;
        opts.timestamps = timestamps;
        return transcribe(samples, opts);
    }

    TranscribeResult transcribe(const std::string &audio_path,
                                const TranscribeOptions &opts) {
        auto audio = read_audio(audio_path);
        return transcribe(audio.samples, opts);
    }

    TranscribeResult transcribe(const axiom::Tensor &samples,
                                const TranscribeOptions &opts) {
        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto features = preprocess_audio(samples, audio_cfg);
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        TranscribeResult result;

        if (opts.timestamps) {
            auto all_ts = use_boost
                              ? tdt_greedy_decode_with_timestamps_boosted(
                                    model_, encoder_out, config_.durations,
                                    trie, opts.boost_score)
                              : tdt_greedy_decode_with_timestamps(
                                    model_, encoder_out, config_.durations);
            if (!all_ts.empty()) {
                result.timestamped_tokens = all_ts[0];
                for (const auto &t : result.timestamped_tokens) {
                    result.token_ids.push_back(t.token_id);
                }
            }
            if (tokenizer_.loaded()) {
                result.text = tokenizer_.decode(result.token_ids);
                result.word_timestamps = group_timestamps(
                    result.timestamped_tokens, tokenizer_.pieces());
            }
        } else {
            auto all_tokens =
                use_boost
                    ? tdt_greedy_decode_boosted(model_, encoder_out,
                                                config_.durations, trie,
                                                opts.boost_score)
                    : tdt_greedy_decode(model_, encoder_out, config_.durations);
            if (!all_tokens.empty()) {
                result.token_ids = all_tokens[0];
                if (tokenizer_.loaded()) {
                    result.text = tokenizer_.decode(result.token_ids);
                }
            }
        }

        return result;
    }

    ParakeetTDT &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    TDTConfig config_;
    ParakeetTDT model_;
    Tokenizer tokenizer_;
    bool use_gpu_ = false;
};

} // namespace parakeet
