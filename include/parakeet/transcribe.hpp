#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/io/safetensors.hpp>

#include "parakeet/audio.hpp"
#include "parakeet/config.hpp"
#include "parakeet/ctc.hpp"
#include "parakeet/tdt.hpp"
#include "parakeet/tdt_ctc.hpp"
#include "parakeet/timestamp.hpp"
#include "parakeet/vocab.hpp"
#include "parakeet/wav.hpp"

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

/// Transcribe a WAV file to text using a ParakeetTDTCTC model.
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

    /// Transcribe a WAV file.
    TranscribeResult transcribe(const std::string &wav_path,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        auto wav = read_wav(wav_path);
        return transcribe(wav.samples, decoder, timestamps);
    }

    /// Transcribe from raw float32 samples (16kHz mono).
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        auto features = preprocess_audio(samples);
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        TranscribeResult result;

        if (timestamps) {
            if (decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto all_ts =
                    ctc_greedy_decode_with_timestamps(log_probs.cpu());
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else {
                auto all_ts = tdt_greedy_decode_with_timestamps(
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
            if (decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                all_tokens = ctc_greedy_decode(log_probs.cpu());
            } else {
                all_tokens = tdt_greedy_decode(model_, encoder_out,
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

    TranscribeResult transcribe(const std::string &wav_path,
                                bool timestamps = false) {
        auto wav = read_wav(wav_path);
        return transcribe(wav.samples, timestamps);
    }

    TranscribeResult transcribe(const axiom::Tensor &samples,
                                bool timestamps = false) {
        auto features = preprocess_audio(samples);
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        TranscribeResult result;

        if (timestamps) {
            auto all_ts = tdt_greedy_decode_with_timestamps(
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
                tdt_greedy_decode(model_, encoder_out, config_.durations);
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
