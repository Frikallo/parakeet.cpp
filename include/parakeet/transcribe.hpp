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
#include "parakeet/vocab.hpp"
#include "parakeet/wav.hpp"

namespace parakeet {

// ─── Transcription Result ───────────────────────────────────────────────────

struct TranscribeResult {
    std::string text;           // Decoded text
    std::vector<int> token_ids; // Raw token IDs before detokenization
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
                                Decoder decoder = Decoder::TDT) {
        auto wav = read_wav(wav_path);
        return transcribe(wav.samples, decoder);
    }

    /// Transcribe from raw float32 samples (16kHz mono).
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                Decoder decoder = Decoder::TDT) {
        auto features = preprocess_audio(samples);
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        std::vector<std::vector<int>> all_tokens;

        if (decoder == Decoder::CTC) {
            auto log_probs = model_.ctc_decoder()(encoder_out);
            all_tokens = ctc_greedy_decode(log_probs.cpu());
        } else {
            all_tokens =
                tdt_greedy_decode(model_, encoder_out, config_.durations);
        }

        TranscribeResult result;
        if (!all_tokens.empty()) {
            result.token_ids = all_tokens[0];
            result.text = tokenizer_.decode(result.token_ids);
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

} // namespace parakeet
