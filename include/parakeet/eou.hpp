#pragma once

#include <functional>
#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/audio.hpp"
#include "parakeet/config.hpp"
#include "parakeet/lstm.hpp"
#include "parakeet/rnnt.hpp"
#include "parakeet/streaming_encoder.hpp"
#include "parakeet/tdt.hpp"
#include "parakeet/vocab.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── EOU Config ──────────────────────────────────────────────────────────────

struct EOUConfig {
    StreamingEncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations = {0, 1, 2, 3, 4};
    int eou_token_id = -1; // End-of-utterance token ID (-1 = disabled)
    int ctc_vocab_size = 1025;
};

inline EOUConfig make_eou_120m_config() {
    EOUConfig cfg;
    cfg.encoder.hidden_size = 512;
    cfg.encoder.num_layers = 17;
    cfg.encoder.num_heads = 8;
    cfg.encoder.ffn_intermediate = 2048;
    cfg.encoder.subsampling_channels = 256;
    cfg.encoder.conv_kernel_size = 9;
    cfg.encoder.att_context_left = 70;
    cfg.encoder.att_context_right = 1;
    cfg.encoder.chunk_size = 20; // ~160ms chunks
    cfg.prediction.vocab_size = 1025;
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 1;
    cfg.joint.encoder_hidden = 512;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 1025;
    cfg.durations = {0, 1, 2, 3, 4};
    cfg.eou_token_id = 1024; // blank acts as EOU for simplicity
    cfg.ctc_vocab_size = 1025;
    return cfg;
}

// ─── Parakeet EOU Model ──────────────────────────────────────────────────────

class ParakeetEOU : public Module {
  public:
    explicit ParakeetEOU(const EOUConfig &config = make_eou_120m_config());

    const EOUConfig &config() const { return config_; }

    StreamingFastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &joint() { return joint_; }

  private:
    EOUConfig config_;
    StreamingFastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    TDTJoint joint_;
};

// ─── Streaming RNNT/TDT Decode ──────────────────────────────────────────────

// Decode state maintained across chunks
struct StreamingDecodeState {
    std::vector<LSTMState> lstm_states;
    Tensor last_token; // (1,) int32
    std::vector<int> tokens;
    bool initialized = false;
};

// Decode a single encoder chunk incrementally.
// Returns new tokens emitted in this chunk.
std::vector<int>
rnnt_streaming_decode_chunk(RNNTPrediction &prediction, TDTJoint &joint,
                            const Tensor &encoder_chunk,
                            const std::vector<int> &durations,
                            StreamingDecodeState &state,
                            int blank_id = 1024,
                            int max_symbols_per_step = 10);

// ─── Streaming Transcriber ───────────────────────────────────────────────────

// Callback for partial transcription results.
using PartialResultCallback = std::function<void(const std::string &partial)>;

class StreamingTranscriber {
  public:
    StreamingTranscriber(const std::string &weights_path,
                         const std::string &vocab_path,
                         const EOUConfig &config = make_eou_120m_config());

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    // Process a chunk of raw audio samples.
    // Returns any new text produced by this chunk.
    std::string transcribe_chunk(const Tensor &samples);

    // Reset state for a new utterance.
    void reset();

    // Set callback for partial results (called each time new tokens are
    // emitted).
    void set_partial_callback(PartialResultCallback cb) {
        partial_callback_ = std::move(cb);
    }

    // Get full transcription so far.
    std::string get_text() const;

    ParakeetEOU &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    EOUConfig config_;
    ParakeetEOU model_;
    Tokenizer tokenizer_;
    StreamingAudioPreprocessor preprocessor_;
    EncoderCache encoder_cache_;
    StreamingDecodeState decode_state_;
    bool use_gpu_ = false;
    PartialResultCallback partial_callback_;
};

} // namespace parakeet
