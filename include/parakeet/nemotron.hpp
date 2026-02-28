#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/eou.hpp"
#include "parakeet/streaming_encoder.hpp"
#include "parakeet/tdt.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Nemotron Config ─────────────────────────────────────────────────────────

struct NemotronConfig {
    StreamingEncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations = {0, 1, 2, 3, 4};

    // Latency mode: configurable right context
    // 0 frames = 80ms latency, 1 = 160ms, 6 = 560ms, 13 = 1120ms
    int latency_frames = 0;
};

// Default: 80ms latency (att_context_right=0)
inline NemotronConfig make_nemotron_600m_config(int latency_frames = 0) {
    NemotronConfig cfg;
    cfg.encoder.hidden_size = 1024;
    cfg.encoder.num_layers = 24;
    cfg.encoder.num_heads = 8;
    cfg.encoder.ffn_intermediate = 4096;
    cfg.encoder.subsampling_channels = 256;
    cfg.encoder.conv_kernel_size = 9;
    cfg.encoder.att_context_left = 70;
    cfg.encoder.att_context_right = latency_frames;
    cfg.encoder.chunk_size = 20;
    cfg.prediction.vocab_size = 8193;
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 2;
    cfg.joint.encoder_hidden = 1024;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 8193;
    cfg.durations = {0, 1, 2, 3, 4};
    cfg.latency_frames = latency_frames;
    return cfg;
}

// ─── Parakeet Nemotron Model ─────────────────────────────────────────────────

class ParakeetNemotron : public Module {
  public:
    explicit ParakeetNemotron(
        const NemotronConfig &config = make_nemotron_600m_config());

    const NemotronConfig &config() const { return config_; }

    StreamingFastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &joint() { return joint_; }

  private:
    NemotronConfig config_;
    StreamingFastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    TDTJoint joint_;
};

// ─── Nemotron Streaming Transcriber ──────────────────────────────────────────

class NemotronTranscriber {
  public:
    NemotronTranscriber(
        const std::string &weights_path, const std::string &vocab_path,
        const NemotronConfig &config = make_nemotron_600m_config());

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    // Process a chunk of raw audio → returns new text from this chunk
    std::string transcribe_chunk(const Tensor &samples);

    // Convenience: process raw float32 PCM buffer.
    std::string transcribe_chunk(const float *data, size_t num_samples) {
        auto t = Tensor::from_data(data, Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Convenience: process raw int16 PCM buffer (converted to float32).
    std::string transcribe_chunk(const int16_t *data, size_t num_samples) {
        std::vector<float> f(num_samples);
        for (size_t i = 0; i < num_samples; ++i)
            f[i] = static_cast<float>(data[i]) / 32768.0f;
        auto t = Tensor::from_data(f.data(), Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Reset for a new utterance
    void reset();

    // Get full transcription so far
    std::string get_text() const;

    void set_partial_callback(PartialResultCallback cb) {
        partial_callback_ = std::move(cb);
    }

    ParakeetNemotron &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    NemotronConfig config_;
    ParakeetNemotron model_;
    Tokenizer tokenizer_;
    StreamingAudioPreprocessor preprocessor_;
    EncoderCache encoder_cache_;
    StreamingDecodeState decode_state_;
    bool use_gpu_ = false;
    PartialResultCallback partial_callback_;
};

} // namespace parakeet
