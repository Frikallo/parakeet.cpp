#pragma once

#include <vector>

namespace parakeet {

// ─── Encoder Config ─────────────────────────────────────────────────────────

struct EncoderConfig {
    int mel_bins = 80;
    int subsampling_factor = 8;
    int subsampling_channels = 256;
    int hidden_size = 1024;
    int num_layers = 24;
    int num_heads = 8;
    int ffn_intermediate = 4096;
    int conv_kernel_size = 9;
    float dropout = 0.1f;
    float layer_norm_eps = 1e-5f;
};

// ─── CTC Config ─────────────────────────────────────────────────────────────

struct CTCConfig {
    EncoderConfig encoder;
    int vocab_size = 1025; // 1024 tokens + 1 blank
};

// ─── Prediction Network Config (shared by RNNT / TDT) ──────────────────────

struct PredictionConfig {
    int vocab_size = 1025;
    int pred_hidden = 640;
    int num_lstm_layers = 2;
    float dropout = 0.1f;
};

// ─── Joint Network Config ───────────────────────────────────────────────────

struct JointConfig {
    int encoder_hidden = 1024;
    int pred_hidden = 640;
    int joint_hidden = 640;
    int vocab_size = 1025;
};

// ─── RNNT Config ────────────────────────────────────────────────────────────

struct RNNTConfig {
    EncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
};

// ─── TDT Config ─────────────────────────────────────────────────────────────

struct TDTConfig {
    EncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations = {0, 1, 2, 3, 4};
};

// ─── TDT-CTC Hybrid Config ─────────────────────────────────────────────────

struct TDTCTCConfig {
    EncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations = {0, 1, 2, 3, 4};
    int ctc_vocab_size = 1025;
};

// ─── Presets ────────────────────────────────────────────────────────────────

// nvidia/parakeet-tdt_ctc-110m (110M params, 459MB .nemo)
inline TDTCTCConfig make_110m_config() {
    TDTCTCConfig cfg;
    cfg.encoder.hidden_size = 512;
    cfg.encoder.num_layers = 17;
    cfg.encoder.num_heads = 8;
    cfg.encoder.ffn_intermediate = 2048;
    cfg.encoder.subsampling_channels = 256;
    cfg.encoder.conv_kernel_size = 9;
    cfg.prediction.vocab_size = 1025;
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 1;
    cfg.joint.encoder_hidden = 512;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 1025;
    cfg.durations = {0, 1, 2, 3, 4};
    cfg.ctc_vocab_size = 1025;
    return cfg;
}

} // namespace parakeet
