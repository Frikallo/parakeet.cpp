#pragma once

#include <vector>

namespace parakeet::models {

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

// ─── Subsampling Activation ─────────────────────────────────────────────────

enum class SubsamplingActivation { SiLU, ReLU };

// ─── Streaming Encoder Config ───────────────────────────────────────────────

struct StreamingEncoderConfig : EncoderConfig {
    int att_context_left = 70; // left context frames for attention
    int att_context_right = 0; // right context frames (0 = causal)
    int chunk_size = 20;       // frames per chunk after subsampling
    SubsamplingActivation subsampling_activation = SubsamplingActivation::ReLU;
    bool xscaling = false; // multiply subsampling output by sqrt(d_model)
};

// ─── Transformer Config ─────────────────────────────────────────────────────

struct TransformerConfig {
    int hidden_size = 192;
    int num_layers = 18;
    int num_heads = 8;
    int ffn_intermediate = 768;
    float dropout = 0.1f;
    float layer_norm_eps = 1e-5f;
    bool pre_ln = true; // true=pre-norm, false=post-norm
    bool has_final_norm = false;
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

// ─── EOU Config ─────────────────────────────────────────────────────────────

struct EOUConfig {
    StreamingEncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations = {0, 1, 2, 3, 4};
    int eou_token_id = -1; // End-of-utterance token ID (-1 = disabled)
    int ctc_vocab_size = 1025;
};

// ─── Nemotron Config ────────────────────────────────────────────────────────

struct NemotronConfig {
    StreamingEncoderConfig encoder;
    PredictionConfig prediction;
    JointConfig joint;
    std::vector<int> durations; // empty = pure RNNT (no duration head)

    // Latency mode: configurable right context
    // 0 frames = 80ms latency, 1 = 160ms, 6 = 560ms, 13 = 1120ms
    int latency_frames = 0;
};

// ─── Sortformer Config ──────────────────────────────────────────────────────

struct DiarizationSegment {
    int speaker_id;
    float start; // seconds
    float end;   // seconds
};

struct SortformerConfig {
    // NEST encoder (FastConformer)
    StreamingEncoderConfig nest_encoder;

    // Projection from encoder hidden to transformer hidden
    int encoder_hidden = 512; // NEST output dim
    int transformer_hidden = 192;

    // Sortformer transformer
    TransformerConfig transformer;

    int max_speakers = 4;
    float activity_threshold = 0.5f; // sigmoid threshold for VAD
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

// nvidia/parakeet-tdt-0.6b-v3 (600M params, multilingual, 128 mel bins)
inline TDTConfig make_tdt_600m_config() {
    TDTConfig cfg;
    cfg.encoder.mel_bins = 128;
    cfg.encoder.hidden_size = 1024;
    cfg.encoder.num_layers = 24;
    cfg.encoder.num_heads = 8;
    cfg.encoder.ffn_intermediate = 4096;
    cfg.encoder.subsampling_channels = 256;
    cfg.encoder.conv_kernel_size = 9;
    cfg.prediction.vocab_size = 8193; // 8192 BPE + 1 blank
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 2;
    cfg.joint.encoder_hidden = 1024;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 8193;
    cfg.durations = {0, 1, 2, 3, 4};
    return cfg;
}

// nvidia/parakeet-rnnt-0.6b (600M params, English RNNT)
inline RNNTConfig make_rnnt_600m_config() {
    RNNTConfig cfg;
    cfg.encoder.hidden_size = 1024;
    cfg.encoder.num_layers = 24;
    cfg.encoder.num_heads = 8;
    cfg.encoder.ffn_intermediate = 4096;
    cfg.encoder.subsampling_channels = 256;
    cfg.encoder.conv_kernel_size = 9;
    cfg.prediction.vocab_size = 1025;
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 2;
    cfg.joint.encoder_hidden = 1024;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 1025;
    return cfg;
}

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
    cfg.prediction.vocab_size = 1025;
    cfg.prediction.pred_hidden = 640;
    cfg.prediction.num_lstm_layers = 2;
    cfg.joint.encoder_hidden = 1024;
    cfg.joint.pred_hidden = 640;
    cfg.joint.joint_hidden = 640;
    cfg.joint.vocab_size = 1025;
    // Nemotron is pure RNNT — no duration head
    cfg.latency_frames = latency_frames;
    return cfg;
}

inline SortformerConfig make_sortformer_117m_config() {
    SortformerConfig cfg;
    // NEST encoder: 17-layer FastConformer, 128 mel bins
    cfg.nest_encoder.mel_bins = 128;
    cfg.nest_encoder.hidden_size = 512;
    cfg.nest_encoder.num_layers = 17;
    cfg.nest_encoder.num_heads = 8;
    cfg.nest_encoder.ffn_intermediate = 2048;
    cfg.nest_encoder.subsampling_channels = 256;
    cfg.nest_encoder.conv_kernel_size = 9;
    cfg.nest_encoder.att_context_left = 70;
    cfg.nest_encoder.att_context_right = 0;
    cfg.nest_encoder.chunk_size = 20;
    cfg.nest_encoder.subsampling_activation = SubsamplingActivation::ReLU;
    cfg.nest_encoder.xscaling = true; // NeMo default: multiply by sqrt(d_model)
    cfg.encoder_hidden = 512;

    // Transformer decoder (post-norm with final layer norm)
    cfg.transformer_hidden = 192;
    cfg.transformer.hidden_size = 192;
    cfg.transformer.num_layers = 18;
    cfg.transformer.num_heads = 8;
    cfg.transformer.ffn_intermediate = 768;
    cfg.transformer.pre_ln = false; // NeMo sortformer uses post-norm
    cfg.transformer.has_final_norm = false;

    cfg.max_speakers = 4;
    cfg.activity_threshold = 0.5f;
    return cfg;
}

} // namespace parakeet::models
