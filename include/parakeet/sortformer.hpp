#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/streaming_encoder.hpp"
#include "parakeet/transformer.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Diarization Types ───────────────────────────────────────────────────────

struct DiarizationSegment {
    int speaker_id;
    float start; // seconds
    float end;   // seconds
};

// ─── Sortformer Config ───────────────────────────────────────────────────────

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

inline SortformerConfig make_sortformer_117m_config() {
    SortformerConfig cfg;
    // NEST encoder: 17-layer FastConformer
    cfg.nest_encoder.hidden_size = 512;
    cfg.nest_encoder.num_layers = 17;
    cfg.nest_encoder.num_heads = 8;
    cfg.nest_encoder.ffn_intermediate = 2048;
    cfg.nest_encoder.subsampling_channels = 256;
    cfg.nest_encoder.conv_kernel_size = 9;
    cfg.nest_encoder.att_context_left = 70;
    cfg.nest_encoder.att_context_right = 0;
    cfg.nest_encoder.chunk_size = 20;
    cfg.encoder_hidden = 512;

    // Transformer decoder
    cfg.transformer_hidden = 192;
    cfg.transformer.hidden_size = 192;
    cfg.transformer.num_layers = 18;
    cfg.transformer.num_heads = 4;
    cfg.transformer.ffn_intermediate = 768;

    cfg.max_speakers = 4;
    cfg.activity_threshold = 0.5f;
    return cfg;
}

// ─── AOSC (Arrival-Order Speaker Cache) ──────────────────────────────────────

class AOSCCache {
  public:
    explicit AOSCCache(int max_speakers = 4);

    // Update cache with new frame activity scores.
    // probs: (T, max_speakers) sigmoid probabilities
    void update(const Tensor &probs);

    // Get speaker ordering (arrival order).
    // Returns mapping: output_speaker_id → internal_speaker_id
    std::vector<int> speaker_order() const;

    void reset();

  private:
    int max_speakers_;
    std::vector<bool> speaker_active_;
    std::vector<int> arrival_order_; // speakers in order of first appearance
};

// ─── Sortformer Model ────────────────────────────────────────────────────────

class Sortformer : public Module {
  public:
    explicit Sortformer(const SortformerConfig &config =
                            make_sortformer_117m_config());

    // Batch diarization: (1, n_frames, n_mels) → segments
    std::vector<DiarizationSegment> diarize(const Tensor &features) const;

    // Streaming: process a chunk, return segments so far
    std::vector<DiarizationSegment>
    diarize_chunk(const Tensor &features, EncoderCache &enc_cache,
                  AOSCCache &aosc_cache) const;

    // Raw forward: features → (batch, T, max_speakers) sigmoid probs
    Tensor forward(const Tensor &features) const;

    const SortformerConfig &config() const { return config_; }

  private:
    SortformerConfig config_;
    StreamingFastConformerEncoder nest_encoder_;
    Linear projection_;       // encoder_hidden → transformer_hidden
    TransformerEncoder transformer_;
    Linear output_proj_;      // transformer_hidden → max_speakers

    // Post-process sigmoid probabilities to segments
    std::vector<DiarizationSegment>
    probs_to_segments(const Tensor &probs) const;
};

} // namespace parakeet
