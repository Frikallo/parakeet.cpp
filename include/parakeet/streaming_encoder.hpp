#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/encoder.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Streaming Encoder Config ────────────────────────────────────────────────

enum class SubsamplingActivation { SiLU, ReLU };

struct StreamingEncoderConfig : EncoderConfig {
    int att_context_left = 70; // left context frames for attention
    int att_context_right = 0; // right context frames (0 = causal)
    int chunk_size = 20;       // frames per chunk after subsampling
    SubsamplingActivation subsampling_activation = SubsamplingActivation::ReLU;
    bool xscaling = false; // multiply subsampling output by sqrt(d_model)
};

// ─── Block Cache (per-layer state) ──────────────────────────────────────────

struct BlockCache {
    Tensor conv_cache; // (batch, hidden, kernel_size-1) left-padding for causal
                       // conv
    Tensor key_cache;  // (batch, heads, cache_len, head_dim)
    Tensor value_cache; // (batch, heads, cache_len, head_dim)
};

// ─── Full Encoder Cache ──────────────────────────────────────────────────────

struct EncoderCache {
    std::vector<BlockCache> layer_caches;
    Tensor subsampling_cache; // leftover audio frames for subsampling overlap
    int frames_seen = 0;      // total encoder frames processed so far

    bool empty() const { return layer_caches.empty(); }
};

// ─── Causal Conformer Convolution Module ─────────────────────────────────────

// Like ConformerConvModule but with left-only padding for streaming.
// Maintains a conv cache of (kernel_size - 1) frames for causal operation.

class CausalConformerConvModule : public Module {
  public:
    explicit CausalConformerConvModule(int groups = 1, int kernel_size = 9,
                                       float dropout = 0.1f);

    // Non-streaming: full-sequence forward (same as regular
    // ConformerConvModule)
    Tensor forward(const Tensor &input) const;

    // Streaming: forward with cache
    Tensor forward_cached(const Tensor &input, Tensor &conv_cache) const;

    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int kernel_size_;
    LayerNorm norm_;
    Conv1d pointwise_conv1_;
    Conv1d depthwise_conv_; // NO padding — we prepend cache manually
    BatchNorm1d batch_norm_;
    Conv1d pointwise_conv2_;
    Dropout dropout_;
};

// ─── Streaming Conformer Attention ───────────────────────────────────────────

// Multi-head self-attention with bounded context window and KV caching.
// In streaming mode, prepends cached K/V to current chunk, then masks
// attention to [t - left, t + right] context window.

class StreamingConformerAttention : public Module {
  public:
    explicit StreamingConformerAttention(int num_heads = 8,
                                         float dropout = 0.1f);

    // Non-streaming: full-sequence relative position attention
    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;

    // Streaming: forward with KV cache and bounded context
    Tensor forward_cached(const Tensor &input, const Tensor &pos_emb,
                          Tensor &key_cache, Tensor &value_cache,
                          int att_context_left, int att_context_right) const;

    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    Linear pos_proj_;
    Dropout dropout_;
    Tensor pos_bias_u_;
    Tensor pos_bias_v_;

    Tensor rel_position_attention(const Tensor &query, const Tensor &key,
                                  const Tensor &value, const Tensor &pos_emb,
                                  const Tensor &mask) const;
    static Tensor rel_shift(const Tensor &x);
};

// ─── Streaming Conformer Block ───────────────────────────────────────────────

class StreamingConformerBlock : public Module {
  public:
    explicit StreamingConformerBlock(const StreamingEncoderConfig &config = {});

    // Non-streaming
    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;

    // Streaming
    Tensor forward_cached(const Tensor &input, const Tensor &pos_emb,
                          BlockCache &cache, int att_context_left,
                          int att_context_right) const;

    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

  private:
    StreamingEncoderConfig config_;
    FeedForward ffn1_;
    StreamingConformerAttention attn_;
    CausalConformerConvModule conv_;
    FeedForward ffn2_;
    LayerNorm final_norm_;
};

// ─── Causal Conv Subsampling ─────────────────────────────────────────────────

// Same architecture as ConvSubsampling but can process chunks with overlap
// cache.

class CausalConvSubsampling : public Module {
  public:
    explicit CausalConvSubsampling(
        int channels = 256,
        SubsamplingActivation act = SubsamplingActivation::SiLU);

    // Non-streaming: full-sequence (identical to ConvSubsampling)
    Tensor forward(const Tensor &input) const;

    // Streaming: process chunk with overlap cache
    Tensor forward_cached(const Tensor &input, Tensor &cache) const;

    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv2d conv1_;
    Conv2d dw1_, dw2_;
    Conv2d conv2_, conv3_;
    Linear proj_;
    SubsamplingActivation activation_;
};

// ─── Streaming FastConformer Encoder ─────────────────────────────────────────

class StreamingFastConformerEncoder : public Module {
  public:
    explicit StreamingFastConformerEncoder(
        const StreamingEncoderConfig &config = {});

    // Non-streaming: full-sequence forward (produces same output as batch mode)
    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;

    // Streaming: process a chunk, updating cache state
    Tensor forward_chunk(const Tensor &input, EncoderCache &cache) const;

    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

    const StreamingEncoderConfig &config() const { return config_; }

  private:
    StreamingEncoderConfig config_;
    CausalConvSubsampling subsampling_;
    ModuleList layers_;

    // Initialize cache with empty state for the first chunk
    void init_cache(EncoderCache &cache, size_t batch_size) const;
};

} // namespace parakeet
