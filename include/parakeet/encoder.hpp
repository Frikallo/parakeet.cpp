#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Feed-Forward Module (Macaron-style half-step) ──────────────────────────

class FeedForward : public Module {
  public:
    explicit FeedForward(float dropout = 0.1f);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    LayerNorm norm_;
    Linear fc1_;
    Linear fc2_;
    Dropout dropout_;
};

// ─── Conformer Convolution Module ───────────────────────────────────────────

class ConformerConvModule : public Module {
  public:
    explicit ConformerConvModule(int groups = 1, float dropout = 0.1f);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    LayerNorm norm_;
    Conv1d pointwise_conv1_; // hidden_size → 2*hidden_size (for GLU)
    Conv1d depthwise_conv_;  // groups=hidden_size, kernel_size=9
    BatchNorm1d batch_norm_;
    Conv1d pointwise_conv2_; // hidden_size → hidden_size
    Dropout dropout_;
};

// ─── Multi-Head Self-Attention with Relative Positional Encoding ────────────

class ConformerAttention : public Module {
  public:
    explicit ConformerAttention(int num_heads = 8, float dropout = 0.1f);

    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    Linear pos_proj_;
    Dropout dropout_;
    Tensor pos_bias_u_; // (num_heads, head_dim) — learned position bias
    Tensor pos_bias_v_; // (num_heads, head_dim) — learned position bias

    // Relative position attention (bypasses mha_.forward)
    Tensor rel_position_attention(const Tensor &query, const Tensor &key,
                                  const Tensor &value, const Tensor &pos_emb,
                                  const Tensor &mask) const;
    static Tensor rel_shift(const Tensor &x);
};

// ─── Conformer Block ────────────────────────────────────────────────────────

class ConformerBlock : public Module {
  public:
    explicit ConformerBlock(const EncoderConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &pos_emb,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input, const Tensor &pos_emb,
                      const Tensor &mask = Tensor()) const {
        return forward(input, pos_emb, mask);
    }

  private:
    FeedForward ffn1_;
    ConformerAttention attn_;
    ConformerConvModule conv_;
    FeedForward ffn2_;
    LayerNorm final_norm_;
};

// ─── Convolutional Subsampling (Conv2d, matches NeMo FastConformer 110M) ────
//
// NeMo structure (nn.Sequential indices):
//   [0] Conv2d(1, C, 3, stride=2, pad=1)              — regular strided
//   [1] SiLU
//   [2] Conv2d(C, C, 3, groups=C, stride=2, pad=1)    — depthwise strided
//   [3] Conv2d(C, C, 1)                                — pointwise
//   [4] SiLU
//   [5] Conv2d(C, C, 3, groups=C, stride=2, pad=1)    — depthwise strided
//   [6] Conv2d(C, C, 1)                                — pointwise
//   Linear(C * mel_bins/8, d_model)                    — projection
//
// Total downsample: stride-2 × 3 = 8×

class ConvSubsampling : public Module {
  public:
    explicit ConvSubsampling(int channels = 256);

    // (batch, mel_length, mel_bins) → (batch, mel_length/8, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv2d conv1_;         // regular strided 3×3 (stride=2)
    Conv2d dw1_, dw2_;     // depthwise strided 3×3 (stride=2, groups=C)
    Conv2d conv2_, conv3_; // pointwise 1×1 (stride=1)
    Linear proj_;
};

// ─── Sinusoidal Positional Embedding ──────────────────────────────────────

// Generate sinusoidal position embeddings for relative positions.
// Returns (2*seq_len - 1, d_model) encoding relative positions -(seq_len-1) to
// +(seq_len-1).
Tensor sinusoidal_position_embedding(int seq_len, int d_model);

// ─── FastConformer Encoder ──────────────────────────────────────────────────

class FastConformerEncoder : public Module {
  public:
    explicit FastConformerEncoder(const EncoderConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    ConvSubsampling subsampling_;
    ModuleList layers_;
};

} // namespace parakeet
