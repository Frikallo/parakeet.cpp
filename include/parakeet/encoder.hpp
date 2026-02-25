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
    FeedForward();

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
    ConformerConvModule();

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
    ConformerAttention();

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    Linear pos_proj_;
    Dropout dropout_;
};

// ─── Conformer Block ────────────────────────────────────────────────────────

class ConformerBlock : public Module {
  public:
    ConformerBlock();

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    FeedForward ffn1_;
    ConformerAttention attn_;
    ConformerConvModule conv_;
    FeedForward ffn2_;
    LayerNorm final_norm_;
};

// ─── Convolutional Subsampling (Conv2d, matches NeMo FastConformer) ─────────
//
// NeMo structure (nn.Sequential indices):
//   [0] Conv2d(1, C, 3, stride=2, pad=1)        — regular
//   [1] activation
//   [2] Conv2d(C, C, 3, groups=C, stride=1, pad=1) — depthwise
//   [3] Conv2d(C, C, 3, stride=2, pad=1)        — regular
//   [4] activation
//   [5] Conv2d(C, C, 3, groups=C, stride=1, pad=1) — depthwise
//   [6] Conv2d(C, C, 3, stride=2, pad=1)        — regular
//   [7] activation
//   [8] Conv2d(C, C, 3, groups=C, stride=1, pad=1) — depthwise
//   Linear(C * ceil(mel_bins/8), d_model)        — projection

class ConvSubsampling : public Module {
  public:
    explicit ConvSubsampling(int channels = 256);

    // (batch, mel_length, mel_bins) → (batch, mel_length/8, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv2d conv1_, conv2_, conv3_; // regular convs (stride=2)
    Conv2d dw1_, dw2_, dw3_;       // depthwise convs (groups=channels)
    Linear proj_;
};

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
