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

// ─── Convolutional Subsampling ──────────────────────────────────────────────

class ConvSubsampling : public Module {
  public:
    ConvSubsampling();

    // (batch, mel_length, mel_bins) → (batch, mel_length/8, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv1d depthwise1_, depthwise2_, depthwise3_;
    Conv1d pointwise1_, pointwise2_, pointwise3_;
    Linear proj_;
};

// ─── FastConformer Encoder ──────────────────────────────────────────────────

class FastConformerEncoder : public Module {
  public:
    FastConformerEncoder();

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
