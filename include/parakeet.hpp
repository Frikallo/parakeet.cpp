#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Model Config ───────────────────────────────────────────────────────────

struct ParakeetConfig {
    int mel_bins = 80;
    int subsampling_factor = 8;
    int subsampling_channels = 256;
    int hidden_size = 1024;
    int num_layers = 24;
    int num_heads = 8;
    int ffn_intermediate = 4096;
    int conv_kernel_size = 9;
    int vocab_size = 1025; // 1024 tokens + 1 blank/pad
    float dropout = 0.1f;
    float layer_norm_eps = 1e-5f;
};

// ─── Batch Normalization ────────────────────────────────────────────────────
// Axiom doesn't provide BatchNorm, so we implement it for the conv module.

class BatchNorm1d : public Module {
  public:
    explicit BatchNorm1d(float eps = 1e-5f);

    // input: (batch, channels, length)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;        // (channels,)
    Tensor bias_;          // (channels,)
    Tensor running_mean_;  // (channels,)
    Tensor running_var_;   // (channels,)
    float eps_;
};

// ─── Feed-Forward Module (Macaron-style half-step) ──────────────────────────

class FeedForward : public Module {
  public:
    FeedForward();

    // input: (batch, seq, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    LayerNorm norm_;
    Linear fc1_; // hidden_size → ffn_intermediate
    Linear fc2_; // ffn_intermediate → hidden_size
};

// ─── Conformer Convolution Module ───────────────────────────────────────────

class ConformerConvModule : public Module {
  public:
    ConformerConvModule();

    // input: (batch, seq, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    LayerNorm norm_;
    Conv1d pointwise_conv1_; // hidden_size → 2*hidden_size (for GLU)
    Conv1d depthwise_conv_;  // groups=hidden_size, kernel_size=9
    BatchNorm1d batch_norm_;
    Conv1d pointwise_conv2_; // hidden_size → hidden_size
};

// ─── Multi-Head Self-Attention with Relative Positional Encoding ────────────

class ConformerAttention : public Module {
  public:
    ConformerAttention();

    // input: (batch, seq, hidden_size)
    Tensor forward(const Tensor &input,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    LayerNorm norm_;
    MultiHeadAttention mha_;
    Linear pos_proj_; // relative positional encoding projection
};

// ─── Conformer Block ────────────────────────────────────────────────────────
// Macaron structure: FFN → MHSA → Conv → FFN (each with residual + LN)

class ConformerBlock : public Module {
  public:
    ConformerBlock();

    // input: (batch, seq, hidden_size)
    Tensor forward(const Tensor &input,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    FeedForward ffn1_;          // first half-step FFN
    ConformerAttention attn_;   // MHSA
    ConformerConvModule conv_;  // convolution module
    FeedForward ffn2_;          // second half-step FFN
    LayerNorm final_norm_;      // final layer norm
};

// ─── Convolutional Subsampling ──────────────────────────────────────────────
// 3 depthwise-separable conv layers with stride 2 each → 8x reduction

class ConvSubsampling : public Module {
  public:
    ConvSubsampling();

    // input: (batch, mel_length, mel_bins)
    // output: (batch, mel_length/8, hidden_size)
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    // 3 stages of depthwise-separable convolution (stride 2 each)
    Conv1d depthwise1_;
    Conv1d pointwise1_;
    Conv1d depthwise2_;
    Conv1d pointwise2_;
    Conv1d depthwise3_;
    Conv1d pointwise3_;
    Linear proj_; // project subsampling_channels → hidden_size
};

// ─── FastConformer Encoder ──────────────────────────────────────────────────

class FastConformerEncoder : public Module {
  public:
    FastConformerEncoder();

    // input: (batch, mel_length, mel_bins)
    // output: (batch, seq_length, hidden_size)
    Tensor forward(const Tensor &input,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    ConvSubsampling subsampling_;
    ModuleList layers_; // 24x ConformerBlock
};

// ─── CTC Decoder Head ──────────────────────────────────────────────────────

class CTCDecoder : public Module {
  public:
    CTCDecoder();

    // input: (batch, seq_length, hidden_size)
    // output: (batch, seq_length, vocab_size) — log probabilities
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Linear proj_; // hidden_size → vocab_size
};

// ─── Parakeet CTC Model ────────────────────────────────────────────────────

class ParakeetCTC : public Module {
  public:
    explicit ParakeetCTC(const ParakeetConfig &config = {});

    // input: (batch, mel_length, mel_bins) — log-mel spectrogram
    // output: (batch, seq_length, vocab_size) — log probabilities
    Tensor forward(const Tensor &input,
                   const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

    const ParakeetConfig &config() const { return config_; }

  private:
    ParakeetConfig config_;
    FastConformerEncoder encoder_;
    CTCDecoder decoder_;
};

} // namespace parakeet
