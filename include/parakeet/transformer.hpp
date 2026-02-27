#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Transformer Config ──────────────────────────────────────────────────────

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

// ─── Standard Transformer Block ──────────────────────────────────────────────

// Pre-norm transformer: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual

class TransformerBlock : public Module {
  public:
    explicit TransformerBlock(const TransformerConfig &config = {});

    // input: (batch, seq, hidden)
    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    LayerNorm norm1_;
    MultiHeadAttention mha_;
    Dropout dropout1_;
    LayerNorm norm2_;
    Linear fc1_;
    Linear fc2_;
    Dropout dropout2_;
    bool pre_ln_ = true;
};

// ─── Transformer Encoder ─────────────────────────────────────────────────────

class TransformerEncoder : public Module {
  public:
    explicit TransformerEncoder(const TransformerConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

  private:
    ModuleList layers_;
    LayerNorm final_norm_;
    bool has_final_norm_ = false;
};

} // namespace parakeet
