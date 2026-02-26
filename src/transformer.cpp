#include "parakeet/transformer.hpp"

#include <cmath>

namespace parakeet {

// ─── TransformerBlock ────────────────────────────────────────────────────────

TransformerBlock::TransformerBlock(const TransformerConfig &config)
    : mha_(config.num_heads), dropout1_(config.dropout), fc1_(true),
      fc2_(true), dropout2_(config.dropout) {
    AX_REGISTER_MODULES(norm1_, mha_, dropout1_, norm2_, fc1_, fc2_, dropout2_);
}

Tensor TransformerBlock::forward(const Tensor &input,
                                  const Tensor &mask) const {
    // Self-attention with pre-norm
    auto x = norm1_(input);

    // Standard MHA (no relative position encoding)
    auto q = mha_.q_proj()(x);
    auto k = mha_.k_proj()(x);
    auto v = mha_.v_proj()(x);

    int num_heads = mha_.num_heads();
    auto d_model = static_cast<int>(q.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = q.shape()[0];
    auto seq_len = q.shape()[1];
    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);

    q = q.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    k = k.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    v = v.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});

    auto scores = ops::matmul(q, k, false, true) * scale;

    if (mask.storage()) {
        scores = ops::masked_fill(scores, mask, -1e9f);
    }

    auto attn_weights = ops::softmax(scores, -1);
    auto out = ops::matmul(attn_weights, v);

    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, seq_len, static_cast<size_t>(d_model)});
    out = mha_.out_proj()(out);
    out = dropout1_(out);
    x = input + out;

    // FFN with pre-norm
    auto ffn_in = norm2_(x);
    auto ffn_out = fc1_(ffn_in);
    ffn_out = ops::relu(ffn_out);
    ffn_out = dropout2_(ffn_out);
    ffn_out = fc2_(ffn_out);
    ffn_out = dropout2_(ffn_out);

    return x + ffn_out;
}

// ─── TransformerEncoder ──────────────────────────────────────────────────────

TransformerEncoder::TransformerEncoder(const TransformerConfig &config) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back<TransformerBlock>(config);
    }
    AX_REGISTER_MODULES(layers_, final_norm_);
}

Tensor TransformerEncoder::forward(const Tensor &input,
                                    const Tensor &mask) const {
    auto x = input;
    for (const auto &block : layers_.each<TransformerBlock>()) {
        x = block(x, mask);
    }
    return final_norm_(x);
}

} // namespace parakeet
