#include "parakeet/encoder.hpp"

#include <cmath>

namespace parakeet {

// ─── Sinusoidal Position Embedding ──────────────────────────────────────────

Tensor sinusoidal_position_embedding(int seq_len, int d_model) {
    // Generate embeddings for positions (seq_len-1) down to -(seq_len-1)
    // matching NeMo's RelPositionalEncoding.
    int total = 2 * seq_len - 1;
    auto pe = Tensor::zeros(
        {static_cast<size_t>(total), static_cast<size_t>(d_model)});

    float *pe_data = pe.typed_data<float>();
    for (int pos_idx = 0; pos_idx < total; ++pos_idx) {
        float position = static_cast<float>(seq_len - 1 - pos_idx);
        for (int i = 0; i < d_model; i += 2) {
            float div_term = std::exp(static_cast<float>(i) *
                                      (-std::log(10000.0f) / d_model));
            pe_data[pos_idx * d_model + i] = std::sin(position * div_term);
            if (i + 1 < d_model) {
                pe_data[pos_idx * d_model + i + 1] =
                    std::cos(position * div_term);
            }
        }
    }
    return pe; // (2*seq_len-1, d_model)
}

// ─── FeedForward ────────────────────────────────────────────────────────────

FeedForward::FeedForward(float dropout)
    : fc1_(true), fc2_(true), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, fc1_, fc2_, dropout_);
}

Tensor FeedForward::forward(const Tensor &input) const {
    auto x = norm_(input);
    x = fc1_(x);
    x = ops::silu(x);
    x = dropout_(x);
    x = fc2_(x);
    return input + x * 0.5f; // macaron half-step
}

// ─── ConformerConvModule ────────────────────────────────────────────────────

ConformerConvModule::ConformerConvModule(int groups, float dropout)
    : pointwise_conv1_(/*stride=*/1),
      depthwise_conv_(/*stride=*/1, /*padding=*/4, /*dilation=*/1,
                      /*groups=*/groups),
      pointwise_conv2_(/*stride=*/1), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, pointwise_conv1_, depthwise_conv_, batch_norm_,
                        pointwise_conv2_, dropout_);
}

Tensor ConformerConvModule::forward(const Tensor &input) const {
    auto x = norm_(input);
    x = x.permute({0, 2, 1}); // (batch, hidden, seq) for conv1d

    x = pointwise_conv1_(x); // (batch, 2*hidden, seq)
    x = ops::glu(x, /*dim=*/1);

    x = depthwise_conv_(x);
    x = batch_norm_(x);
    x = ops::silu(x);

    x = pointwise_conv2_(x);
    x = dropout_(x);
    x = x.permute({0, 2, 1}); // back to (batch, seq, hidden)

    return input + x;
}

// ─── ConformerAttention ─────────────────────────────────────────────────────

ConformerAttention::ConformerAttention(int num_heads, float dropout)
    : mha_(num_heads), pos_proj_(false), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, mha_, pos_proj_, dropout_);
    AX_REGISTER_PARAMETERS(pos_bias_u_, pos_bias_v_);
}

Tensor ConformerAttention::rel_shift(const Tensor &x) {
    // x: (batch, heads, seq_len, 2*seq_len-1)
    // Returns: (batch, heads, seq_len, seq_len)
    auto shape = x.shape();
    size_t batch = shape[0];
    size_t heads = shape[1];
    size_t seq_len = shape[2];
    size_t pos_len = shape[3]; // 2*seq_len - 1

    // Pad left column with zero: (batch, heads, seq_len, 2*seq_len)
    auto padded = ops::pad(x, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});

    // Reshape to (batch, heads, 2*seq_len, seq_len)
    padded = padded.reshape({batch, heads, pos_len + 1, seq_len});

    // Slice off first row: (batch, heads, 2*seq_len-1, seq_len)
    padded = padded.slice({Slice(), Slice(), Slice(1), Slice()});

    // Reshape back: (batch, heads, seq_len, 2*seq_len-1)
    padded = padded.reshape({batch, heads, seq_len, pos_len});

    // Take first seq_len columns: (batch, heads, seq_len, seq_len)
    return padded.slice(
        {Slice(), Slice(), Slice(), Slice(0, static_cast<int64_t>(seq_len))});
}

Tensor ConformerAttention::rel_position_attention(const Tensor &query,
                                                  const Tensor &key,
                                                  const Tensor &value,
                                                  const Tensor &pos_emb,
                                                  const Tensor &mask) const {
    // query/key/value: (batch, seq, d_model)
    // pos_emb: (2*seq-1, d_model)

    int num_heads = mha_.num_heads();
    auto q = mha_.q_proj()(query);
    auto k = mha_.k_proj()(key);
    auto v = mha_.v_proj()(value);

    auto d_model = static_cast<int>(q.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = q.shape()[0];
    auto seq_len = q.shape()[1];

    // Reshape to multi-head: (batch, seq, heads, head_dim) → (batch, heads,
    // seq, head_dim)
    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);
    q = q.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    k = k.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    v = v.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});

    // q: (batch, heads, seq, head_dim)
    // pos_bias_u/v: (heads, head_dim) → broadcast as (1, heads, 1, head_dim)
    auto bias_u = pos_bias_u_.reshape({1, nh, 1, hd});
    auto bias_v = pos_bias_v_.reshape({1, nh, 1, hd});

    // Content attention: (Q + pos_bias_u) @ K^T → (batch, heads, seq, seq)
    auto content_score = ops::matmul(q + bias_u, k, false, true);

    // Position attention: project position embeddings
    auto p = pos_proj_(pos_emb); // (2*seq-1, d_model)
    auto pos_len = p.shape()[0];
    // Reshape to (1, 2*seq-1, heads, head_dim) → (1, heads, 2*seq-1, head_dim)
    p = p.reshape({1, pos_len, nh, hd}).transpose({0, 2, 1, 3});

    // (Q + pos_bias_v) @ P^T → (batch, heads, seq, 2*seq-1)
    auto pos_score = ops::matmul(q + bias_v, p, false, true);

    // Shift to align relative positions
    pos_score = rel_shift(pos_score);

    // Combined scores
    auto scores = (content_score + pos_score) * scale;

    // Apply mask if present
    if (mask.storage()) {
        scores = ops::masked_fill(scores, mask, -1e9f);
    }

    // Softmax over last dim
    auto attn_weights = ops::softmax(scores, -1);

    // Weighted sum: (batch, heads, seq, head_dim)
    auto out = ops::matmul(attn_weights, v);

    // Reshape back: (batch, seq, d_model)
    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, seq_len, static_cast<size_t>(d_model)});

    return mha_.out_proj()(out);
}

Tensor ConformerAttention::forward(const Tensor &input, const Tensor &pos_emb,
                                   const Tensor &mask) const {
    auto x = norm_(input);
    x = rel_position_attention(x, x, x, pos_emb, mask);
    x = dropout_(x);
    return input + x;
}

// ─── ConformerBlock ─────────────────────────────────────────────────────────

ConformerBlock::ConformerBlock(const EncoderConfig &config)
    : ffn1_(config.dropout), attn_(config.num_heads, config.dropout),
      conv_(config.hidden_size, config.dropout), ffn2_(config.dropout) {
    AX_REGISTER_MODULES(ffn1_, attn_, conv_, ffn2_, final_norm_);
}

Tensor ConformerBlock::forward(const Tensor &input, const Tensor &pos_emb,
                               const Tensor &mask) const {
    auto x = ffn1_(input);
    x = attn_(x, pos_emb, mask);
    x = conv_(x);
    x = ffn2_(x);
    x = final_norm_(x);
    return x;
}

// ─── ConvSubsampling (Conv2d) ────────────────────────────────────────────────

ConvSubsampling::ConvSubsampling(int channels)
    : conv1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      dw1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      dw2_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      conv2_(/*stride=*/{1, 1}, /*padding=*/{0, 0}),
      conv3_(/*stride=*/{1, 1}, /*padding=*/{0, 0}), proj_(true) {
    AX_REGISTER_MODULES(conv1_, dw1_, conv2_, dw2_, conv3_, proj_);
}

Tensor ConvSubsampling::forward(const Tensor &input) const {
    // input: (batch, mel_length, mel_bins)
    auto x = input.unsqueeze(1); // (batch, 1, mel_length, mel_bins)

    x = conv1_(x);
    x = ops::silu(x);

    x = dw1_(x);
    x = conv2_(x);
    x = ops::silu(x);

    x = dw2_(x);
    x = conv3_(x);
    x = ops::silu(x);

    // Flatten channels and freq: (batch, C, T/8, F/8) → (batch, T/8, C*F/8)
    auto shape = x.shape();
    x = x.permute({0, 2, 1, 3}); // (batch, T/8, C, F/8)
    x = x.ascontiguousarray();
    x = x.reshape({shape[0], shape[2], shape[1] * shape[3]});

    return proj_(x); // (batch, T/8, d_model)
}

// ─── FastConformerEncoder ───────────────────────────────────────────────────

FastConformerEncoder::FastConformerEncoder(const EncoderConfig &config)
    : subsampling_(config.subsampling_channels) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back<ConformerBlock>(config);
    }
    AX_REGISTER_MODULES(subsampling_, layers_);
}

Tensor FastConformerEncoder::forward(const Tensor &input,
                                     const Tensor &mask) const {
    auto x = subsampling_(input);

    // Generate sinusoidal position embeddings for the sequence length
    int seq_len = static_cast<int>(x.shape()[1]);
    int d_model = static_cast<int>(x.shape()[2]);
    auto pos_emb = sinusoidal_position_embedding(seq_len, d_model);

    // Match pos_emb device to input
    if (x.device() != pos_emb.device()) {
        pos_emb = pos_emb.to(x.device());
    }

    for (const auto &block : layers_.each<ConformerBlock>()) {
        x = block(x, pos_emb, mask);
    }
    return x;
}

} // namespace parakeet
