#include "parakeet/streaming_encoder.hpp"

#include <cmath>

namespace parakeet {

// ─── CausalConformerConvModule ───────────────────────────────────────────────

CausalConformerConvModule::CausalConformerConvModule(int groups,
                                                     int kernel_size,
                                                     float dropout)
    : kernel_size_(kernel_size), pointwise_conv1_(/*stride=*/1),
      depthwise_conv_(/*stride=*/1, /*padding=*/0, /*dilation=*/1,
                      /*groups=*/groups),
      pointwise_conv2_(/*stride=*/1), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, pointwise_conv1_, depthwise_conv_, batch_norm_,
                        pointwise_conv2_, dropout_);
}

Tensor CausalConformerConvModule::forward(const Tensor &input) const {
    auto x = norm_(input);
    x = x.permute({0, 2, 1});

    x = pointwise_conv1_(x);
    x = ops::glu(x, /*dim=*/1);

    // Full-sequence: use symmetric padding (same as non-streaming)
    int pad = (kernel_size_ - 1) / 2;
    x = ops::pad(x, {{0, 0}, {0, 0}, {pad, pad}});
    x = depthwise_conv_(x);
    x = batch_norm_(x);
    x = ops::silu(x);

    x = pointwise_conv2_(x);
    x = dropout_(x);
    x = x.permute({0, 2, 1});

    return input + x;
}

Tensor CausalConformerConvModule::forward_cached(const Tensor &input,
                                                 Tensor &conv_cache) const {
    auto x = norm_(input);
    x = x.permute({0, 2, 1}); // (batch, hidden, seq)

    x = pointwise_conv1_(x);
    x = ops::glu(x, /*dim=*/1);

    // Causal: prepend cache (left-only padding)
    int cache_len = kernel_size_ - 1;
    if (conv_cache.storage()) {
        x = Tensor::cat({conv_cache, x}, /*dim=*/2);
    } else {
        // First chunk: zero-pad left
        auto shape = x.shape();
        auto pad_tensor =
            Tensor::zeros({shape[0], shape[1], static_cast<size_t>(cache_len)});
        if (x.device() != axiom::Device::CPU)
            pad_tensor = pad_tensor.to(x.device());
        x = Tensor::cat({pad_tensor, x}, 2);
    }

    // Update cache: last (kernel_size - 1) frames
    auto x_shape = x.shape();
    int total = static_cast<int>(x_shape[2]);
    conv_cache = x.slice({Slice(), Slice(), Slice(total - cache_len, total)})
                     .ascontiguousarray();

    x = depthwise_conv_(x);
    x = batch_norm_(x);
    x = ops::silu(x);

    x = pointwise_conv2_(x);
    x = dropout_(x);
    x = x.permute({0, 2, 1});

    return input + x;
}

// ─── StreamingConformerAttention ─────────────────────────────────────────────

StreamingConformerAttention::StreamingConformerAttention(int num_heads,
                                                         float dropout)
    : mha_(num_heads), pos_proj_(false), dropout_(dropout) {
    AX_REGISTER_MODULES(norm_, mha_, pos_proj_, dropout_);
    AX_REGISTER_PARAMETERS(pos_bias_u_, pos_bias_v_);
}

Tensor StreamingConformerAttention::rel_shift(const Tensor &x) {
    auto shape = x.shape();
    size_t batch = shape[0];
    size_t heads = shape[1];
    size_t seq_len = shape[2];
    size_t pos_len = shape[3];

    auto padded = ops::pad(x, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
    padded = padded.reshape({batch, heads, pos_len + 1, seq_len});
    padded = padded.slice({Slice(), Slice(), Slice(1), Slice()});
    padded = padded.reshape({batch, heads, seq_len, pos_len});
    return padded.slice(
        {Slice(), Slice(), Slice(), Slice(0, static_cast<int64_t>(seq_len))});
}

Tensor StreamingConformerAttention::rel_position_attention(
    const Tensor &query, const Tensor &key, const Tensor &value,
    const Tensor &pos_emb, const Tensor &mask) const {
    int num_heads = mha_.num_heads();
    auto q = mha_.q_proj()(query);
    auto k = mha_.k_proj()(key);
    auto v = mha_.v_proj()(value);

    auto d_model = static_cast<int>(q.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = q.shape()[0];
    auto seq_len = q.shape()[1];
    auto kv_len = k.shape()[1];

    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);
    q = q.reshape({batch, seq_len, nh, hd}).transpose({0, 2, 1, 3});
    k = k.reshape({batch, kv_len, nh, hd}).transpose({0, 2, 1, 3});
    v = v.reshape({batch, kv_len, nh, hd}).transpose({0, 2, 1, 3});

    auto bias_u = pos_bias_u_.reshape({1, nh, 1, hd});
    auto bias_v = pos_bias_v_.reshape({1, nh, 1, hd});

    auto content_score = ops::matmul(q + bias_u, k, false, true);

    auto p = pos_proj_(pos_emb);
    auto pos_len = p.shape()[0];
    p = p.reshape({1, pos_len, nh, hd}).transpose({0, 2, 1, 3});

    auto pos_score = ops::matmul(q + bias_v, p, false, true);
    pos_score = rel_shift(pos_score);

    auto scores = (content_score + pos_score) * scale;

    if (mask.storage()) {
        scores = ops::masked_fill(scores, mask, -1e9f);
    }

    auto attn_weights = ops::softmax(scores, -1);
    auto out = ops::matmul(attn_weights, v);

    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, seq_len, static_cast<size_t>(d_model)});

    return mha_.out_proj()(out);
}

Tensor StreamingConformerAttention::forward(const Tensor &input,
                                            const Tensor &pos_emb,
                                            const Tensor &mask) const {
    auto x = norm_(input);
    x = rel_position_attention(x, x, x, pos_emb, mask);
    x = dropout_(x);
    return input + x;
}

Tensor StreamingConformerAttention::forward_cached(
    const Tensor &input, const Tensor &pos_emb, Tensor &key_cache,
    Tensor &value_cache, int att_context_left, int att_context_right) const {
    auto x = norm_(input);

    int num_heads = mha_.num_heads();
    auto q = mha_.q_proj()(x);
    auto k = mha_.k_proj()(x);
    auto v = mha_.v_proj()(x);

    auto d_model = static_cast<int>(q.shape().back());
    int head_dim = d_model / num_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto batch = q.shape()[0];
    auto chunk_len = q.shape()[1];
    auto nh = static_cast<size_t>(num_heads);
    auto hd = static_cast<size_t>(head_dim);

    q = q.reshape({batch, chunk_len, nh, hd}).transpose({0, 2, 1, 3});
    k = k.reshape({batch, chunk_len, nh, hd}).transpose({0, 2, 1, 3});
    v = v.reshape({batch, chunk_len, nh, hd}).transpose({0, 2, 1, 3});

    // Prepend cached K/V
    if (key_cache.storage()) {
        k = Tensor::cat({key_cache, k}, /*dim=*/2);
        v = Tensor::cat({value_cache, v}, /*dim=*/2);
    }

    auto kv_len = k.shape()[2];

    // Update cache: keep last att_context_left frames
    int max_cache = att_context_left;
    if (static_cast<int>(kv_len) > max_cache) {
        int start = static_cast<int>(kv_len) - max_cache;
        key_cache =
            k.slice({Slice(), Slice(),
                     Slice(start, static_cast<int64_t>(kv_len)), Slice()})
                .ascontiguousarray();
        value_cache =
            v.slice({Slice(), Slice(),
                     Slice(start, static_cast<int64_t>(kv_len)), Slice()})
                .ascontiguousarray();
    } else {
        key_cache = k.ascontiguousarray();
        value_cache = v.ascontiguousarray();
    }

    // Position bias
    auto bias_u = pos_bias_u_.reshape({1, nh, 1, hd});
    auto bias_v = pos_bias_v_.reshape({1, nh, 1, hd});

    // Content attention
    auto content_score = ops::matmul(q + bias_u, k, false, true);

    // Position attention
    auto p = pos_proj_(pos_emb);
    auto pos_len = p.shape()[0];
    p = p.reshape({1, pos_len, nh, hd}).transpose({0, 2, 1, 3});
    auto pos_score = ops::matmul(q + bias_v, p, false, true);

    // For cached attention, we need to handle the position scores differently
    // Slice to match kv_len if needed
    if (pos_score.shape()[3] > kv_len) {
        // Take the rightmost kv_len columns
        int start =
            static_cast<int>(pos_score.shape()[3]) - static_cast<int>(kv_len);
        pos_score = pos_score.slice(
            {Slice(), Slice(), Slice(),
             Slice(start, static_cast<int64_t>(pos_score.shape()[3]))});
    }

    auto scores = (content_score + pos_score) * scale;

    // Build attention mask for bounded context
    // Query positions attend to KV positions within [q_pos - left, q_pos +
    // right] relative to the KV sequence
    if (att_context_left >= 0 || att_context_right >= 0) {
        int q_len = static_cast<int>(chunk_len);
        int kv = static_cast<int>(kv_len);
        // q positions are at the end of the kv sequence
        // q[i] is at absolute position (kv - q_len + i) in the kv sequence
        std::vector<float> mask_data(q_len * kv, 0.0f);
        for (int qi = 0; qi < q_len; ++qi) {
            int abs_pos = kv - q_len + qi;
            for (int ki = 0; ki < kv; ++ki) {
                int dist = abs_pos - ki;
                if (dist > att_context_left || -dist > att_context_right) {
                    mask_data[qi * kv + ki] = 1.0f;
                }
            }
        }
        auto attn_mask = Tensor::from_data(
            mask_data.data(),
            Shape{1, 1, static_cast<size_t>(q_len), static_cast<size_t>(kv)},
            true);
        if (scores.device() != axiom::Device::CPU)
            attn_mask = attn_mask.to(scores.device());
        scores = ops::masked_fill(scores, attn_mask, -1e9f);
    }

    auto attn_weights = ops::softmax(scores, -1);
    auto out = ops::matmul(attn_weights, v);

    out = out.transpose({0, 2, 1, 3});
    out = out.reshape({batch, chunk_len, static_cast<size_t>(d_model)});

    x = mha_.out_proj()(out);
    x = dropout_(x);
    return input + x;
}

// ─── StreamingConformerBlock ─────────────────────────────────────────────────

StreamingConformerBlock::StreamingConformerBlock(
    const StreamingEncoderConfig &config)
    : config_(config), ffn1_(config.dropout),
      attn_(config.num_heads, config.dropout),
      conv_(config.hidden_size, config.conv_kernel_size, config.dropout),
      ffn2_(config.dropout) {
    AX_REGISTER_MODULES(ffn1_, attn_, conv_, ffn2_, final_norm_);
}

Tensor StreamingConformerBlock::forward(const Tensor &input,
                                        const Tensor &pos_emb,
                                        const Tensor &mask) const {
    auto x = ffn1_(input);
    x = attn_(x, pos_emb, mask);
    x = conv_(x);
    x = ffn2_(x);
    x = final_norm_(x);
    return x;
}

Tensor StreamingConformerBlock::forward_cached(const Tensor &input,
                                               const Tensor &pos_emb,
                                               BlockCache &cache,
                                               int att_context_left,
                                               int att_context_right) const {
    auto x = ffn1_(input);
    x = attn_.forward_cached(x, pos_emb, cache.key_cache, cache.value_cache,
                             att_context_left, att_context_right);
    x = conv_.forward_cached(x, cache.conv_cache);
    x = ffn2_(x);
    x = final_norm_(x);
    return x;
}

// ─── CausalConvSubsampling ──────────────────────────────────────────────────

CausalConvSubsampling::CausalConvSubsampling(int channels,
                                             SubsamplingActivation act)
    : conv1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      dw1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      dw2_(/*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      conv2_(/*stride=*/{1, 1}, /*padding=*/{0, 0}),
      conv3_(/*stride=*/{1, 1}, /*padding=*/{0, 0}), proj_(true),
      activation_(act) {
    AX_REGISTER_MODULES(conv1_, dw1_, conv2_, dw2_, conv3_, proj_);
}

Tensor CausalConvSubsampling::forward(const Tensor &input) const {
    auto act = [this](const Tensor &t) {
        return activation_ == SubsamplingActivation::ReLU ? ops::relu(t)
                                                          : ops::silu(t);
    };
    auto x = input.unsqueeze(1);
    x = conv1_(x);
    x = act(x);
    x = dw1_(x);
    x = conv2_(x);
    x = act(x);
    x = dw2_(x);
    x = conv3_(x);
    x = act(x);

    auto shape = x.shape();
    x = x.permute({0, 2, 1, 3});
    x = x.ascontiguousarray();
    x = x.reshape({shape[0], shape[2], shape[1] * shape[3]});

    return proj_(x);
}

Tensor CausalConvSubsampling::forward_cached(const Tensor &input,
                                             Tensor &cache) const {
    // For streaming: concatenate leftover frames from previous chunk
    Tensor mel_input;
    if (cache.storage()) {
        mel_input = Tensor::cat({cache, input}, /*dim=*/1);
    } else {
        mel_input = input;
    }

    // Subsampling factor is 8. Keep leftover frames that don't fill a full
    // stride.
    int total_frames = static_cast<int>(mel_input.shape()[1]);
    // After 3 stride-2 convs: output_len = floor((total + 2*pad - kernel) /
    // stride + 1) The exact number of consumed frames depends on the conv
    // architecture. For simplicity, we process what we can and cache the
    // remainder.
    int consumable = (total_frames / 8) * 8;
    if (consumable == 0) {
        // Not enough frames yet — cache everything
        cache = mel_input.ascontiguousarray();
        return Tensor(); // empty
    }

    // Cache leftover
    int leftover = total_frames - consumable;
    if (leftover > 0) {
        cache =
            mel_input.slice({Slice(), Slice(consumable, total_frames), Slice()})
                .ascontiguousarray();
    } else {
        cache = Tensor();
    }

    // Process consumable portion
    auto to_process = mel_input.slice({Slice(), Slice(0, consumable), Slice()});
    return forward(to_process);
}

// ─── StreamingFastConformerEncoder ──────────────────────────────────────────

StreamingFastConformerEncoder::StreamingFastConformerEncoder(
    const StreamingEncoderConfig &config)
    : config_(config),
      subsampling_(config.subsampling_channels, config.subsampling_activation) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back<StreamingConformerBlock>(config);
    }
    AX_REGISTER_MODULES(subsampling_, layers_);
}

Tensor StreamingFastConformerEncoder::forward(const Tensor &input,
                                              const Tensor &mask) const {
    auto x = subsampling_(input);

    // xscaling: multiply by sqrt(d_model) before conformer layers
    if (config_.xscaling) {
        float scale = std::sqrt(static_cast<float>(config_.hidden_size));
        x = x * scale;
    }

    int seq_len = static_cast<int>(x.shape()[1]);
    int d_model = static_cast<int>(x.shape()[2]);
    auto pos_emb = sinusoidal_position_embedding(seq_len, d_model);

    if (x.device() != pos_emb.device()) {
        pos_emb = pos_emb.to(x.device());
    }

    for (const auto &block : layers_.each<StreamingConformerBlock>()) {
        x = block(x, pos_emb, mask);
    }
    return x;
}

void StreamingFastConformerEncoder::init_cache(EncoderCache &cache,
                                               size_t batch_size) const {
    cache.layer_caches.resize(config_.num_layers);
    cache.frames_seen = 0;
    // Individual layer caches start empty (no storage)
}

Tensor StreamingFastConformerEncoder::forward_chunk(const Tensor &input,
                                                    EncoderCache &cache) const {
    if (cache.empty()) {
        init_cache(cache, input.shape()[0]);
    }

    // Subsampling with cache
    auto x = subsampling_.forward_cached(input, cache.subsampling_cache);

    // If subsampling returned empty (not enough frames), return empty
    if (!x.storage() || x.shape().size() == 0) {
        return Tensor();
    }

    // xscaling: multiply by sqrt(d_model) before conformer layers
    if (config_.xscaling) {
        float scale = std::sqrt(static_cast<float>(config_.hidden_size));
        x = x * scale;
    }

    int chunk_len = static_cast<int>(x.shape()[1]);
    int d_model = static_cast<int>(x.shape()[2]);

    // Generate position embeddings for this chunk
    // For streaming, we need positions relative to the current chunk + cache
    int total_context = config_.att_context_left + chunk_len;
    auto pos_emb = sinusoidal_position_embedding(total_context, d_model);
    if (x.device() != pos_emb.device()) {
        pos_emb = pos_emb.to(x.device());
    }

    // Process through each layer
    int layer_idx = 0;
    for (auto &block : layers_.each<StreamingConformerBlock>()) {
        x = const_cast<StreamingConformerBlock &>(block).forward_cached(
            x, pos_emb, cache.layer_caches[layer_idx], config_.att_context_left,
            config_.att_context_right);
        ++layer_idx;
    }

    cache.frames_seen += chunk_len;
    return x;
}

} // namespace parakeet
