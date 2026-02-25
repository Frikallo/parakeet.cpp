#include "parakeet/encoder.hpp"

namespace parakeet {

// ─── FeedForward ────────────────────────────────────────────────────────────

FeedForward::FeedForward() : fc1_(true), fc2_(true), dropout_(0.1f) {
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

ConformerConvModule::ConformerConvModule()
    : pointwise_conv1_(/*stride=*/1),
      depthwise_conv_(/*stride=*/1, /*padding=*/4),
      pointwise_conv2_(/*stride=*/1), dropout_(0.1f) {
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

ConformerAttention::ConformerAttention()
    : mha_(8), pos_proj_(false), dropout_(0.1f) {
    AX_REGISTER_MODULES(norm_, mha_, pos_proj_, dropout_);
}

Tensor ConformerAttention::forward(const Tensor &input,
                                   const Tensor &mask) const {
    auto x = norm_(input);
    x = mha_.forward(x, x, x, mask);
    x = dropout_(x);
    return input + x;
}

// ─── ConformerBlock ─────────────────────────────────────────────────────────

ConformerBlock::ConformerBlock() {
    AX_REGISTER_MODULES(ffn1_, attn_, conv_, ffn2_, final_norm_);
}

Tensor ConformerBlock::forward(const Tensor &input, const Tensor &mask) const {
    auto x = ffn1_(input);
    x = attn_(x, mask);
    x = conv_(x);
    x = ffn2_(x);
    x = final_norm_(x);
    return x;
}

// ─── ConvSubsampling (Conv2d) ────────────────────────────────────────────────

ConvSubsampling::ConvSubsampling(int channels)
    : conv1_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      conv2_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      conv3_(/*stride=*/{2, 2}, /*padding=*/{1, 1}),
      dw1_(/*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      dw2_(/*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      dw3_(/*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
           /*groups=*/channels),
      proj_(true) {
    AX_REGISTER_MODULES(conv1_, dw1_, conv2_, dw2_, conv3_, dw3_, proj_);
}

Tensor ConvSubsampling::forward(const Tensor &input) const {
    // input: (batch, mel_length, mel_bins)
    auto x = input.unsqueeze(1); // (batch, 1, mel_length, mel_bins)

    x = ops::silu(conv1_(x)); // (batch, C, T/2, F/2)
    x = dw1_(x);
    x = ops::silu(conv2_(x)); // (batch, C, T/4, F/4)
    x = dw2_(x);
    x = ops::silu(conv3_(x)); // (batch, C, T/8, F/8)
    x = dw3_(x);

    // Flatten channels and freq: (batch, C, T/8, F/8) → (batch, T/8, C*F/8)
    auto shape = x.shape();
    x = x.permute({0, 2, 1, 3}); // (batch, T/8, C, F/8)
    x = x.reshape({shape[0], shape[2], shape[1] * shape[3]});

    return proj_(x); // (batch, T/8, d_model)
}

// ─── FastConformerEncoder ───────────────────────────────────────────────────

FastConformerEncoder::FastConformerEncoder(const EncoderConfig &config)
    : subsampling_(config.subsampling_channels) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back<ConformerBlock>();
    }
    AX_REGISTER_MODULES(subsampling_, layers_);
}

Tensor FastConformerEncoder::forward(const Tensor &input,
                                     const Tensor &mask) const {
    auto x = subsampling_(input);
    for (const auto &block : layers_.each<ConformerBlock>()) {
        x = block(x, mask);
    }
    return x;
}

} // namespace parakeet
