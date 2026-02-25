#include "../include/parakeet.hpp"

#include <iostream>

namespace parakeet {

// ─── BatchNorm1d ────────────────────────────────────────────────────────────

BatchNorm1d::BatchNorm1d(float eps) : eps_(eps) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
    register_parameter("running_mean", running_mean_);
    register_parameter("running_var", running_var_);
}

Tensor BatchNorm1d::forward(const Tensor &input) const {
    // input: (batch, channels, length)
    // Normalize over batch and length dimensions using running stats
    auto mean = running_mean_.unsqueeze(0).unsqueeze(-1); // (1, C, 1)
    auto var = running_var_.unsqueeze(0).unsqueeze(-1);   // (1, C, 1)
    auto w = weight_.unsqueeze(0).unsqueeze(-1);          // (1, C, 1)
    auto b = bias_.unsqueeze(0).unsqueeze(-1);            // (1, C, 1)

    auto normalized = (input - mean) / (var + eps_).sqrt();
    return normalized * w + b;
}

// ─── FeedForward ────────────────────────────────────────────────────────────

FeedForward::FeedForward() : fc1_(true), fc2_(true) {
    register_module("norm", norm_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}

Tensor FeedForward::forward(const Tensor &input) const {
    // Pre-norm → Linear → SiLU → Linear → half-step residual
    auto x = norm_(input);
    x = fc1_(x);
    x = ops::silu(x);
    x = fc2_(x);
    return input + x * 0.5f; // macaron half-step
}

// ─── ConformerConvModule ────────────────────────────────────────────────────

ConformerConvModule::ConformerConvModule()
    : pointwise_conv1_(/*stride=*/1, /*padding=*/0, /*dilation=*/1,
                       /*groups=*/1, /*bias=*/true),
      depthwise_conv_(/*stride=*/1, /*padding=*/4, /*dilation=*/1,
                      /*groups=*/0, /*bias=*/true), // groups set by weights
      pointwise_conv2_(/*stride=*/1, /*padding=*/0, /*dilation=*/1,
                       /*groups=*/1, /*bias=*/true) {
    register_module("norm", norm_);
    register_module("pointwise_conv1", pointwise_conv1_);
    register_module("depthwise_conv", depthwise_conv_);
    register_module("batch_norm", batch_norm_);
    register_module("pointwise_conv2", pointwise_conv2_);
}

Tensor ConformerConvModule::forward(const Tensor &input) const {
    // input: (batch, seq, hidden_size)
    auto x = norm_(input);

    // Transpose to (batch, hidden_size, seq) for conv1d
    x = x.transpose({0, 2, 1});

    // Pointwise conv → GLU gating
    x = pointwise_conv1_(x); // (batch, 2*hidden_size, seq)

    // GLU: split in half along channel dim, gate with sigmoid
    auto chunks_size = x.shape()[1] / 2;
    auto gate = x.index_select(1, Tensor::arange(0, chunks_size));
    auto value = x.index_select(1, Tensor::arange(chunks_size, x.shape()[1]));
    x = value * ops::sigmoid(gate);

    // Depthwise conv → BatchNorm → SiLU
    x = depthwise_conv_(x);
    x = batch_norm_(x);
    x = ops::silu(x);

    // Pointwise conv back to hidden_size
    x = pointwise_conv2_(x);

    // Transpose back to (batch, seq, hidden_size)
    x = x.transpose({0, 2, 1});

    return input + x; // residual
}

// ─── ConformerAttention ─────────────────────────────────────────────────────

ConformerAttention::ConformerAttention()
    : mha_(8), pos_proj_(false) { // num_heads set, no bias on pos_proj
    register_module("norm", norm_);
    register_module("mha", mha_);
    register_module("pos_proj", pos_proj_);
}

Tensor ConformerAttention::forward(const Tensor &input,
                                   const Tensor &mask) const {
    // Pre-norm → self-attention → residual
    auto x = norm_(input);
    x = mha_.forward(x, x, x, mask);
    return input + x;
}

// ─── ConformerBlock ─────────────────────────────────────────────────────────

ConformerBlock::ConformerBlock() {
    register_module("ffn1", ffn1_);
    register_module("attn", attn_);
    register_module("conv", conv_);
    register_module("ffn2", ffn2_);
    register_module("final_norm", final_norm_);
}

Tensor ConformerBlock::forward(const Tensor &input,
                               const Tensor &mask) const {
    // Macaron-style: FFN → MHSA → Conv → FFN → LayerNorm
    auto x = ffn1_(input);
    x = attn_(x, mask);
    x = conv_(x);
    x = ffn2_(x);
    x = final_norm_(x);
    return x;
}

// ─── ConvSubsampling ────────────────────────────────────────────────────────

ConvSubsampling::ConvSubsampling()
    : depthwise1_(/*stride=*/2, /*padding=*/1, /*dilation=*/1, /*groups=*/0,
                  /*bias=*/true),
      pointwise1_(/*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                  /*bias=*/true),
      depthwise2_(/*stride=*/2, /*padding=*/1, /*dilation=*/1, /*groups=*/0,
                  /*bias=*/true),
      pointwise2_(/*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                  /*bias=*/true),
      depthwise3_(/*stride=*/2, /*padding=*/1, /*dilation=*/1, /*groups=*/0,
                  /*bias=*/true),
      pointwise3_(/*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/1,
                  /*bias=*/true),
      proj_(true) {
    register_module("depthwise1", depthwise1_);
    register_module("pointwise1", pointwise1_);
    register_module("depthwise2", depthwise2_);
    register_module("pointwise2", pointwise2_);
    register_module("depthwise3", depthwise3_);
    register_module("pointwise3", pointwise3_);
    register_module("proj", proj_);
}

Tensor ConvSubsampling::forward(const Tensor &input) const {
    // input: (batch, mel_length, mel_bins)
    // Transpose to (batch, mel_bins, mel_length) for conv1d
    auto x = input.transpose({0, 2, 1});

    // Stage 1: depthwise-separable conv, stride 2
    x = ops::silu(pointwise1_(depthwise1_(x)));

    // Stage 2: depthwise-separable conv, stride 2
    x = ops::silu(pointwise2_(depthwise2_(x)));

    // Stage 3: depthwise-separable conv, stride 2
    x = ops::silu(pointwise3_(depthwise3_(x)));

    // Transpose back to (batch, seq, channels) and project to hidden_size
    x = x.transpose({0, 2, 1});
    x = proj_(x);
    return x;
}

// ─── FastConformerEncoder ───────────────────────────────────────────────────

FastConformerEncoder::FastConformerEncoder() {
    // Build 24 conformer blocks (populated by load_state_dict)
    for (int i = 0; i < 24; ++i) {
        layers_.emplace_back<ConformerBlock>();
    }
    register_module("subsampling", subsampling_);
    register_module("layers", layers_);
}

Tensor FastConformerEncoder::forward(const Tensor &input,
                                     const Tensor &mask) const {
    // Subsample input
    auto x = subsampling_(input);

    // Pass through all conformer blocks
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto &block = static_cast<const ConformerBlock &>(layers_[i]);
        x = block(x, mask);
    }
    return x;
}

// ─── CTCDecoder ─────────────────────────────────────────────────────────────

CTCDecoder::CTCDecoder() : proj_(true) {
    register_module("proj", proj_);
}

Tensor CTCDecoder::forward(const Tensor &input) const {
    // Linear projection → log softmax over vocab dimension
    auto logits = proj_(input);
    return ops::log_softmax(logits, /*axis=*/-1);
}

// ─── ParakeetCTC ────────────────────────────────────────────────────────────

ParakeetCTC::ParakeetCTC(const ParakeetConfig &config) : config_(config) {
    register_module("encoder", encoder_);
    register_module("decoder", decoder_);
}

Tensor ParakeetCTC::forward(const Tensor &input, const Tensor &mask) const {
    // input: (batch, mel_length, mel_bins)
    auto encoded = encoder_(input, mask);
    // encoded: (batch, seq_length, hidden_size)
    auto log_probs = decoder_(encoded);
    // log_probs: (batch, seq_length, vocab_size)
    return log_probs;
}

} // namespace parakeet

int main() {
    using namespace parakeet;

    ParakeetConfig config;
    ParakeetCTC model(config);

    std::cout << "Parakeet CTC model created" << std::endl;
    std::cout << "  Encoder: FastConformer" << std::endl;
    std::cout << "  Layers: " << config.num_layers << std::endl;
    std::cout << "  Hidden size: " << config.hidden_size << std::endl;
    std::cout << "  Attention heads: " << config.num_heads << std::endl;
    std::cout << "  FFN intermediate: " << config.ffn_intermediate << std::endl;
    std::cout << "  Conv kernel: " << config.conv_kernel_size << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;

    return 0;
}
