#include "../include/parakeet.hpp"

#include <iostream>

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

// ─── ConvSubsampling ────────────────────────────────────────────────────────

ConvSubsampling::ConvSubsampling()
    : depthwise1_(/*stride=*/2, /*padding=*/1),
      depthwise2_(/*stride=*/2, /*padding=*/1),
      depthwise3_(/*stride=*/2, /*padding=*/1), pointwise1_(/*stride=*/1),
      pointwise2_(/*stride=*/1), pointwise3_(/*stride=*/1), proj_(true) {
    AX_REGISTER_MODULES(depthwise1_, pointwise1_, depthwise2_, pointwise2_,
                        depthwise3_, pointwise3_, proj_);
}

Tensor ConvSubsampling::forward(const Tensor &input) const {
    auto x = input.permute({0, 2, 1}); // (batch, mel_bins, mel_length)

    x = ops::silu(pointwise1_(depthwise1_(x)));
    x = ops::silu(pointwise2_(depthwise2_(x)));
    x = ops::silu(pointwise3_(depthwise3_(x)));

    x = x.permute({0, 2, 1}); // (batch, seq, channels)
    return proj_(x);
}

// ─── FastConformerEncoder ───────────────────────────────────────────────────

FastConformerEncoder::FastConformerEncoder() {
    for (int i = 0; i < 24; ++i) {
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

// ─── CTCDecoder ─────────────────────────────────────────────────────────────

CTCDecoder::CTCDecoder() : proj_(true) { AX_REGISTER_MODULE(proj_); }

Tensor CTCDecoder::forward(const Tensor &input) const {
    return ops::log_softmax(proj_(input), /*axis=*/-1);
}

// ─── ParakeetCTC ────────────────────────────────────────────────────────────

ParakeetCTC::ParakeetCTC(const ParakeetConfig &config) : config_(config) {
    AX_REGISTER_MODULES(encoder_, decoder_);
}

Tensor ParakeetCTC::forward(const Tensor &input, const Tensor &mask) const {
    return decoder_(encoder_(input, mask));
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
