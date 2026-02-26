#include "parakeet/nemotron.hpp"

namespace parakeet {

// ─── ParakeetNemotron ────────────────────────────────────────────────────────

ParakeetNemotron::ParakeetNemotron(const NemotronConfig &config)
    : config_(config), encoder_(config.encoder),
      prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── NemotronTranscriber ─────────────────────────────────────────────────────

NemotronTranscriber::NemotronTranscriber(const std::string &weights_path,
                                         const std::string &vocab_path,
                                         const NemotronConfig &config)
    : config_(config), model_(config) {
    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string NemotronTranscriber::transcribe_chunk(const Tensor &samples) {
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return "";
    }

    if (use_gpu_) {
        features = features.gpu();
    }

    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return "";
    }

    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_);

    if (!new_tokens.empty() && tokenizer_.loaded()) {
        auto text = tokenizer_.decode(new_tokens);
        if (partial_callback_) {
            partial_callback_(text);
        }
        return text;
    }

    return "";
}

void NemotronTranscriber::reset() {
    preprocessor_.reset();
    encoder_cache_ = EncoderCache{};
    decode_state_ = StreamingDecodeState{};
}

std::string NemotronTranscriber::get_text() const {
    if (tokenizer_.loaded() && !decode_state_.tokens.empty()) {
        return tokenizer_.decode(decode_state_.tokens);
    }
    return "";
}

} // namespace parakeet
