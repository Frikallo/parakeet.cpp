#include "parakeet/api/transcribe.hpp"

#include <axiom/io/safetensors.hpp>

namespace parakeet::api {

// ─── StreamingTranscriber ────────────────────────────────────────────────────

StreamingTranscriber::StreamingTranscriber(const std::string &weights_path,
                                           const std::string &vocab_path,
                                           const EOUConfig &config)
    : config_(config), model_(config) {
    // Configure preprocessor with correct mel bins from encoder config
    AudioConfig audio_cfg;
    audio_cfg.n_mels = config.encoder.mel_bins;
    preprocessor_ = StreamingAudioPreprocessor(audio_cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string
StreamingTranscriber::transcribe_chunk(const axiom::Tensor &samples) {
    // 1. Preprocess chunk
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return ""; // not enough samples yet
    }

    if (use_fp16_)
        features = features.half();
    if (use_gpu_) {
        features = features.gpu();
    }

    // 2. Encode chunk
    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return ""; // not enough frames for subsampling
    }

    // 3. Decode chunk (blank = last token in vocab)
    int blank_id = config_.joint.vocab_size - 1;
    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_, blank_id);

    // 4. Convert new tokens to text
    if (!new_tokens.empty() && tokenizer_.loaded()) {
        auto text = tokenizer_.decode(new_tokens);
        if (partial_callback_) {
            partial_callback_(text);
        }
        return text;
    }

    return "";
}

void StreamingTranscriber::reset() {
    preprocessor_.reset();
    encoder_cache_ = EncoderCache{};
    decode_state_ = StreamingDecodeState{};
}

std::string StreamingTranscriber::get_text() const {
    if (tokenizer_.loaded() && !decode_state_.tokens.empty()) {
        return tokenizer_.decode(decode_state_.tokens);
    }
    return "";
}

// ─── NemotronTranscriber ─────────────────────────────────────────────────────

NemotronTranscriber::NemotronTranscriber(const std::string &weights_path,
                                         const std::string &vocab_path,
                                         const NemotronConfig &config)
    : config_(config), model_(config) {
    // Configure preprocessor with correct mel bins from encoder config
    AudioConfig audio_cfg;
    audio_cfg.n_mels = config.encoder.mel_bins;
    preprocessor_ = StreamingAudioPreprocessor(audio_cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string
NemotronTranscriber::transcribe_chunk(const axiom::Tensor &samples) {
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return "";
    }

    if (use_fp16_)
        features = features.half();
    if (use_gpu_) {
        features = features.gpu();
    }

    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return "";
    }

    int blank_id = config_.joint.vocab_size - 1;
    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_, blank_id);

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

} // namespace parakeet::api
