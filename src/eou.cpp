#include "parakeet/eou.hpp"

namespace parakeet {

// ─── ParakeetEOU ─────────────────────────────────────────────────────────────

ParakeetEOU::ParakeetEOU(const EOUConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── Streaming RNNT/TDT Decode ──────────────────────────────────────────────

std::vector<int>
rnnt_streaming_decode_chunk(RNNTPrediction &prediction, TDTJoint &joint,
                            const Tensor &encoder_chunk,
                            const std::vector<int> &durations,
                            StreamingDecodeState &state, int blank_id,
                            int max_symbols_per_step) {
    if (!state.initialized) {
        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        state.lstm_states.resize(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            state.lstm_states[l] = {Tensor::zeros({1, hs}),
                                    Tensor::zeros({1, hs})};
        }
        state.last_token = Tensor::zeros({1}, DType::Int32);
        state.initialized = true;
    }

    // encoder_chunk: (1, chunk_len, hidden)
    int chunk_len = static_cast<int>(encoder_chunk.shape()[1]);
    std::vector<int> new_tokens;

    int t = 0;
    while (t < chunk_len) {
        auto enc_t = encoder_chunk.slice({Slice(), Slice(t, t + 1)});

        for (int sym = 0; sym < max_symbols_per_step; ++sym) {
            auto pred = prediction.step(state.last_token, state.lstm_states);
            pred = pred.unsqueeze(1);

            auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

            auto best_label =
                ops::argmax(label_lp.squeeze(0).squeeze(0), -1);
            auto best_dur = ops::argmax(dur_lp.squeeze(0).squeeze(0), -1);

            int token_id = best_label.item<int>();
            int dur_idx = best_dur.item<int>();
            int skip = (dur_idx < static_cast<int>(durations.size()))
                           ? durations[dur_idx]
                           : 1;

            if (token_id == blank_id) {
                t += std::max(skip, 1);
                break;
            }

            new_tokens.push_back(token_id);
            state.tokens.push_back(token_id);
            state.last_token = Tensor({1}, DType::Int32);
            state.last_token.fill(token_id);

            if (skip > 0) {
                t += skip;
                break;
            }
        }
    }

    return new_tokens;
}

// ─── StreamingTranscriber ────────────────────────────────────────────────────

StreamingTranscriber::StreamingTranscriber(const std::string &weights_path,
                                           const std::string &vocab_path,
                                           const EOUConfig &config)
    : config_(config), model_(config) {
    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
    tokenizer_.load(vocab_path);
}

std::string StreamingTranscriber::transcribe_chunk(const Tensor &samples) {
    // 1. Preprocess chunk
    auto features = preprocessor_.process_chunk(samples);
    if (!features.storage()) {
        return ""; // not enough samples yet
    }

    if (use_gpu_) {
        features = features.gpu();
    }

    // 2. Encode chunk
    auto encoder_out = model_.encoder().forward_chunk(features, encoder_cache_);
    if (!encoder_out.storage()) {
        return ""; // not enough frames for subsampling
    }

    // 3. Decode chunk
    auto new_tokens = rnnt_streaming_decode_chunk(
        model_.prediction(), model_.joint(), encoder_out, config_.durations,
        decode_state_);

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

} // namespace parakeet
