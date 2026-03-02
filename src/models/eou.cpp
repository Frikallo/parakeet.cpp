#include "parakeet/models/eou.hpp"

#include <cmath>

namespace parakeet::models {

// ─── ParakeetEOU ─────────────────────────────────────────────────────────────

ParakeetEOU::ParakeetEOU(const EOUConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── Streaming RNNT/TDT Decode ──────────────────────────────────────────────

std::vector<int> rnnt_streaming_decode_chunk(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_chunk,
    const std::vector<int> &durations, StreamingDecodeState &state,
    int blank_id, int max_symbols_per_step) {
    if (!state.initialized) {
        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        state.lstm_states.resize(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            state.lstm_states[l] = {
                Tensor::zeros({1, hs}, encoder_chunk.dtype()),
                Tensor::zeros({1, hs}, encoder_chunk.dtype())};
        }
        state.last_token = Tensor({1}, DType::Int32);
        state.last_token.fill(blank_id);
        state.initialized = true;
    }

    // encoder_chunk: (1, chunk_len, hidden)
    int chunk_len = static_cast<int>(encoder_chunk.shape()[1]);
    std::vector<int> new_tokens;

    int base_frame = state.frame_offset;
    int t = 0;
    while (t < chunk_len) {
        auto enc_t = encoder_chunk.slice({Slice(), Slice(t, t + 1)});

        bool frame_advanced = false;
        for (int sym = 0; sym < max_symbols_per_step; ++sym) {
            auto saved_states = state.lstm_states;

            auto pred = prediction.step(state.last_token, state.lstm_states);
            pred = pred.unsqueeze(1);

            auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

            // Manual argmax on label log-probs for index + confidence
            auto label_1d = label_lp.squeeze(0)
                                .squeeze(0)
                                .cpu()
                                .to_float()
                                .ascontiguousarray();
            const float *label_data = label_1d.typed_data<float>();
            int vocab_size = static_cast<int>(label_1d.shape()[0]);
            int token_id = 0;
            float best_lp = label_data[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (label_data[v] > best_lp) {
                    best_lp = label_data[v];
                    token_id = v;
                }
            }
            float confidence = std::exp(best_lp);

            auto best_dur = ops::argmax(dur_lp.squeeze(0).squeeze(0), -1);
            int dur_idx = best_dur.item<int>();
            int skip = (dur_idx < static_cast<int>(durations.size()))
                           ? durations[dur_idx]
                           : 1;

            if (token_id == blank_id) {
                state.lstm_states = saved_states;
                t += std::max(skip, 1);
                frame_advanced = true;
                break;
            }

            new_tokens.push_back(token_id);
            state.tokens.push_back(token_id);

            int abs_frame = base_frame + t;
            int end_frame = abs_frame + std::max(skip, 1) - 1;
            state.timestamped_tokens.push_back(
                {token_id, abs_frame, end_frame, confidence});

            state.last_token = Tensor({1}, DType::Int32);
            state.last_token.fill(token_id);

            if (skip > 0) {
                t += skip;
                frame_advanced = true;
                break;
            }
        }

        // Force advance when max_symbols_per_step reached without
        // a blank or positive-duration token.
        if (!frame_advanced) {
            t += 1;
        }
    }

    state.frame_offset += chunk_len;
    return new_tokens;
}

} // namespace parakeet::models
