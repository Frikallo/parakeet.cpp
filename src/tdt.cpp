#include "parakeet/tdt.hpp"

namespace parakeet {

// ─── TDTJoint ───────────────────────────────────────────────────────────────

TDTJoint::TDTJoint(const JointConfig &config, int num_durations)
    : config_(config), num_durations_(num_durations), enc_proj_(true),
      pred_proj_(true), label_proj_(true), duration_proj_(true) {
    AX_REGISTER_MODULES(enc_proj_, pred_proj_, label_proj_, duration_proj_);
}

TDTJoint::Output TDTJoint::forward(const Tensor &encoder_out,
                                   const Tensor &prediction_out) const {
    auto hidden = enc_proj_(encoder_out) + pred_proj_(prediction_out);
    hidden = ops::relu(hidden);

    auto label_logits = ops::log_softmax(label_proj_(hidden), /*axis=*/-1);
    auto dur_logits = ops::log_softmax(duration_proj_(hidden), /*axis=*/-1);

    return {label_logits, dur_logits};
}

// ─── ParakeetTDT ────────────────────────────────────────────────────────────

ParakeetTDT::ParakeetTDT(const TDTConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── TDT Greedy Decode ─────────────────────────────────────────────────────

std::vector<std::vector<int>>
tdt_greedy_decode(RNNTPrediction &prediction, TDTJoint &joint,
                  const Tensor &encoder_out, const std::vector<int> &durations,
                  int blank_id, int max_symbols_per_step) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)}); // (1, seq, hidden)

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}), Tensor::zeros({1, hs})};
        }

        auto token = Tensor::zeros({1}, DType::Int32);
        int t = 0;

        while (t < seq_len) {
            auto enc_t =
                enc.slice({Slice(), Slice(t, t + 1)}); // (1, 1, hidden)

            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto pred = prediction.step(token, states);
                pred = pred.unsqueeze(1); // (1, 1, pred_hidden)

                auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

                // Get best label and duration
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

                results[b].push_back(token_id);
                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                // Non-blank with duration > 0 means skip frames
                if (skip > 0) {
                    t += skip;
                    break;
                }
                // duration 0: emit another symbol on same frame
            }
        }
    }

    return results;
}

std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDT &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id,
                  int max_symbols_per_step) {
    return tdt_greedy_decode(model.prediction(), model.joint(), encoder_out,
                             durations, blank_id, max_symbols_per_step);
}

// ─── Timestamped TDT Greedy Decode ────────────────────────────────────────

std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id, int max_symbols_per_step) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)});

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}), Tensor::zeros({1, hs})};
        }

        auto token = Tensor::zeros({1}, DType::Int32);
        int t = 0;

        while (t < seq_len) {
            auto enc_t = enc.slice({Slice(), Slice(t, t + 1)});

            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto pred = prediction.step(token, states);
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

                // Record token with start=t, end=t+duration
                int end_frame = t + std::max(skip, 1) - 1;
                if (end_frame >= seq_len)
                    end_frame = seq_len - 1;
                results[b].push_back({token_id, t, end_frame});

                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                if (skip > 0) {
                    t += skip;
                    break;
                }
            }
        }
    }

    return results;
}

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps(ParakeetTDT &model, const Tensor &encoder_out,
                                  const std::vector<int> &durations,
                                  int blank_id, int max_symbols_per_step) {
    return tdt_greedy_decode_with_timestamps(model.prediction(), model.joint(),
                                             encoder_out, durations, blank_id,
                                             max_symbols_per_step);
}

} // namespace parakeet
