#include "parakeet/rnnt.hpp"

namespace parakeet {

// ─── RNNTPrediction ─────────────────────────────────────────────────────────

RNNTPrediction::RNNTPrediction(const PredictionConfig &config)
    : config_(config), lstm_(config.num_lstm_layers), dropout_(config.dropout) {
    AX_REGISTER_MODULES(embed_, lstm_, dropout_);
}

Tensor RNNTPrediction::forward(const Tensor &labels,
                               std::vector<LSTMState> &states) const {
    auto x = embed_(labels); // (batch, label_seq, pred_hidden)
    x = lstm_.forward(x, states);
    x = dropout_(x);
    return x;
}

Tensor RNNTPrediction::step(const Tensor &token,
                            std::vector<LSTMState> &states) const {
    auto x = embed_(token); // (batch, pred_hidden)
    x = lstm_.step(x, states);
    x = dropout_(x);
    return x;
}

// ─── RNNTJoint ──────────────────────────────────────────────────────────────

RNNTJoint::RNNTJoint(const JointConfig &config)
    : config_(config), enc_proj_(true), pred_proj_(false), out_proj_(true) {
    AX_REGISTER_MODULES(enc_proj_, pred_proj_, out_proj_);
}

Tensor RNNTJoint::forward(const Tensor &encoder_out,
                          const Tensor &prediction_out) const {
    // Broadcast add: enc (batch, 1, joint) + pred (batch, 1, joint)
    auto x = enc_proj_(encoder_out) + pred_proj_(prediction_out);
    x = ops::relu(x);
    x = out_proj_(x);
    return ops::log_softmax(x, /*axis=*/-1);
}

// ─── ParakeetRNNT ───────────────────────────────────────────────────────────

ParakeetRNNT::ParakeetRNNT(const RNNTConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      joint_(config.joint) {
    AX_REGISTER_MODULES(encoder_, prediction_, joint_);
}

// ─── RNNT Greedy Decode ─────────────────────────────────────────────────────

std::vector<std::vector<int>> rnnt_greedy_decode(ParakeetRNNT &model,
                                                 const Tensor &encoder_out,
                                                 int blank_id,
                                                 int max_symbols_per_step) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        // Extract single batch element: (1, seq, hidden)
        auto enc = encoder_out.slice({Slice(b, b + 1)});

        // Initialize LSTM states to zeros
        int num_layers = model.prediction().config().num_lstm_layers;
        size_t hs = model.prediction().config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}), Tensor::zeros({1, hs})};
        }

        // Start with blank token (SOS)
        auto token = Tensor({1}, DType::Int32);
        token.fill(blank_id);

        for (int t = 0; t < seq_len; ++t) {
            auto enc_t =
                enc.slice({Slice(), Slice(t, t + 1)}); // (1, 1, hidden)

            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto saved_states = states;

                auto pred = model.prediction().step(token, states);
                pred = pred.unsqueeze(1); // (1, 1, pred_hidden)

                auto logits = model.joint().forward(enc_t, pred);
                // logits: (1, 1, vocab)
                auto best =
                    ops::argmax(logits.squeeze(0).squeeze(0), /*axis=*/-1);
                int token_id = best.item<int>();

                if (token_id == blank_id) {
                    states = saved_states;
                    break;
                }

                results[b].push_back(token_id);
                token = Tensor({1}, DType::Int32);
                token.fill(token_id);
            }
        }
    }

    return results;
}

} // namespace parakeet
