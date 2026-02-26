#pragma once

#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/encoder.hpp"
#include "parakeet/lstm.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── RNNT Prediction Network ───────────────────────────────────────────────

// Autoregressive decoder: Embedding → LSTM → Dropout
// Feeds previous token back in; prepends SOS (zero vector) at start.

class RNNTPrediction : public Module {
  public:
    explicit RNNTPrediction(const PredictionConfig &config = {});

    // Full sequence: (batch, label_seq) → (batch, label_seq, pred_hidden)
    Tensor forward(const Tensor &labels, std::vector<LSTMState> &states) const;

    // Single step: token (batch,) → (batch, pred_hidden), updates states
    Tensor step(const Tensor &token, std::vector<LSTMState> &states) const;

    const PredictionConfig &config() const { return config_; }

  private:
    PredictionConfig config_;
    Embedding embed_;
    LSTM lstm_;
    Dropout dropout_;
};

// ─── RNNT Joint Network ────────────────────────────────────────────────────

// Combines encoder and prediction outputs:
//   Linear(enc) + Linear(pred, no bias) → ReLU → Linear → log_softmax

class RNNTJoint : public Module {
  public:
    explicit RNNTJoint(const JointConfig &config = {});

    // enc: (batch, 1, joint_hidden), pred: (batch, 1, joint_hidden)
    //   → (batch, vocab_size) log probs
    Tensor forward(const Tensor &encoder_out,
                   const Tensor &prediction_out) const;

    const JointConfig &config() const { return config_; }

  private:
    JointConfig config_;
    Linear enc_proj_;
    Linear pred_proj_;
    Linear out_proj_;
};

// ─── Parakeet RNNT Model ───────────────────────────────────────────────────

class ParakeetRNNT : public Module {
  public:
    explicit ParakeetRNNT(const RNNTConfig &config = {});

    const RNNTConfig &config() const { return config_; }

    FastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    RNNTJoint &joint() { return joint_; }

  private:
    RNNTConfig config_;
    FastConformerEncoder encoder_;
    RNNTPrediction prediction_;
    RNNTJoint joint_;
};

// ─── RNNT Greedy Decode ────────────────────────────────────────────────────

// Per encoder frame, loop prediction until blank emitted.
// encoder_out: (batch, seq, hidden) → per-batch token sequences
std::vector<std::vector<int>> rnnt_greedy_decode(ParakeetRNNT &model,
                                                 const Tensor &encoder_out,
                                                 int blank_id = 1024,
                                                 int max_symbols_per_step = 10);

} // namespace parakeet
