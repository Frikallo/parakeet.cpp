#pragma once

#include <utility>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── LSTM Cell ──────────────────────────────────────────────────────────────

// Single LSTM cell built from axiom primitives:
//   gates = Linear(input) + Linear(h_prev)          // (batch, 4*hidden)
//   i, f, g, o = gates.chunk(4, -1)
//   c_new = sigmoid(f)*c + sigmoid(i)*tanh(g)
//   h_new = sigmoid(o)*tanh(c_new)

using LSTMState = std::pair<Tensor, Tensor>; // (h, c)

class LSTMCell : public Module {
  public:
    LSTMCell();

    // Single timestep: (input, (h, c)) → (h_new, c_new)
    LSTMState forward(const Tensor &input, const LSTMState &state) const;

  private:
    Linear input_proj_;  // input_size → 4*hidden_size
    Linear hidden_proj_; // hidden_size → 4*hidden_size (no bias)
};

// ─── Stacked LSTM ───────────────────────────────────────────────────────────

class LSTM : public Module {
  public:
    explicit LSTM(int num_layers = 1);

    // Full sequence: (batch, seq, input) → (batch, seq, hidden)
    Tensor forward(const Tensor &input, std::vector<LSTMState> &states) const;

    // Single timestep: (batch, input) → (batch, hidden), updates states
    Tensor step(const Tensor &input, std::vector<LSTMState> &states) const;

    int num_layers() const { return num_layers_; }

  private:
    int num_layers_;
    ModuleList cells_;
};

} // namespace parakeet
