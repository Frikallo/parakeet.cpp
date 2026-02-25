#include "parakeet/lstm.hpp"

namespace parakeet {

// ─── LSTMCell ───────────────────────────────────────────────────────────────

LSTMCell::LSTMCell() : input_proj_(true), hidden_proj_(false) {
    AX_REGISTER_MODULES(input_proj_, hidden_proj_);
}

LSTMState LSTMCell::forward(const Tensor &input, const LSTMState &state) const {
    auto &[h, c] = state;

    // gates: (batch, 4*hidden_size)
    auto gates = input_proj_(input) + hidden_proj_(h);

    // Split into 4 gates
    auto chunks = gates.chunk(4, -1);
    auto i_gate = ops::sigmoid(chunks[0]); // input gate
    auto f_gate = ops::sigmoid(chunks[1]); // forget gate
    auto g_gate = ops::tanh(chunks[2]);    // cell candidate
    auto o_gate = ops::sigmoid(chunks[3]); // output gate

    // Cell and hidden state update
    auto c_new = f_gate * c + i_gate * g_gate;
    auto h_new = o_gate * ops::tanh(c_new);

    return {h_new, c_new};
}

// ─── LSTM ───────────────────────────────────────────────────────────────────

LSTM::LSTM(int num_layers) : num_layers_(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        cells_.emplace_back<LSTMCell>();
    }
    AX_REGISTER_MODULE(cells_);
}

Tensor LSTM::step(const Tensor &input, std::vector<LSTMState> &states) const {
    auto x = input;
    int layer = 0;
    for (const auto &cell : cells_.each<LSTMCell>()) {
        states[layer] = cell.forward(x, states[layer]);
        x = states[layer].first; // h is the output
        ++layer;
    }
    return x;
}

Tensor LSTM::forward(const Tensor &input,
                     std::vector<LSTMState> &states) const {
    // input: (batch, seq, features)
    auto shape = input.shape();
    int seq_len = static_cast<int>(shape[1]);

    std::vector<Tensor> outputs;
    outputs.reserve(seq_len);

    for (int t = 0; t < seq_len; ++t) {
        auto x_t = input.slice({Slice(), Slice(t, t + 1)})
                       .squeeze(1); // (batch, features)
        outputs.push_back(step(x_t, states));
    }

    return Tensor::stack(outputs, /*axis=*/1); // (batch, seq, hidden)
}

} // namespace parakeet
