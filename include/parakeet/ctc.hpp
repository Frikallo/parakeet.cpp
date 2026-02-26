#pragma once

#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/encoder.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── CTC Decoder Head ──────────────────────────────────────────────────────

class CTCDecoder : public Module {
  public:
    CTCDecoder();

    // (batch, seq, hidden_size) → (batch, seq, vocab_size) log probs
    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Conv1d proj_; // kernel_size=1 conv, equivalent to per-frame linear
};

// ─── Parakeet CTC Model ────────────────────────────────────────────────────

class ParakeetCTC : public Module {
  public:
    explicit ParakeetCTC(const CTCConfig &config = {});

    Tensor forward(const Tensor &input, const Tensor &mask = Tensor()) const;
    Tensor operator()(const Tensor &input,
                      const Tensor &mask = Tensor()) const {
        return forward(input, mask);
    }

    const CTCConfig &config() const { return config_; }

  private:
    CTCConfig config_;
    FastConformerEncoder encoder_;
    CTCDecoder decoder_;
};

// ─── CTC Greedy Decode ─────────────────────────────────────────────────────

// Takes (batch, seq, vocab_size) log probs, returns per-batch token sequences.
// Applies argmax, collapses repeats, removes blank (token 0).
std::vector<std::vector<int>> ctc_greedy_decode(const Tensor &log_probs,
                                                int blank_id = 1024);

} // namespace parakeet
