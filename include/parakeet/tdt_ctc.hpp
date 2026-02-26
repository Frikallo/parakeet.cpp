#pragma once

#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include "parakeet/config.hpp"
#include "parakeet/ctc.hpp"
#include "parakeet/encoder.hpp"
#include "parakeet/tdt.hpp"

namespace parakeet {

using namespace axiom;
using namespace axiom::nn;

// ─── Parakeet TDT-CTC Hybrid Model ─────────────────────────────────────────

// Shared encoder with both a TDT head and a CTC head.
// Can switch decoder at inference time.

class ParakeetTDTCTC : public Module {
  public:
    explicit ParakeetTDTCTC(const TDTCTCConfig &config = {});

    const TDTCTCConfig &config() const { return config_; }

    FastConformerEncoder &encoder() { return encoder_; }
    RNNTPrediction &prediction() { return prediction_; }
    TDTJoint &tdt_joint() { return tdt_joint_; }
    CTCDecoder &ctc_decoder() { return ctc_decoder_; }

  private:
    TDTCTCConfig config_;
    FastConformerEncoder encoder_;

    // TDT head
    RNNTPrediction prediction_;
    TDTJoint tdt_joint_;

    // CTC head
    CTCDecoder ctc_decoder_;
};

// ─── TDT-CTC Decode Helpers ───────────────────────────────────────────────

// TDT greedy decode using the TDT head of a ParakeetTDTCTC model.
std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDTCTC &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id = 1024,
                  int max_symbols_per_step = 10);

} // namespace parakeet
