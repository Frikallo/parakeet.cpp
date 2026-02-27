#include "parakeet/tdt_ctc.hpp"

namespace parakeet {

ParakeetTDTCTC::ParakeetTDTCTC(const TDTCTCConfig &config)
    : config_(config), encoder_(config.encoder), prediction_(config.prediction),
      tdt_joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, tdt_joint_, ctc_decoder_);
}

// ─── TDT-CTC Decode Helpers ───────────────────────────────────────────────

std::vector<std::vector<int>>
tdt_greedy_decode(ParakeetTDTCTC &model, const Tensor &encoder_out,
                  const std::vector<int> &durations, int blank_id,
                  int max_symbols_per_step) {
    return tdt_greedy_decode(model.prediction(), model.tdt_joint(), encoder_out,
                             durations, blank_id, max_symbols_per_step);
}

std::vector<std::vector<TimestampedToken>> tdt_greedy_decode_with_timestamps(
    ParakeetTDTCTC &model, const Tensor &encoder_out,
    const std::vector<int> &durations, int blank_id, int max_symbols_per_step) {
    return tdt_greedy_decode_with_timestamps(
        model.prediction(), model.tdt_joint(), encoder_out, durations, blank_id,
        max_symbols_per_step);
}

} // namespace parakeet
