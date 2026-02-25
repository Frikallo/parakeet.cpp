#include "parakeet/tdt_ctc.hpp"

namespace parakeet {

ParakeetTDTCTC::ParakeetTDTCTC(const TDTCTCConfig &config)
    : config_(config), prediction_(config.prediction),
      tdt_joint_(config.joint, static_cast<int>(config.durations.size())) {
    AX_REGISTER_MODULES(encoder_, prediction_, tdt_joint_, ctc_decoder_);
}

} // namespace parakeet
