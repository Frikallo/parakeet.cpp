#pragma once

#include <string>

#include <axiom/axiom.hpp>

namespace parakeet {

struct WavData {
    axiom::Tensor samples; // float32, shape (num_samples,)
    int sample_rate;
    int num_channels; // original channel count before downmix
    int num_samples;  // number of frames (after downmix = mono samples)
};

// Deprecated: use read_audio() from <parakeet/audio_io.hpp> instead.
// read_wav now forwards to read_audio internally.
//
// Read a WAV file and return float32 samples in [-1, 1].
// Multi-channel audio is downmixed to mono.
WavData read_wav(const std::string &path);

} // namespace parakeet
