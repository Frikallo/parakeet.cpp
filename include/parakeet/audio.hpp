#pragma once

#include <axiom/axiom.hpp>

namespace parakeet {

struct AudioConfig {
    int sample_rate = 16000;
    int n_fft = 512;
    int win_length = 400; // 0.025s * 16000
    int hop_length = 160; // 0.01s * 16000
    int n_mels = 80;
    float dither = 1e-5f;
    float f_min = 0.0f;
    float f_max = -1.0f; // defaults to sample_rate / 2
};

// NeMo-compatible audio preprocessing:
//   dither -> mel spectrogram (STFT + mel filterbank + log) -> per-feature
//   normalize
// Input:  1D float32 tensor (num_samples,)
// Output: (1, n_frames, n_mels) float32 tensor ready for encoder
axiom::Tensor preprocess_audio(const axiom::Tensor &waveform,
                               const AudioConfig &config = {});

} // namespace parakeet
