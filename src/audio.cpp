#include "parakeet/audio.hpp"

#include <axiom/audio.hpp>
#include <axiom/fft.hpp>

namespace parakeet {

using namespace axiom;

Tensor preprocess_audio(const Tensor &waveform, const AudioConfig &config) {
    auto x = waveform; // (num_samples,)

    // 1. Dither: add small Gaussian noise
    if (config.dither > 0.0f) {
        auto noise = Tensor::randn(x.shape()) * config.dither;
        x = x + noise;
    }

    // 2. STFT with explicit win_length (400) != n_fft (512)
    auto window = fft::hann_window(config.win_length);
    auto stft_out =
        fft::stft(x, config.n_fft, config.hop_length, config.win_length, window,
                  /*center=*/true, /*pad_mode=*/"reflect");
    // stft_out: complex tensor (n_fft/2+1, n_frames)

    // 3. Power spectrum: |X|^2
    auto magnitudes = ops::abs(stft_out); // (n_fft/2+1, n_frames)
    auto power = magnitudes * magnitudes;

    // 4. Apply mel filterbank
    float f_max = config.f_max > 0 ? config.f_max : config.sample_rate / 2.0f;
    auto mel_fb =
        audio::mel_filterbank(config.n_fft / 2 + 1, config.n_mels,
                              config.sample_rate, config.f_min, f_max);
    // mel_fb: (n_freqs, n_mels)
    // mel_spec: (n_mels, n_frames) = mel_fb^T @ power
    auto mel_spec = ops::matmul(mel_fb.transpose(), power);

    // 5. Log (with guard value matching NeMo's 2^-24)
    constexpr float LOG_GUARD = 5.96e-8f; // 2^-24
    auto log_mel = ops::log(mel_spec + LOG_GUARD);

    // 6. Per-feature normalization (per mel bin, over time dimension)
    // log_mel: (n_mels, n_frames)
    auto mean = ops::mean(log_mel, {-1}, /*keep_dims=*/true); // (n_mels, 1)
    auto centered = log_mel - mean;
    auto var = ops::mean(centered * centered, {-1}, /*keep_dims=*/true);
    auto norm = centered / (ops::sqrt(var) + 1e-5f);

    // 7. Transpose and add batch dim: (n_mels, n_frames) -> (1, n_frames,
    // n_mels)
    return norm.transpose().unsqueeze(0);
}

} // namespace parakeet
