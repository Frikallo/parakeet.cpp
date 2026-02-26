#include "parakeet/audio.hpp"

#include <axiom/fft.hpp>

#include <cmath>
#include <vector>

namespace parakeet {

using namespace axiom;

// ─── Slaney Mel Scale (matches librosa / NeMo) ──────────────────────────────

namespace {

// Slaney mel scale: linear below 1000 Hz, log above
constexpr double MEL_BREAK_FREQ = 1000.0;
constexpr double MEL_BREAK_MEL = 15.0;               // 1000 / (200/3)
constexpr double MEL_LINEAR_SCALE = 200.0 / 3.0;     // 3/200 Hz per mel
constexpr double MEL_LOG_STEP = 0.06875177742094912; // ln(6.4) / 27

double hz_to_mel_slaney(double freq) {
    if (freq < MEL_BREAK_FREQ) {
        return freq / MEL_LINEAR_SCALE;
    }
    return MEL_BREAK_MEL + std::log(freq / MEL_BREAK_FREQ) / MEL_LOG_STEP;
}

double mel_to_hz_slaney(double mel) {
    if (mel < MEL_BREAK_MEL) {
        return mel * MEL_LINEAR_SCALE;
    }
    return MEL_BREAK_FREQ * std::exp((mel - MEL_BREAK_MEL) * MEL_LOG_STEP);
}

// Build mel filterbank with slaney normalization (area normalization)
// Returns: (n_freqs, n_mels)
Tensor build_mel_filterbank(int n_freqs, int n_mels, float sample_rate,
                            float f_min, float f_max) {
    double mel_min = hz_to_mel_slaney(f_min);
    double mel_max = hz_to_mel_slaney(f_max);

    // n_mels + 2 evenly spaced mel points
    std::vector<double> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_min + static_cast<double>(i) * (mel_max - mel_min) /
                                   static_cast<double>(n_mels + 1);
    }

    // Convert to Hz
    std::vector<double> hz_pts(mel_pts.size());
    for (size_t i = 0; i < mel_pts.size(); ++i) {
        hz_pts[i] = mel_to_hz_slaney(mel_pts[i]);
    }

    // FFT bin frequencies
    std::vector<double> fft_freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        fft_freqs[i] = static_cast<double>(i) *
                       static_cast<double>(sample_rate) /
                       (2.0 * static_cast<double>(n_freqs - 1));
    }

    // Build filterbank (n_freqs, n_mels)
    std::vector<float> fb(n_freqs * n_mels, 0.0f);

    for (int m = 0; m < n_mels; ++m) {
        double left = hz_pts[m];
        double center = hz_pts[m + 1];
        double right = hz_pts[m + 2];

        // Slaney normalization: 2 / (right - left)
        double enorm = 2.0 / (right - left);

        for (int f = 0; f < n_freqs; ++f) {
            double freq = fft_freqs[f];
            double val = 0.0;

            if (freq >= left && freq <= center && center > left) {
                val = (freq - left) / (center - left);
            } else if (freq > center && freq <= right && right > center) {
                val = (right - freq) / (right - center);
            }

            fb[f * n_mels + m] = static_cast<float>(val * enorm);
        }
    }

    return Tensor::from_data(
        fb.data(),
        Shape{static_cast<size_t>(n_freqs), static_cast<size_t>(n_mels)}, true);
}

} // namespace

// ─── Preprocessing ───────────────────────────────────────────────────────────

Tensor preprocess_audio(const Tensor &waveform, const AudioConfig &config) {
    auto x = waveform; // (num_samples,)

    // 1. Preemphasis: x[n] = x[n] - 0.97 * x[n-1]
    {
        auto x_cont = x.ascontiguousarray();
        size_t n = x_cont.shape()[0];
        const float *src = x_cont.typed_data<float>();
        std::vector<float> pre(n);
        pre[0] = src[0];
        for (size_t i = 1; i < n; ++i) {
            pre[i] = src[i] - 0.97f * src[i - 1];
        }
        x = Tensor::from_data(pre.data(), Shape{n}, true);
    }

    // 2. STFT with symmetric Hann window (periodic=false, matching NeMo)
    auto window = fft::hann_window(config.win_length, /*periodic=*/false);
    auto stft_out =
        fft::stft(x, config.n_fft, config.hop_length, config.win_length, window,
                  /*center=*/true, /*pad_mode=*/"reflect");

    // 3. Power spectrum: |X|^2
    auto magnitudes = ops::abs(stft_out); // (n_fft/2+1, n_frames)
    auto power = magnitudes * magnitudes;

    // 4. Apply mel filterbank (slaney scale + slaney normalization)
    float f_max = config.f_max > 0 ? config.f_max : config.sample_rate / 2.0f;
    auto mel_fb = build_mel_filterbank(config.n_fft / 2 + 1, config.n_mels,
                                       config.sample_rate, config.f_min, f_max);
    // mel_spec: (n_mels, n_frames) = mel_fb^T @ power
    auto mel_spec = ops::matmul(mel_fb.transpose(), power);

    // 5. Log (with guard value matching NeMo's 2^-24)
    constexpr float LOG_GUARD = 5.96046448e-8f; // 2^-24
    auto log_mel = ops::log(mel_spec + LOG_GUARD);

    // 6. Per-feature normalization (per mel bin, over time dimension)
    // NeMo uses unbiased variance (N-1 denominator)
    // log_mel: (n_mels, n_frames)
    auto mean = ops::mean(log_mel, {1}).unsqueeze(1); // (n_mels, 1)
    auto centered = log_mel - mean;
    int n_frames = static_cast<int>(log_mel.shape()[1]);
    // Unbiased variance: sum((x-mean)^2) / (N-1)
    auto var_sum = ops::sum(centered * centered, {1}).unsqueeze(1);
    auto var = var_sum / static_cast<float>(n_frames - 1);
    auto norm = centered / (ops::sqrt(var) + 1e-5f);

    // 7. Transpose and add batch dim: (n_mels, n_frames) -> (1, n_frames,
    // n_mels)
    auto result = norm.transpose().unsqueeze(0);
    return result;
}

} // namespace parakeet
