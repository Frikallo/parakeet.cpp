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
    float f_max = -1.0f;   // defaults to sample_rate / 2
    bool normalize = true; // per-feature normalization (false = "NA" in NeMo)
};

// NeMo-compatible audio preprocessing:
//   dither -> mel spectrogram (STFT + mel filterbank + log) -> per-feature
//   normalize
// Input:  1D float32 tensor (num_samples,)
// Output: (1, n_frames, n_mels) float32 tensor ready for encoder
axiom::Tensor preprocess_audio(const axiom::Tensor &waveform,
                               const AudioConfig &config = {});

// Overload accepting AudioData (from audio_io.hpp).
// Validates sample rate matches config before preprocessing.
struct AudioData; // forward declaration
axiom::Tensor preprocess_audio(const AudioData &audio,
                               const AudioConfig &config = {});

// ─── Streaming Audio Preprocessor ────────────────────────────────────────────

// Maintains preemphasis state and STFT overlap buffer for chunk-wise
// processing. Output is NOT per-feature normalized (streaming doesn't have
// full-sequence stats). The encoder should handle unnormalized input or use
// running stats.

class StreamingAudioPreprocessor {
  public:
    explicit StreamingAudioPreprocessor(const AudioConfig &config = {});

    // Process a chunk of raw audio samples → (1, n_frames, n_mels)
    // May return empty tensor if not enough samples for a frame.
    axiom::Tensor process_chunk(const axiom::Tensor &samples);

    // Reset state for a new utterance
    void reset();

  private:
    AudioConfig config_;
    float preemph_last_sample_ = 0.0f; // state for preemphasis
    std::vector<float>
        overlap_buffer_;   // STFT overlap (win_length - hop_length samples)
    axiom::Tensor mel_fb_; // cached mel filterbank
    bool mel_fb_built_ = false;

    // Running statistics for normalization
    std::vector<double> mel_sum_;    // per-bin sum
    std::vector<double> mel_sq_sum_; // per-bin sum of squares
    int64_t frame_count_ = 0;

    void build_mel_filterbank();
};

} // namespace parakeet
