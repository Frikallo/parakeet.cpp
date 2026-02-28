#pragma once

#include <cstdint>
#include <string>

#include <axiom/axiom.hpp>

namespace parakeet {

enum class AudioFormat { Unknown, WAV, FLAC, MP3, OGG };

struct AudioData {
    axiom::Tensor samples;  // float32, (num_samples,), mono, [-1,1]
    int sample_rate;         // always target rate (16000) after read_audio
    int original_sample_rate; // source rate before resampling
    int num_channels;        // original channel count before downmix
    int num_samples;         // = samples.shape()[0]
    float duration;          // seconds
    AudioFormat format;
};

// File loading (auto-detects format, resamples to target_sample_rate)
AudioData read_audio(const std::string &path, int target_sample_rate = 16000);

// Memory buffer: encoded bytes (WAV/FLAC/MP3/OGG detected by magic)
AudioData read_audio(const uint8_t *data, size_t len,
                     int target_sample_rate = 16000);

// Memory buffer: raw float32 PCM
AudioData read_audio(const float *pcm, size_t num_samples, int sample_rate,
                     int target_sample_rate = 16000);

// Memory buffer: raw int16 PCM
AudioData read_audio(const int16_t *pcm, size_t num_samples, int sample_rate,
                     int target_sample_rate = 16000);

// Duration query (header-only, no full decode)
float get_audio_duration(const std::string &path);

// Public resampler
axiom::Tensor resample(const axiom::Tensor &samples, int src_rate,
                       int dst_rate);

// Format detection helpers
AudioFormat detect_format_by_extension(const std::string &path);
AudioFormat detect_format_by_magic(const uint8_t *data, size_t len);

} // namespace parakeet
