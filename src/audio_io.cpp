// Audio I/O: multi-format loading, resampling, memory buffer input
//
// Uses single-header C decoders:
//   - dr_wav, dr_flac, dr_mp3 (mackron/dr_libs)
//   - stb_vorbis (nothings/stb)

#define DR_WAV_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION

#include "dr_wav.h"
#include "dr_flac.h"
#include "dr_mp3.h"

// stb_vorbis is a .c file — include as extern "C", then clean up leaked macros
extern "C" {
#include "stb_vorbis.c"
}
#undef C
#undef R
#undef L

#include "parakeet/audio_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace parakeet {

// ─── Format Detection ────────────────────────────────────────────────────────

AudioFormat detect_format_by_extension(const std::string &path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos)
        return AudioFormat::Unknown;

    std::string ext = path.substr(dot + 1);
    // lowercase
    for (auto &c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "wav" || ext == "wave")
        return AudioFormat::WAV;
    if (ext == "flac")
        return AudioFormat::FLAC;
    if (ext == "mp3")
        return AudioFormat::MP3;
    if (ext == "ogg" || ext == "oga" || ext == "ogv")
        return AudioFormat::OGG;
    return AudioFormat::Unknown;
}

AudioFormat detect_format_by_magic(const uint8_t *data, size_t len) {
    if (len < 2)
        return AudioFormat::Unknown;

    // MP3: frame sync (0xFF 0xFB/0xFA/0xF3/0xF2/0xE0-0xFF) — check early (2 bytes)
    if (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
        return AudioFormat::MP3;
    }

    // MP3: ID3 tag (3 bytes)
    if (len >= 3 && data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
        return AudioFormat::MP3;
    }

    if (len < 4)
        return AudioFormat::Unknown;

    // RIFF....WAVE
    if (len >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' &&
        data[3] == 'F' && data[8] == 'W' && data[9] == 'A' &&
        data[10] == 'V' && data[11] == 'E') {
        return AudioFormat::WAV;
    }

    // fLaC
    if (data[0] == 'f' && data[1] == 'L' && data[2] == 'a' &&
        data[3] == 'C') {
        return AudioFormat::FLAC;
    }

    // OGG: OggS
    if (data[0] == 'O' && data[1] == 'g' && data[2] == 'g' &&
        data[3] == 'S') {
        return AudioFormat::OGG;
    }

    return AudioFormat::Unknown;
}

// ─── Sinc Resampler ──────────────────────────────────────────────────────────

namespace {

// Modified Bessel function I0 (for Kaiser window)
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 30; ++k) {
        term *= (x * x) / (4.0 * k * k);
        sum += term;
        if (term < 1e-12 * sum)
            break;
    }
    return sum;
}

// Kaiser window
double kaiser_window(double n, double N, double beta) {
    double arg = 2.0 * n / N - 1.0;
    double val = 1.0 - arg * arg;
    if (val < 0.0)
        val = 0.0;
    return bessel_i0(beta * std::sqrt(val)) / bessel_i0(beta);
}

// Windowed sinc interpolation
std::vector<float> sinc_resample(const float *input, size_t input_len,
                                  int src_rate, int dst_rate) {
    if (src_rate == dst_rate) {
        return std::vector<float>(input, input + input_len);
    }

    // GCD simplification
    int g = std::gcd(src_rate, dst_rate);
    int up = dst_rate / g;
    int down = src_rate / g;

    size_t output_len =
        static_cast<size_t>((static_cast<int64_t>(input_len) * up + down - 1) /
                            down);
    std::vector<float> output(output_len);

    // Kaiser window parameters: 80dB stopband, 16-tap half-width
    constexpr int HALF_WIDTH = 16;
    constexpr double BETA = 7.857; // ~80dB stopband

    double ratio = static_cast<double>(src_rate) / dst_rate;
    double cutoff = std::min(1.0, 1.0 / std::max(ratio, 1.0));
    double filter_scale = cutoff;

    // If downsampling, widen the filter
    double sample_ratio = static_cast<double>(dst_rate) / src_rate;
    double width_factor = std::max(1.0, ratio);

    for (size_t i = 0; i < output_len; ++i) {
        double src_pos = static_cast<double>(i) / sample_ratio;
        int center = static_cast<int>(std::floor(src_pos));

        double sum = 0.0;
        double weight_sum = 0.0;

        int start = center - HALF_WIDTH + 1;
        int end = center + HALF_WIDTH;

        for (int j = start; j <= end; ++j) {
            if (j < 0 || j >= static_cast<int>(input_len))
                continue;

            double dist = src_pos - j;
            double window_pos = dist / width_factor;

            // Kaiser window
            double w = 0.0;
            if (std::abs(window_pos) <= HALF_WIDTH) {
                w = kaiser_window(window_pos + HALF_WIDTH, 2.0 * HALF_WIDTH,
                                  BETA);
            } else {
                continue;
            }

            // Sinc function
            double sinc_val;
            double x = dist * cutoff * M_PI;
            if (std::abs(x) < 1e-10) {
                sinc_val = 1.0;
            } else {
                sinc_val = std::sin(x) / x;
            }

            double weight = sinc_val * w * filter_scale;
            sum += input[j] * weight;
            weight_sum += weight;
        }

        output[i] =
            (weight_sum > 1e-10) ? static_cast<float>(sum / weight_sum) : 0.0f;
    }

    return output;
}

// Downmix interleaved multi-channel to mono
std::vector<float> downmix_to_mono(const float *interleaved, size_t total,
                                    int channels) {
    if (channels == 1) {
        return std::vector<float>(interleaved, interleaved + total);
    }
    size_t frames = total / channels;
    std::vector<float> mono(frames);
    float inv_ch = 1.0f / static_cast<float>(channels);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += interleaved[i * channels + c];
        }
        mono[i] = sum * inv_ch;
    }
    return mono;
}

// Build AudioData from decoded mono samples
AudioData make_audio_data(std::vector<float> &&mono, int src_rate,
                           int target_rate, int original_channels,
                           AudioFormat fmt) {
    int original_rate = src_rate;
    int num_mono = static_cast<int>(mono.size());

    // Resample if needed
    std::vector<float> final_samples;
    if (src_rate != target_rate) {
        final_samples =
            sinc_resample(mono.data(), mono.size(), src_rate, target_rate);
    } else {
        final_samples = std::move(mono);
    }

    int num_samples = static_cast<int>(final_samples.size());
    float duration =
        static_cast<float>(num_mono) / static_cast<float>(original_rate);

    auto tensor = axiom::Tensor::from_data(
        final_samples.data(),
        axiom::Shape{static_cast<size_t>(num_samples)}, true);

    return AudioData{
        std::move(tensor),
        target_rate,
        original_rate,
        original_channels,
        num_samples,
        duration,
        fmt,
    };
}

} // namespace

// ─── Public Resampler ────────────────────────────────────────────────────────

axiom::Tensor resample(const axiom::Tensor &samples, int src_rate,
                        int dst_rate) {
    if (src_rate == dst_rate)
        return samples;

    auto cont = samples.ascontiguousarray();
    const float *data = cont.typed_data<float>();
    size_t len = cont.shape()[0];

    auto resampled = sinc_resample(data, len, src_rate, dst_rate);
    return axiom::Tensor::from_data(
        resampled.data(), axiom::Shape{resampled.size()}, true);
}

// ─── Per-Format Decoders ─────────────────────────────────────────────────────

namespace {

// WAV via dr_wav
AudioData read_wav_dr(const std::string &path, int target_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    size_t total_frames = wav.totalPCMFrameCount;
    int channels = wav.channels;
    int sample_rate = wav.sampleRate;

    std::vector<float> interleaved(total_frames * channels);
    size_t frames_read =
        drwav_read_pcm_frames_f32(&wav, total_frames, interleaved.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        throw std::runtime_error("Failed to decode WAV: " + path);
    }
    interleaved.resize(frames_read * channels);

    auto mono = downmix_to_mono(interleaved.data(), interleaved.size(), channels);
    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::WAV);
}

AudioData read_wav_dr_memory(const uint8_t *data, size_t len,
                              int target_rate) {
    drwav wav;
    if (!drwav_init_memory(&wav, data, len, nullptr)) {
        throw std::runtime_error("Cannot decode WAV from memory buffer");
    }

    size_t total_frames = wav.totalPCMFrameCount;
    int channels = wav.channels;
    int sample_rate = wav.sampleRate;

    std::vector<float> interleaved(total_frames * channels);
    size_t frames_read =
        drwav_read_pcm_frames_f32(&wav, total_frames, interleaved.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        throw std::runtime_error("Failed to decode WAV from memory");
    }
    interleaved.resize(frames_read * channels);

    auto mono = downmix_to_mono(interleaved.data(), interleaved.size(), channels);
    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::WAV);
}

// FLAC via dr_flac
AudioData read_flac_dr(const std::string &path, int target_rate) {
    unsigned int channels, sample_rate;
    drflac_uint64 total_frames;
    float *interleaved = drflac_open_file_and_read_pcm_frames_f32(
        path.c_str(), &channels, &sample_rate, &total_frames, nullptr);

    if (!interleaved) {
        throw std::runtime_error("Cannot open FLAC file: " + path);
    }

    auto mono = downmix_to_mono(interleaved, total_frames * channels,
                                 static_cast<int>(channels));
    drflac_free(interleaved, nullptr);

    return make_audio_data(std::move(mono), static_cast<int>(sample_rate),
                            target_rate, static_cast<int>(channels),
                            AudioFormat::FLAC);
}

AudioData read_flac_dr_memory(const uint8_t *data, size_t len,
                               int target_rate) {
    unsigned int channels, sample_rate;
    drflac_uint64 total_frames;
    float *interleaved = drflac_open_memory_and_read_pcm_frames_f32(
        data, len, &channels, &sample_rate, &total_frames, nullptr);

    if (!interleaved) {
        throw std::runtime_error("Cannot decode FLAC from memory buffer");
    }

    auto mono = downmix_to_mono(interleaved, total_frames * channels,
                                 static_cast<int>(channels));
    drflac_free(interleaved, nullptr);

    return make_audio_data(std::move(mono), static_cast<int>(sample_rate),
                            target_rate, static_cast<int>(channels),
                            AudioFormat::FLAC);
}

// MP3 via dr_mp3
AudioData read_mp3_dr(const std::string &path, int target_rate) {
    drmp3_config config;
    drmp3_uint64 total_frames;
    float *interleaved = drmp3_open_file_and_read_pcm_frames_f32(
        path.c_str(), &config, &total_frames, nullptr);

    if (!interleaved) {
        throw std::runtime_error("Cannot open MP3 file: " + path);
    }

    int channels = config.channels;
    int sample_rate = config.sampleRate;

    auto mono = downmix_to_mono(interleaved, total_frames * channels, channels);
    drmp3_free(interleaved, nullptr);

    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::MP3);
}

AudioData read_mp3_dr_memory(const uint8_t *data, size_t len,
                              int target_rate) {
    drmp3_config config;
    drmp3_uint64 total_frames;
    float *interleaved = drmp3_open_memory_and_read_pcm_frames_f32(
        data, len, &config, &total_frames, nullptr);

    if (!interleaved) {
        throw std::runtime_error("Cannot decode MP3 from memory buffer");
    }

    int channels = config.channels;
    int sample_rate = config.sampleRate;

    auto mono = downmix_to_mono(interleaved, total_frames * channels, channels);
    drmp3_free(interleaved, nullptr);

    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::MP3);
}

// OGG Vorbis via stb_vorbis
AudioData read_ogg_stb(const std::string &path, int target_rate) {
    int channels, sample_rate;
    short *raw_data;
    int total_samples =
        stb_vorbis_decode_filename(path.c_str(), &channels, &sample_rate,
                                    &raw_data);
    if (total_samples < 0) {
        throw std::runtime_error("Cannot open OGG file: " + path);
    }

    // Convert int16 to float32
    size_t total = static_cast<size_t>(total_samples) * channels;
    std::vector<float> interleaved(total);
    for (size_t i = 0; i < total; ++i) {
        interleaved[i] = static_cast<float>(raw_data[i]) / 32768.0f;
    }
    free(raw_data);

    auto mono = downmix_to_mono(interleaved.data(), interleaved.size(), channels);
    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::OGG);
}

AudioData read_ogg_stb_memory(const uint8_t *data, size_t len,
                               int target_rate) {
    int channels, sample_rate;
    short *raw_data;
    int total_samples =
        stb_vorbis_decode_memory(data, static_cast<int>(len), &channels,
                                  &sample_rate, &raw_data);
    if (total_samples < 0) {
        throw std::runtime_error("Cannot decode OGG from memory buffer");
    }

    size_t total = static_cast<size_t>(total_samples) * channels;
    std::vector<float> interleaved(total);
    for (size_t i = 0; i < total; ++i) {
        interleaved[i] = static_cast<float>(raw_data[i]) / 32768.0f;
    }
    free(raw_data);

    auto mono = downmix_to_mono(interleaved.data(), interleaved.size(), channels);
    return make_audio_data(std::move(mono), sample_rate, target_rate, channels,
                            AudioFormat::OGG);
}

} // namespace

// ─── File Loading ────────────────────────────────────────────────────────────

AudioData read_audio(const std::string &path, int target_sample_rate) {
    // Try extension first
    auto fmt = detect_format_by_extension(path);

    // If unknown extension, try magic bytes
    if (fmt == AudioFormat::Unknown) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open audio file: " + path);
        }
        uint8_t header[12];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        size_t bytes_read = static_cast<size_t>(file.gcount());
        fmt = detect_format_by_magic(header, bytes_read);
    }

    switch (fmt) {
    case AudioFormat::WAV:
        return read_wav_dr(path, target_sample_rate);
    case AudioFormat::FLAC:
        return read_flac_dr(path, target_sample_rate);
    case AudioFormat::MP3:
        return read_mp3_dr(path, target_sample_rate);
    case AudioFormat::OGG:
        return read_ogg_stb(path, target_sample_rate);
    default:
        throw std::runtime_error(
            "Unsupported or unrecognized audio format: " + path);
    }
}

// ─── Memory Buffer: Encoded Bytes ────────────────────────────────────────────

AudioData read_audio(const uint8_t *data, size_t len,
                      int target_sample_rate) {
    auto fmt = detect_format_by_magic(data, len);

    switch (fmt) {
    case AudioFormat::WAV:
        return read_wav_dr_memory(data, len, target_sample_rate);
    case AudioFormat::FLAC:
        return read_flac_dr_memory(data, len, target_sample_rate);
    case AudioFormat::MP3:
        return read_mp3_dr_memory(data, len, target_sample_rate);
    case AudioFormat::OGG:
        return read_ogg_stb_memory(data, len, target_sample_rate);
    default:
        throw std::runtime_error(
            "Unsupported or unrecognized audio format in memory buffer");
    }
}

// ─── Memory Buffer: Raw float32 PCM ──────────────────────────────────────────

AudioData read_audio(const float *pcm, size_t num_samples, int sample_rate,
                      int target_sample_rate) {
    std::vector<float> mono(pcm, pcm + num_samples);
    return make_audio_data(std::move(mono), sample_rate, target_sample_rate, 1,
                            AudioFormat::Unknown);
}

// ─── Memory Buffer: Raw int16 PCM ────────────────────────────────────────────

AudioData read_audio(const int16_t *pcm, size_t num_samples, int sample_rate,
                      int target_sample_rate) {
    std::vector<float> mono(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        mono[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }
    return make_audio_data(std::move(mono), sample_rate, target_sample_rate, 1,
                            AudioFormat::Unknown);
}

// ─── Duration Query ──────────────────────────────────────────────────────────

float get_audio_duration(const std::string &path) {
    auto fmt = detect_format_by_extension(path);

    // If unknown extension, try magic
    if (fmt == AudioFormat::Unknown) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open audio file: " + path);
        }
        uint8_t header[12];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        size_t bytes_read = static_cast<size_t>(file.gcount());
        fmt = detect_format_by_magic(header, bytes_read);
    }

    switch (fmt) {
    case AudioFormat::WAV: {
        drwav wav;
        if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
            throw std::runtime_error("Cannot open WAV file: " + path);
        }
        float duration = static_cast<float>(wav.totalPCMFrameCount) /
                          static_cast<float>(wav.sampleRate);
        drwav_uninit(&wav);
        return duration;
    }
    case AudioFormat::FLAC: {
        drflac *flac = drflac_open_file(path.c_str(), nullptr);
        if (!flac) {
            throw std::runtime_error("Cannot open FLAC file: " + path);
        }
        float duration = static_cast<float>(flac->totalPCMFrameCount) /
                          static_cast<float>(flac->sampleRate);
        drflac_close(flac);
        return duration;
    }
    case AudioFormat::MP3: {
        // MP3 needs full scan for accurate duration — fall back to full decode
        auto audio = read_audio(path);
        return audio.duration;
    }
    case AudioFormat::OGG: {
        // stb_vorbis can open and read info without full decode
        int error;
        stb_vorbis *v = stb_vorbis_open_filename(path.c_str(), &error, nullptr);
        if (!v) {
            throw std::runtime_error("Cannot open OGG file: " + path);
        }
        stb_vorbis_info info = stb_vorbis_get_info(v);
        unsigned int total =
            stb_vorbis_stream_length_in_samples(v);
        float duration =
            static_cast<float>(total) / static_cast<float>(info.sample_rate);
        stb_vorbis_close(v);
        return duration;
    }
    default:
        throw std::runtime_error(
            "Unsupported or unrecognized audio format: " + path);
    }
}

} // namespace parakeet
