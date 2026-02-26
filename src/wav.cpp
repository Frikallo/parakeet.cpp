#include "parakeet/wav.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace parakeet {

namespace {

struct RiffHeader {
    char riff[4];       // "RIFF"
    uint32_t file_size; // total file size - 8
    char wave[4];       // "WAVE"
};

struct ChunkHeader {
    char id[4];
    uint32_t size;
};

struct FmtChunk {
    uint16_t audio_format; // 1 = PCM, 3 = IEEE float
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

} // namespace

WavData read_wav(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    // Read RIFF header
    RiffHeader riff{};
    file.read(reinterpret_cast<char *>(&riff), sizeof(riff));
    if (std::strncmp(riff.riff, "RIFF", 4) != 0 ||
        std::strncmp(riff.wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Not a valid WAV file: " + path);
    }

    // Scan chunks (don't assume 44-byte header)
    FmtChunk fmt{};
    bool found_fmt = false;
    std::vector<float> samples;

    while (file.good() && !file.eof()) {
        ChunkHeader chunk{};
        file.read(reinterpret_cast<char *>(&chunk), sizeof(chunk));
        if (!file.good())
            break;

        auto chunk_start = file.tellg();

        if (std::strncmp(chunk.id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char *>(&fmt),
                      std::min(static_cast<size_t>(chunk.size), sizeof(fmt)));
            found_fmt = true;

        } else if (std::strncmp(chunk.id, "data", 4) == 0) {
            if (!found_fmt) {
                throw std::runtime_error("WAV: data chunk before fmt chunk");
            }

            if (fmt.audio_format == 1 && fmt.bits_per_sample == 16) {
                // 16-bit PCM -> float32 in [-1, 1]
                int num_samples_total =
                    static_cast<int>(chunk.size / sizeof(int16_t));
                std::vector<int16_t> raw(num_samples_total);
                file.read(reinterpret_cast<char *>(raw.data()), chunk.size);

                samples.resize(num_samples_total);
                for (int i = 0; i < num_samples_total; ++i) {
                    samples[i] = static_cast<float>(raw[i]) / 32768.0f;
                }

            } else if (fmt.audio_format == 3 && fmt.bits_per_sample == 32) {
                // 32-bit IEEE float
                int num_samples_total =
                    static_cast<int>(chunk.size / sizeof(float));
                samples.resize(num_samples_total);
                file.read(reinterpret_cast<char *>(samples.data()), chunk.size);

            } else {
                throw std::runtime_error(
                    "Unsupported WAV format: format=" +
                    std::to_string(fmt.audio_format) +
                    " bits=" + std::to_string(fmt.bits_per_sample));
            }
        }

        // Advance to next chunk (chunks are word-aligned)
        auto next_pos =
            chunk_start + static_cast<std::streamoff>((chunk.size + 1) & ~1u);
        file.seekg(next_pos);
    }

    if (!found_fmt || samples.empty()) {
        throw std::runtime_error("WAV file missing fmt or data chunk: " + path);
    }

    int num_channels = fmt.num_channels;
    int total_frames = static_cast<int>(samples.size()) / num_channels;

    // Stereo -> mono downmix (average channels)
    std::vector<float> mono;
    if (num_channels > 1) {
        mono.resize(total_frames);
        for (int i = 0; i < total_frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < num_channels; ++c) {
                sum += samples[i * num_channels + c];
            }
            mono[i] = sum / static_cast<float>(num_channels);
        }
    } else {
        mono = std::move(samples);
    }

    // Create tensor from raw data
    auto tensor = axiom::Tensor::from_data(mono.data(), {mono.size()});

    return WavData{
        std::move(tensor),
        static_cast<int>(fmt.sample_rate),
        num_channels,
        total_frames,
    };
}

} // namespace parakeet
