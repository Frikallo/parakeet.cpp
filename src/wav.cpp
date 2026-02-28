// Backward-compatible WAV reader — forwards to read_audio()
//
// Deprecated: use read_audio() from <parakeet/audio_io.hpp> instead.
// This wrapper is kept for existing code that uses read_wav().

#include "parakeet/wav.hpp"

#include "parakeet/audio_io.hpp"

namespace parakeet {

WavData read_wav(const std::string &path) {
    auto audio = read_audio(path);
    return WavData{
        std::move(audio.samples),
        audio.sample_rate,
        audio.num_channels,
        audio.num_samples,
    };
}

} // namespace parakeet
