// Voice Activity Detection — standalone VAD and ASR with VAD preprocessing
//
// Usage: example-vad <model.safetensors> <vocab.txt> <vad.safetensors>
// <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <vad.safetensors> "
                     "<audio.wav>\n";
        return 1;
    }

    const char *model_path = argv[1];
    const char *vocab_path = argv[2];
    const char *vad_path = argv[3];
    const char *audio_path = argv[4];

    // --- Standalone VAD ---
    std::cout << "=== Standalone VAD ===\n";
    parakeet::audio::SileroVAD vad(vad_path);
    auto audio = parakeet::read_audio(audio_path);
    auto segments = vad.detect(audio.samples);

    std::cout << "Detected " << segments.size() << " speech segment(s):\n";
    for (const auto &seg : segments) {
        float start = static_cast<float>(seg.start_sample) / 16000.0f;
        float end = static_cast<float>(seg.end_sample) / 16000.0f;
        std::cout << "  [" << std::fixed << std::setprecision(2) << start
                  << "s - " << end << "s]\n";
    }

    // Collect speech-only audio
    auto speech = parakeet::audio::collect_speech(audio.samples, segments);
    std::cout << "Original samples: " << audio.samples.shape()[0]
              << ", speech-only: " << speech.shape()[0] << "\n\n";

    // --- ASR with VAD preprocessing ---
    std::cout << "=== ASR + VAD ===\n";
    parakeet::Transcriber t(model_path, vocab_path);
    t.enable_vad(vad_path);

    parakeet::TranscribeOptions opts;
    opts.use_vad = true;
    opts.timestamps = true;
    auto result = t.transcribe(audio_path, opts);

    std::cout << "Text: " << result.text << "\n\n";
    for (const auto &w : result.word_timestamps) {
        std::cout << "  [" << std::fixed << std::setprecision(2) << w.start
                  << "s - " << w.end << "s] " << w.word << "\n";
    }

    return 0;
}
