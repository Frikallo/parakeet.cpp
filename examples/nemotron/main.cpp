// Nemotron streaming transcription with configurable latency
//
// Usage: example-nemotron <model.safetensors> <vocab.txt> <audio.wav> [latency]
//
// Latency frames: 0=80ms, 1=160ms, 6=560ms, 13=1120ms (default: 1)

#include <parakeet/parakeet.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav> [latency]\n"
                  << "  latency: 0=80ms, 1=160ms, 6=560ms, 13=1120ms "
                     "(default: 1)\n";
        return 1;
    }

    int latency = (argc == 5) ? std::atoi(argv[4]) : 1;
    auto cfg = parakeet::make_nemotron_600m_config(latency);

    std::cout << "Nemotron 600M — latency=" << latency << " ("
              << (latency + 1) * 80 << "ms)\n\n";

    parakeet::NemotronTranscriber t(argv[1], argv[2], cfg);

    // Simulate streaming with 0.5s chunks
    auto audio = parakeet::read_audio(argv[3]);
    const float *data = audio.samples.typed_data<float>();
    int64_t total = audio.samples.shape()[0];
    int64_t chunk_size = 8000; // 0.5s at 16kHz

    for (int64_t offset = 0; offset < total; offset += chunk_size) {
        int64_t n = std::min(chunk_size, total - offset);
        auto text = t.transcribe_chunk(data + offset, static_cast<size_t>(n));
        if (!text.empty()) {
            std::cout << text << std::flush;
        }
    }

    std::cout << "\n\nFinal: " << t.get_text() << std::endl;

    return 0;
}
