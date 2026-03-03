// Streaming transcription with EOU (end-of-utterance) model
//
// Usage: example-stream <model.safetensors> <vocab.txt> <audio.wav>
//
// Simulates streaming by feeding the audio in fixed-size chunks.

#include <parakeet/parakeet.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav>\n"
                  << "  Requires an EOU-120M model.\n";
        return 1;
    }

    auto config = parakeet::make_eou_120m_config();
    parakeet::StreamingTranscriber t(argv[1], argv[2], config);
    auto audio = parakeet::read_audio(argv[3]);
    const float *data = audio.samples.typed_data<float>();
    int64_t total = audio.samples.shape()[0];
    int64_t chunk_size = 8000; // 0.5s at 16kHz

    std::cout << "Streaming " << (total / 16000.0) << "s of audio in "
              << chunk_size / 16000.0 << "s chunks...\n\n";

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
