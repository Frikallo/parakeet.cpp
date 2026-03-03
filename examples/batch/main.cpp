// Batch transcription — multiple files in one forward pass
//
// Usage: example-batch <model.safetensors> <vocab.txt> <audio1.wav> [audio2.wav
// ...]

#include <parakeet/parakeet.hpp>

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio1.wav> "
                     "[audio2.wav ...]\n";
        return 1;
    }

    parakeet::Transcriber t(argv[1], argv[2]);

    std::vector<std::string> audio_paths;
    for (int i = 3; i < argc; ++i)
        audio_paths.push_back(argv[i]);

    std::cout << "Transcribing " << audio_paths.size()
              << " file(s) in batch...\n\n";

    auto results = t.transcribe_batch(audio_paths);

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[" << (i + 1) << "] " << audio_paths[i] << "\n"
                  << "    " << results[i].text << "\n\n";
    }

    return 0;
}
