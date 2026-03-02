// Basic transcription — simplest possible usage of parakeet.cpp
//
// Usage: example-basic <model.safetensors> <vocab.txt> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav>\n";
        return 1;
    }

    parakeet::Transcriber t(argv[1], argv[2]);
    auto result = t.transcribe(argv[3]);
    std::cout << result.text << std::endl;

    return 0;
}
