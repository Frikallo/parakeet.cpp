// Phrase boosting (context biasing) for domain-specific vocabulary
//
// Usage: example-phrase-boost <model.safetensors> <vocab.txt> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav>\n";
        return 1;
    }

    parakeet::Transcriber t(argv[1], argv[2]);

    // Without boosting
    auto baseline = t.transcribe(argv[3]);
    std::cout << "Baseline: " << baseline.text << "\n";

    // With phrase boosting — bias decoder toward specific terms
    parakeet::TranscribeOptions opts;
    opts.boost_phrases = {"Phoebe", "portrait"};
    opts.boost_score = 5.0f;
    auto boosted = t.transcribe(argv[3], opts);
    std::cout << "Boosted:  " << boosted.text << "\n";

    return 0;
}
