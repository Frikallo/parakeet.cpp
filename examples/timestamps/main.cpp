// Word-level and token-level timestamps with confidence scores
//
// Usage: example-timestamps <model.safetensors> <vocab.txt> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav>\n";
        return 1;
    }

    parakeet::Transcriber t(argv[1], argv[2]);

    // TDT decoder with timestamps enabled
    auto result =
        t.transcribe(argv[3], parakeet::Decoder::TDT, /*timestamps=*/true);

    std::cout << "Text: " << result.text << "\n\n";

    // Word-level timestamps with confidence
    std::cout << "Word timestamps:\n";
    for (const auto &w : result.word_timestamps) {
        std::cout << "  [" << std::fixed << std::setprecision(2) << w.start
                  << "s - " << w.end << "s] (" << w.confidence << ") " << w.word
                  << "\n";
    }

    // Token-level timestamps
    std::cout << "\nToken timestamps (" << result.timestamped_tokens.size()
              << " tokens):\n";
    for (const auto &tok : result.timestamped_tokens) {
        std::cout << "  token=" << tok.token_id << " frames=["
                  << tok.start_frame << "-" << tok.end_frame
                  << "] conf=" << std::setprecision(3) << tok.confidence
                  << "\n";
    }

    return 0;
}
