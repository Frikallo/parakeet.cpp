// Diarized transcription — ASR + Sortformer speaker attribution
//
// Usage: example-diarized-transcription <model.safetensors>
// <sortformer.safetensors> <vocab.txt> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <iomanip>
#include <iostream>
#include <map>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <model.safetensors> <sortformer.safetensors> <vocab.txt> "
               "<audio.wav>\n";
        return 1;
    }

    parakeet::DiarizedTranscriber dt(argv[1], argv[2], argv[3]);

    auto result = dt.transcribe(argv[4]);

    std::cout << "Full text: " << result.text << "\n\n";

    // Print speaker-grouped output
    std::cout << "Speaker-attributed transcript:\n\n";
    int prev_speaker = -2;
    for (const auto &w : result.words) {
        if (w.speaker_id != prev_speaker) {
            if (prev_speaker != -2)
                std::cout << "\n";
            std::cout << "Speaker " << w.speaker_id << " [" << std::fixed
                      << std::setprecision(2) << w.start << "s]: ";
            prev_speaker = w.speaker_id;
        }
        std::cout << w.word << " ";
    }
    std::cout << "\n\n";

    // Per-word detail
    std::cout << "Per-word detail:\n";
    for (const auto &w : result.words) {
        std::cout << "  Speaker " << w.speaker_id << " [" << std::fixed
                  << std::setprecision(2) << w.start << "s - " << w.end
                  << "s] (" << w.confidence << ") " << w.word << "\n";
    }

    return 0;
}
