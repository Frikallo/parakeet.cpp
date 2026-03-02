// Beam search decoding with optional ARPA language model
//
// Usage: example-beam-search <model.safetensors> <vocab.txt> <audio.wav>
// [lm.arpa]

#include <parakeet/parakeet.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav> [lm.arpa]\n";
        return 1;
    }

    parakeet::Transcriber t(argv[1], argv[2]);

    // Greedy baseline
    auto greedy = t.transcribe(argv[3], parakeet::Decoder::TDT);
    std::cout << "Greedy TDT: " << greedy.text << "\n";

    // CTC beam search
    parakeet::TranscribeOptions ctc_opts;
    ctc_opts.decoder = parakeet::Decoder::CTC_BEAM;
    ctc_opts.beam_width = 16;
    if (argc == 5)
        ctc_opts.lm_path = argv[4];
    auto ctc_beam = t.transcribe(argv[3], ctc_opts);
    std::cout << "CTC beam:   " << ctc_beam.text << "\n";

    // TDT beam search
    parakeet::TranscribeOptions tdt_opts;
    tdt_opts.decoder = parakeet::Decoder::TDT_BEAM;
    tdt_opts.beam_width = 4;
    if (argc == 5) {
        tdt_opts.lm_path = argv[4];
        tdt_opts.lm_weight = 0.5f;
    }
    auto tdt_beam = t.transcribe(argv[3], tdt_opts);
    std::cout << "TDT beam:   " << tdt_beam.text << "\n";

    // TDT beam search with timestamps
    tdt_opts.timestamps = true;
    auto tdt_ts = t.transcribe(argv[3], tdt_opts);
    std::cout << "\nTDT beam with timestamps:\n";
    for (const auto &w : tdt_ts.word_timestamps) {
        std::cout << "  [" << w.start << "s - " << w.end << "s] " << w.word
                  << "\n";
    }

    return 0;
}
