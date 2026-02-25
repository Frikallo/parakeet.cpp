#include "parakeet/parakeet.hpp"

#include <iostream>

int main() {
    using namespace parakeet;

    // ── CTC ─────────────────────────────────────────────────────────────
    {
        CTCConfig cfg;
        ParakeetCTC model(cfg);
        std::cout << "ParakeetCTC" << std::endl;
        std::cout << "  Encoder layers: " << cfg.encoder.num_layers
                  << std::endl;
        std::cout << "  Hidden size:    " << cfg.encoder.hidden_size
                  << std::endl;
        std::cout << "  Vocab size:     " << cfg.vocab_size << std::endl;
        std::cout << std::endl;
    }

    // ── RNNT ────────────────────────────────────────────────────────────
    {
        RNNTConfig cfg;
        ParakeetRNNT model(cfg);
        std::cout << "ParakeetRNNT" << std::endl;
        std::cout << "  Encoder layers: " << cfg.encoder.num_layers
                  << std::endl;
        std::cout << "  Pred hidden:    " << cfg.prediction.pred_hidden
                  << std::endl;
        std::cout << "  Joint hidden:   " << cfg.joint.joint_hidden
                  << std::endl;
        std::cout << "  LSTM layers:    " << cfg.prediction.num_lstm_layers
                  << std::endl;
        std::cout << std::endl;
    }

    // ── TDT ─────────────────────────────────────────────────────────────
    {
        TDTConfig cfg;
        ParakeetTDT model(cfg);
        std::cout << "ParakeetTDT" << std::endl;
        std::cout << "  Encoder layers: " << cfg.encoder.num_layers
                  << std::endl;
        std::cout << "  Durations:      [";
        for (size_t i = 0; i < cfg.durations.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << cfg.durations[i];
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }

    // ── TDT-CTC Hybrid ─────────────────────────────────────────────────
    {
        TDTCTCConfig cfg;
        ParakeetTDTCTC model(cfg);
        std::cout << "ParakeetTDTCTC" << std::endl;
        std::cout << "  Encoder layers:  " << cfg.encoder.num_layers
                  << std::endl;
        std::cout << "  TDT vocab size:  " << cfg.joint.vocab_size << std::endl;
        std::cout << "  CTC vocab size:  " << cfg.ctc_vocab_size << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
