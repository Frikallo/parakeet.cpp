#include "parakeet/parakeet.hpp"

#include <axiom/io/safetensors.hpp>

#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    using namespace parakeet;

    if (argc < 2) {
        // No weights file — just print model info
        TDTCTCConfig cfg = make_110m_config();
        ParakeetTDTCTC model(cfg);

        std::cout << "ParakeetTDTCTC (110M config)" << std::endl;
        std::cout << "  Encoder layers:  " << cfg.encoder.num_layers
                  << std::endl;
        std::cout << "  Hidden size:     " << cfg.encoder.hidden_size
                  << std::endl;
        std::cout << "  TDT vocab size:  " << cfg.joint.vocab_size << std::endl;
        std::cout << "  CTC vocab size:  " << cfg.ctc_vocab_size << std::endl;
        std::cout << "  Pred LSTM layers:" << cfg.prediction.num_lstm_layers
                  << std::endl;
        std::cout << std::endl;

        // Dump named parameters for debugging name mapping
        auto params = model.named_parameters();
        std::cout << "Named parameters (" << params.size() << "):" << std::endl;
        for (const auto &[name, param] : params) {
            std::cout << "  " << name << std::endl;
        }

        return 0;
    }

    // Load weights from safetensors file
    std::string weights_path = argv[1];
    std::cout << "Loading weights from: " << weights_path << std::endl;

    TDTCTCConfig cfg = make_110m_config();
    ParakeetTDTCTC model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    std::cout << "Loaded " << weights.size() << " tensors from safetensors"
              << std::endl;

    // Non-strict: CTC decoder weights may not be in checkpoint
    model.load_state_dict(weights, /*prefix=*/"", /*strict=*/false);
    std::cout << "Successfully loaded state dict!" << std::endl;

    // Print parameter count
    size_t total_params = 0;
    for (const auto &[name, param] : weights) {
        size_t count = 1;
        for (auto dim : param.shape()) {
            count *= dim;
        }
        total_params += count;
    }
    std::cout << "Total parameters: " << total_params << std::endl;

    return 0;
}
