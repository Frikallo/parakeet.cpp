#include "parakeet/parakeet.hpp"

#include <axiom/io/numpy.hpp>
#include <axiom/io/safetensors.hpp>

#include <chrono>
#include <iostream>
#include <string>

static void print_usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " <model.safetensors> <audio.wav> [options]\n"
        << "\nOptions:\n"
        << "  --ctc          Use CTC decoder (default: TDT)\n"
        << "  --tdt          Use TDT decoder\n"
        << "  --vocab PATH   SentencePiece vocab file for detokenization\n"
        << "  --features PATH  Load pre-computed features from .npy file\n"
        << std::endl;
}

int main(int argc, char *argv[]) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {

        // Parse arguments
        std::string weights_path = argv[1];
        std::string audio_path = argv[2];
        bool use_ctc = false;
        std::string vocab_path;
        std::string features_path;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--ctc") {
                use_ctc = true;
            } else if (arg == "--tdt") {
                use_ctc = false;
            } else if (arg == "--vocab" && i + 1 < argc) {
                vocab_path = argv[++i];
            } else if (arg == "--features" && i + 1 < argc) {
                features_path = argv[++i];
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }

        // 1. Load model
        std::cout << "Loading model from: " << weights_path << std::endl;
        TDTCTCConfig cfg = make_110m_config();
        ParakeetTDTCTC model(cfg);

        auto weights = axiom::io::safetensors::load(weights_path);
        model.load_state_dict(weights, /*prefix=*/"", /*strict=*/false);
        std::cout << "Model loaded (" << weights.size() << " tensors)"
                  << std::endl;

        // 2. Load vocab (optional)
        Tokenizer tokenizer;
        if (!vocab_path.empty()) {
            tokenizer.load(vocab_path);
            std::cout << "Vocab loaded (" << tokenizer.vocab_size()
                      << " tokens)" << std::endl;
        }

        // 3. Read WAV
        std::cout << "Reading audio: " << audio_path << std::endl;
        auto wav = read_wav(audio_path);
        std::cout << "  Sample rate: " << wav.sample_rate
                  << ", channels: " << wav.num_channels
                  << ", samples: " << wav.num_samples << std::endl;

        if (wav.sample_rate != 16000) {
            std::cerr << "Warning: expected 16kHz audio, got "
                      << wav.sample_rate << "Hz" << std::endl;
        }

        // 4. Preprocess audio (or load from file)
        auto t0 = Clock::now();
        axiom::Tensor features;
        if (!features_path.empty()) {
            features = axiom::io::numpy::load(features_path);
            std::cout << "  Loaded features from: " << features_path << " "
                      << features.shape() << std::endl;
        } else {
            features = preprocess_audio(wav.samples);
        }
        auto t1 = Clock::now();

        auto feat_shape = features.shape();
        std::cout << "  Features: (" << feat_shape[0] << ", " << feat_shape[1]
                  << ", " << feat_shape[2] << ")" << std::endl;
        std::cout << "  Preprocessing: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        // 5. Encoder forward pass
        t0 = Clock::now();
        auto encoder_out = model.encoder()(features);
        t1 = Clock::now();

        auto enc_shape = encoder_out.shape();
        std::cout << "  Encoder out: (" << enc_shape[0] << ", " << enc_shape[1]
                  << ", " << enc_shape[2] << ")" << std::endl;
        std::cout << "  Encoder: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        // 6. Decode
        t0 = Clock::now();
        std::vector<std::vector<int>> token_ids;

        if (use_ctc) {
            auto log_probs = model.ctc_decoder()(encoder_out);
            token_ids = ctc_greedy_decode(log_probs);
            std::cout << "  Decoder: CTC" << std::endl;
        } else {
            token_ids = tdt_greedy_decode(model, encoder_out, cfg.durations);
            std::cout << "  Decoder: TDT" << std::endl;
        }
        t1 = Clock::now();

        std::cout << "  Decode: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        // 7. Detokenize and print
        for (size_t b = 0; b < token_ids.size(); ++b) {
            const auto &tokens = token_ids[b];
            std::cout << "\n--- Transcription";
            if (token_ids.size() > 1) {
                std::cout << " [" << b << "]";
            }
            std::cout << " (" << tokens.size() << " tokens) ---" << std::endl;

            if (tokenizer.loaded()) {
                std::cout << tokenizer.decode(tokens) << std::endl;
            } else {
                // Print raw token IDs
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i > 0)
                        std::cout << " ";
                    std::cout << tokens[i];
                }
                std::cout << std::endl;
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
