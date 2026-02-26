#include "parakeet/parakeet.hpp"

#include <axiom/io/numpy.hpp>
#include <axiom/io/safetensors.hpp>
#include <axiom/system.hpp>

#include <chrono>
#include <iomanip>
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
        << "  --gpu          Run on Metal GPU\n"
        << "  --timestamps   Show word-level timestamps\n"
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
        bool use_gpu = false;
        bool show_timestamps = false;
        std::string vocab_path;
        std::string features_path;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--ctc") {
                use_ctc = true;
            } else if (arg == "--tdt") {
                use_ctc = false;
            } else if (arg == "--gpu") {
                use_gpu = true;
            } else if (arg == "--timestamps") {
                show_timestamps = true;
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

        // Move model to GPU if requested
        if (use_gpu) {
            if (!axiom::system::is_metal_available()) {
                std::cerr << "Error: Metal GPU not available" << std::endl;
                return 1;
            }
            auto t_gpu = Clock::now();
            model.to(axiom::Device::GPU);
            auto t_gpu_done = Clock::now();
            std::cout << "Model moved to GPU ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             t_gpu_done - t_gpu)
                             .count()
                      << " ms)" << std::endl;
        }

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
        if (use_gpu) {
            features = features.gpu();
        }
        t0 = Clock::now();
        auto encoder_out = model.encoder()(features);
        t1 = Clock::now();

        // Force sync for timing (read a value to materialize GPU results)
        {
            auto tmp = encoder_out.cpu().ascontiguousarray();
            auto enc_shape = tmp.shape();
            std::cout << "  Encoder out: (" << enc_shape[0] << ", "
                      << enc_shape[1] << ", " << enc_shape[2] << ")"
                      << std::endl;
            // Print stats
            auto flat = tmp.flatten();
            size_t n = flat.shape()[0];
            const float *d = flat.typed_data<float>();
            float mn = d[0], mx = d[0], sum = 0;
            int nan_count = 0;
            for (size_t i = 0; i < n; ++i) {
                if (std::isnan(d[i])) {
                    nan_count++;
                    continue;
                }
                if (d[i] < mn)
                    mn = d[i];
                if (d[i] > mx)
                    mx = d[i];
                sum += d[i];
            }
            std::cout << "  Encoder stats: min=" << mn << " max=" << mx
                      << " mean=" << sum / static_cast<float>(n)
                      << " nans=" << nan_count << std::endl;
        }
        std::cout << "  Encoder: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                           t0)
                         .count()
                  << " ms" << std::endl;

        // 6. Decode
        t0 = Clock::now();
        std::vector<std::vector<int>> token_ids;
        std::vector<std::vector<TimestampedToken>> timestamped_tokens;

        if (use_ctc) {
            auto log_probs = model.ctc_decoder()(encoder_out);
            if (show_timestamps) {
                timestamped_tokens =
                    ctc_greedy_decode_with_timestamps(log_probs.cpu());
                token_ids.resize(timestamped_tokens.size());
                for (size_t b = 0; b < timestamped_tokens.size(); ++b) {
                    for (const auto &t : timestamped_tokens[b]) {
                        token_ids[b].push_back(t.token_id);
                    }
                }
            } else {
                token_ids = ctc_greedy_decode(log_probs.cpu());
            }
            std::cout << "  Decoder: CTC" << std::endl;
        } else {
            if (show_timestamps) {
                timestamped_tokens = tdt_greedy_decode_with_timestamps(
                    model, encoder_out, cfg.durations);
                token_ids.resize(timestamped_tokens.size());
                for (size_t b = 0; b < timestamped_tokens.size(); ++b) {
                    for (const auto &t : timestamped_tokens[b]) {
                        token_ids[b].push_back(t.token_id);
                    }
                }
            } else {
                token_ids =
                    tdt_greedy_decode(model, encoder_out, cfg.durations);
            }
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
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i > 0)
                        std::cout << " ";
                    std::cout << tokens[i];
                }
                std::cout << std::endl;
            }

            // Print timestamps if requested
            if (show_timestamps && b < timestamped_tokens.size() &&
                tokenizer.loaded()) {
                auto words = group_timestamps(timestamped_tokens[b],
                                              tokenizer.pieces());
                std::cout << "\n--- Word Timestamps ---" << std::endl;
                for (const auto &w : words) {
                    std::cout << "  [" << std::fixed << std::setprecision(2)
                              << w.start << "s - " << w.end << "s] " << w.word
                              << std::endl;
                }
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
