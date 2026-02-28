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
        << "\nModel types:\n"
        << "  --model TYPE   Model type (default: tdt-ctc-110m)\n"
        << "                 Types: tdt-ctc-110m, tdt-600m, rnnt-600m,\n"
        << "                        eou-120m, nemotron-600m, sortformer,\n"
        << "                        diarized\n"
        << "\nDecoder options:\n"
        << "  --ctc          Use CTC decoder (default: TDT)\n"
        << "  --tdt          Use TDT decoder\n"
        << "\nOther options:\n"
        << "  --vocab PATH   SentencePiece vocab file for detokenization\n"
        << "  --sortformer-weights PATH  Sortformer weights (for diarized "
           "mode)\n"
        << "  --features PATH  Load pre-computed features from .npy file\n"
        << "  --gpu          Run on Metal GPU\n"
        << "  --timestamps   Show word-level timestamps\n"
        << "  --streaming    Use streaming mode (eou/nemotron models)\n"
        << "  --latency N    Right context frames for nemotron (0/1/6/13)\n"
        << std::endl;
}

// ─── TDT-CTC 110M mode (original) ───────────────────────────────────────────

static int run_tdt_ctc_110m(const std::string &weights_path,
                            const std::string &audio_path,
                            const std::string &vocab_path,
                            const std::string &features_path, bool use_ctc,
                            bool use_gpu, bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-ctc-110m" << std::endl;
    TDTCTCConfig cfg = make_110m_config();
    ParakeetTDTCTC model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_gpu) {
        auto t = Clock::now();
        model.to(axiom::Device::GPU);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - t)
                      .count();
        std::cout << "Model moved to GPU (" << dt << " ms)" << std::endl;
    }

    Tokenizer tokenizer;
    if (!vocab_path.empty()) {
        tokenizer.load(vocab_path);
        std::cout << "Vocab loaded (" << tokenizer.vocab_size() << " tokens)"
                  << std::endl;
    }

    auto audio = read_audio(audio_path);
    std::cout << "Audio: " << audio.original_sample_rate << "Hz"
              << (audio.original_sample_rate != audio.sample_rate
                      ? " -> " + std::to_string(audio.sample_rate) + "Hz"
                      : "")
              << ", " << audio.num_samples << " samples, " << std::fixed
              << std::setprecision(1) << audio.duration << "s" << std::endl;

    auto t0 = Clock::now();
    axiom::Tensor features;
    if (!features_path.empty()) {
        features = axiom::io::numpy::load(features_path);
    } else {
        features = preprocess_audio(audio.samples);
    }
    auto t1 = Clock::now();
    std::cout << "Preprocessing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    if (use_gpu)
        features = features.gpu();

    t0 = Clock::now();
    auto encoder_out = model.encoder()(features);
    t1 = Clock::now();
    std::cout << "Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    t0 = Clock::now();
    std::vector<std::vector<int>> token_ids;
    std::vector<std::vector<TimestampedToken>> timestamped_tokens;

    if (use_ctc) {
        auto log_probs = model.ctc_decoder()(encoder_out);
        if (show_timestamps) {
            timestamped_tokens =
                ctc_greedy_decode_with_timestamps(log_probs.cpu());
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids = ctc_greedy_decode(log_probs.cpu());
        }
        std::cout << "Decoder: CTC" << std::endl;
    } else {
        if (show_timestamps) {
            timestamped_tokens = tdt_greedy_decode_with_timestamps(
                model, encoder_out, cfg.durations);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids = tdt_greedy_decode(model, encoder_out, cfg.durations);
        }
        std::cout << "Decoder: TDT" << std::endl;
    }
    t1 = Clock::now();
    std::cout << "Decode: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    for (size_t b = 0; b < token_ids.size(); ++b) {
        const auto &tokens = token_ids[b];
        std::cout << "\n--- Transcription (" << tokens.size() << " tokens) ---"
                  << std::endl;
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

        if (show_timestamps && b < timestamped_tokens.size() &&
            tokenizer.loaded()) {
            auto words =
                group_timestamps(timestamped_tokens[b], tokenizer.pieces());
            std::cout << "\n--- Word Timestamps ---" << std::endl;
            for (const auto &w : words) {
                std::cout << "  [" << std::fixed << std::setprecision(2)
                          << w.start << "s - " << w.end << "s] ("
                          << w.confidence << ") " << w.word << std::endl;
            }
        }
    }
    return 0;
}

// ─── TDT 600M mode ──────────────────────────────────────────────────────────

static int run_tdt_600m(const std::string &weights_path,
                        const std::string &audio_path,
                        const std::string &vocab_path, bool use_gpu,
                        bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-600m" << std::endl;
    TDTConfig cfg = make_tdt_600m_config();
    ParakeetTDT model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_gpu) {
        model.to(axiom::Device::GPU);
        std::cout << "Model moved to GPU" << std::endl;
    }

    Tokenizer tokenizer;
    if (!vocab_path.empty()) {
        tokenizer.load(vocab_path);
    }

    auto audio = read_audio(audio_path);
    AudioConfig audio_cfg;
    audio_cfg.n_mels = cfg.encoder.mel_bins;
    auto features = preprocess_audio(audio.samples, audio_cfg);
    if (use_gpu)
        features = features.gpu();

    auto t0 = Clock::now();
    auto encoder_out = model.encoder()(features);
    auto t1 = Clock::now();
    std::cout << "Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    std::vector<std::vector<int>> token_ids;
    std::vector<std::vector<TimestampedToken>> timestamped_tokens;

    int blank_id = cfg.prediction.vocab_size - 1;

    if (show_timestamps) {
        timestamped_tokens = tdt_greedy_decode_with_timestamps(
            model, encoder_out, cfg.durations, blank_id);
        token_ids.resize(timestamped_tokens.size());
        for (size_t b = 0; b < timestamped_tokens.size(); ++b)
            for (const auto &t : timestamped_tokens[b])
                token_ids[b].push_back(t.token_id);
    } else {
        token_ids =
            tdt_greedy_decode(model, encoder_out, cfg.durations, blank_id);
    }

    for (size_t b = 0; b < token_ids.size(); ++b) {
        std::cout << "\n--- Transcription (" << token_ids[b].size()
                  << " tokens) ---" << std::endl;
        if (tokenizer.loaded()) {
            std::cout << tokenizer.decode(token_ids[b]) << std::endl;
        }
        if (show_timestamps && b < timestamped_tokens.size() &&
            tokenizer.loaded()) {
            auto words =
                group_timestamps(timestamped_tokens[b], tokenizer.pieces());
            std::cout << "\n--- Word Timestamps ---" << std::endl;
            for (const auto &w : words) {
                std::cout << "  [" << std::fixed << std::setprecision(2)
                          << w.start << "s - " << w.end << "s] ("
                          << w.confidence << ") " << w.word << std::endl;
            }
        }
    }
    return 0;
}

// ─── RNNT 600M mode ─────────────────────────────────────────────────────────

static int run_rnnt_600m(const std::string &weights_path,
                         const std::string &audio_path,
                         const std::string &vocab_path, bool use_gpu,
                         bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: rnnt-600m" << std::endl;
    RNNTConfig cfg = make_rnnt_600m_config();
    ParakeetRNNT model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_gpu) {
        model.to(axiom::Device::GPU);
        std::cout << "Model moved to GPU" << std::endl;
    }

    Tokenizer tokenizer;
    if (!vocab_path.empty()) {
        tokenizer.load(vocab_path);
    }

    auto audio = read_audio(audio_path);
    auto features = preprocess_audio(audio.samples);
    if (use_gpu)
        features = features.gpu();

    auto t0 = Clock::now();
    auto encoder_out = model.encoder()(features);
    auto t1 = Clock::now();
    std::cout << "Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    int blank_id = cfg.prediction.vocab_size - 1;

    std::vector<std::vector<int>> token_ids;
    std::vector<std::vector<TimestampedToken>> timestamped_tokens;

    t0 = Clock::now();
    if (show_timestamps) {
        timestamped_tokens =
            rnnt_greedy_decode_with_timestamps(model, encoder_out, blank_id);
        token_ids.resize(timestamped_tokens.size());
        for (size_t b = 0; b < timestamped_tokens.size(); ++b)
            for (const auto &t : timestamped_tokens[b])
                token_ids[b].push_back(t.token_id);
    } else {
        token_ids = rnnt_greedy_decode(model, encoder_out, blank_id);
    }
    t1 = Clock::now();
    std::cout << "Decoder: RNNT ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms)" << std::endl;

    for (size_t b = 0; b < token_ids.size(); ++b) {
        std::cout << "\n--- Transcription (" << token_ids[b].size()
                  << " tokens) ---" << std::endl;
        if (tokenizer.loaded()) {
            std::cout << tokenizer.decode(token_ids[b]) << std::endl;
        }
        if (show_timestamps && b < timestamped_tokens.size() &&
            tokenizer.loaded()) {
            auto words =
                group_timestamps(timestamped_tokens[b], tokenizer.pieces());
            std::cout << "\n--- Word Timestamps ---" << std::endl;
            for (const auto &w : words) {
                std::cout << "  [" << std::fixed << std::setprecision(2)
                          << w.start << "s - " << w.end << "s] ("
                          << w.confidence << ") " << w.word << std::endl;
            }
        }
    }
    return 0;
}

// ─── EOU Streaming mode ──────────────────────────────────────────────────────

static int run_eou_streaming(const std::string &weights_path,
                             const std::string &audio_path,
                             const std::string &vocab_path, bool use_gpu,
                             bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: eou-120m (streaming)" << std::endl;
    StreamingTranscriber transcriber(weights_path, vocab_path,
                                     make_eou_120m_config());
    if (use_gpu)
        transcriber.to_gpu();

    auto audio = read_audio(audio_path);
    std::cout << "Audio: " << audio.original_sample_rate << "Hz"
              << (audio.original_sample_rate != audio.sample_rate
                      ? " -> " + std::to_string(audio.sample_rate) + "Hz"
                      : "")
              << ", " << audio.num_samples << " samples, " << std::fixed
              << std::setprecision(1) << audio.duration << "s" << std::endl;

    // Simulate streaming: process in ~160ms chunks (2560 samples at 16kHz)
    constexpr int CHUNK_SAMPLES = 2560;
    auto samples = audio.samples.ascontiguousarray();
    int total = static_cast<int>(samples.shape()[0]);
    const float *data = samples.typed_data<float>();

    std::string full_text;
    auto t0 = Clock::now();
    for (int offset = 0; offset < total; offset += CHUNK_SAMPLES) {
        int chunk_len = std::min(CHUNK_SAMPLES, total - offset);
        auto chunk = axiom::Tensor::from_data(
            data + offset, axiom::Shape{static_cast<size_t>(chunk_len)}, true);
        auto text = transcriber.transcribe_chunk(chunk);
        if (!text.empty()) {
            std::cout << text << std::flush;
            full_text += text;
        }
    }
    auto t1 = Clock::now();

    std::cout << "\n\n--- Full Transcription ---" << std::endl;
    std::cout << transcriber.get_text() << std::endl;
    std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    if (show_timestamps && transcriber.tokenizer().loaded()) {
        auto words = group_timestamps(transcriber.get_timestamped_tokens(),
                                      transcriber.tokenizer().pieces());
        std::cout << "\n--- Word Timestamps ---" << std::endl;
        for (const auto &w : words) {
            std::cout << "  [" << std::fixed << std::setprecision(2) << w.start
                      << "s - " << w.end << "s] (" << w.confidence << ") "
                      << w.word << std::endl;
        }
    }
    return 0;
}

// ─── Nemotron Streaming mode ─────────────────────────────────────────────────

static int run_nemotron_streaming(const std::string &weights_path,
                                  const std::string &audio_path,
                                  const std::string &vocab_path, bool use_gpu,
                                  bool show_timestamps, int latency_frames) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: nemotron-600m (streaming, latency="
              << latency_frames << " frames)" << std::endl;
    NemotronTranscriber transcriber(weights_path, vocab_path,
                                    make_nemotron_600m_config(latency_frames));
    if (use_gpu)
        transcriber.to_gpu();

    auto audio = read_audio(audio_path);

    constexpr int CHUNK_SAMPLES = 2560;
    auto samples = audio.samples.ascontiguousarray();
    int total = static_cast<int>(samples.shape()[0]);
    const float *data = samples.typed_data<float>();

    auto t0 = Clock::now();
    for (int offset = 0; offset < total; offset += CHUNK_SAMPLES) {
        int chunk_len = std::min(CHUNK_SAMPLES, total - offset);
        auto chunk = axiom::Tensor::from_data(
            data + offset, axiom::Shape{static_cast<size_t>(chunk_len)}, true);
        auto text = transcriber.transcribe_chunk(chunk);
        if (!text.empty()) {
            std::cout << text << std::flush;
        }
    }
    auto t1 = Clock::now();

    std::cout << "\n\n--- Full Transcription ---" << std::endl;
    std::cout << transcriber.get_text() << std::endl;
    std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    if (show_timestamps && transcriber.tokenizer().loaded()) {
        auto words = group_timestamps(transcriber.get_timestamped_tokens(),
                                      transcriber.tokenizer().pieces());
        std::cout << "\n--- Word Timestamps ---" << std::endl;
        for (const auto &w : words) {
            std::cout << "  [" << std::fixed << std::setprecision(2) << w.start
                      << "s - " << w.end << "s] (" << w.confidence << ") "
                      << w.word << std::endl;
        }
    }
    return 0;
}

// ─── Sortformer Diarization mode ─────────────────────────────────────────────

static int run_sortformer(const std::string &weights_path,
                          const std::string &audio_path, bool use_gpu) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: sortformer" << std::endl;
    SortformerConfig cfg = make_sortformer_117m_config();
    Sortformer model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);

    if (use_gpu) {
        model.to(axiom::Device::GPU);
    }

    auto audio = read_audio(audio_path);
    AudioConfig audio_cfg;
    audio_cfg.n_mels = cfg.nest_encoder.mel_bins;
    audio_cfg.normalize = false; // sortformer uses normalize: NA
    auto features = preprocess_audio(audio.samples, audio_cfg);
    if (use_gpu)
        features = features.gpu();

    auto t0 = Clock::now();
    auto segments = model.diarize(features);
    auto t1 = Clock::now();

    std::cout << "Diarization: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    std::cout << "\n--- Speaker Segments (" << segments.size()
              << " segments) ---" << std::endl;
    for (const auto &seg : segments) {
        std::cout << "  Speaker " << seg.speaker_id << ": [" << std::fixed
                  << std::setprecision(2) << seg.start << "s - " << seg.end
                  << "s]" << std::endl;
    }
    return 0;
}

// ─── Diarized Transcription mode ─────────────────────────────────────────────

static int run_diarized(const std::string &weights_path,
                        const std::string &audio_path,
                        const std::string &vocab_path,
                        const std::string &sortformer_weights_path,
                        bool use_ctc, bool use_gpu) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    if (sortformer_weights_path.empty()) {
        std::cerr << "Error: --sortformer-weights required for diarized mode"
                  << std::endl;
        return 1;
    }
    if (vocab_path.empty()) {
        std::cerr << "Error: --vocab required for diarized mode" << std::endl;
        return 1;
    }

    std::cout << "Loading diarized transcriber (ASR + Sortformer)" << std::endl;
    auto t0 = Clock::now();
    DiarizedTranscriber dt(weights_path, sortformer_weights_path, vocab_path);
    auto t1 = Clock::now();
    std::cout << "Models loaded ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms)" << std::endl;

    if (use_gpu) {
        t0 = Clock::now();
        dt.to_gpu();
        t1 = Clock::now();
        std::cout << "Models moved to GPU ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                           t0)
                         .count()
                  << " ms)" << std::endl;
    }

    auto audio = read_audio(audio_path);
    std::cout << "Audio: " << audio.original_sample_rate << "Hz"
              << (audio.original_sample_rate != audio.sample_rate
                      ? " -> " + std::to_string(audio.sample_rate) + "Hz"
                      : "")
              << ", " << audio.num_samples << " samples, " << std::fixed
              << std::setprecision(1) << audio.duration << "s" << std::endl;

    Decoder decoder = use_ctc ? Decoder::CTC : Decoder::TDT;
    t0 = Clock::now();
    auto result = dt.transcribe(audio.samples, decoder);
    t1 = Clock::now();
    std::cout << "Transcription + diarization: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    // Print speaker-grouped output
    std::cout << "\n--- Diarized Transcription ---" << std::endl;
    int current_speaker = -2; // sentinel
    for (const auto &w : result.words) {
        if (w.speaker_id != current_speaker) {
            if (current_speaker != -2)
                std::cout << std::endl;
            if (w.speaker_id >= 0) {
                std::cout << "Speaker " << w.speaker_id << ": ";
            } else {
                std::cout << "Unknown: ";
            }
            current_speaker = w.speaker_id;
        } else {
            std::cout << " ";
        }
        std::cout << w.word;
    }
    if (!result.words.empty())
        std::cout << std::endl;

    // Print detailed word list
    std::cout << "\n--- Diarized Words ---" << std::endl;
    for (const auto &w : result.words) {
        std::cout << "  [" << std::fixed << std::setprecision(2) << w.start
                  << "s - " << w.end << "s] (spk=" << w.speaker_id
                  << ", conf=" << w.confidence << ") " << w.word << std::endl;
    }

    // Print speaker segments
    std::cout << "\n--- Speaker Segments (" << result.segments.size()
              << " segments) ---" << std::endl;
    for (const auto &seg : result.segments) {
        std::cout << "  Speaker " << seg.speaker_id << ": [" << std::fixed
                  << std::setprecision(2) << seg.start << "s - " << seg.end
                  << "s]" << std::endl;
    }

    return 0;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string weights_path = argv[1];
        std::string audio_path = argv[2];
        std::string model_type = "tdt-ctc-110m";
        bool use_ctc = false;
        bool use_gpu = false;
        bool show_timestamps = false;
        bool streaming = false;
        int latency_frames = 0;
        std::string vocab_path;
        std::string features_path;
        std::string sortformer_weights_path;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                model_type = argv[++i];
            } else if (arg == "--ctc") {
                use_ctc = true;
            } else if (arg == "--tdt") {
                use_ctc = false;
            } else if (arg == "--gpu") {
                use_gpu = true;
            } else if (arg == "--timestamps") {
                show_timestamps = true;
            } else if (arg == "--streaming") {
                streaming = true;
            } else if (arg == "--latency" && i + 1 < argc) {
                latency_frames = std::stoi(argv[++i]);
            } else if (arg == "--vocab" && i + 1 < argc) {
                vocab_path = argv[++i];
            } else if (arg == "--features" && i + 1 < argc) {
                features_path = argv[++i];
            } else if (arg == "--sortformer-weights" && i + 1 < argc) {
                sortformer_weights_path = argv[++i];
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        }

        if (use_gpu && !axiom::system::is_metal_available()) {
            std::cerr << "Error: Metal GPU not available" << std::endl;
            return 1;
        }

        if (model_type == "tdt-ctc-110m") {
            return run_tdt_ctc_110m(weights_path, audio_path, vocab_path,
                                    features_path, use_ctc, use_gpu,
                                    show_timestamps);
        } else if (model_type == "tdt-600m") {
            return run_tdt_600m(weights_path, audio_path, vocab_path, use_gpu,
                                show_timestamps);
        } else if (model_type == "rnnt-600m") {
            return run_rnnt_600m(weights_path, audio_path, vocab_path, use_gpu,
                                 show_timestamps);
        } else if (model_type == "eou-120m") {
            return run_eou_streaming(weights_path, audio_path, vocab_path,
                                     use_gpu, show_timestamps);
        } else if (model_type == "nemotron-600m") {
            return run_nemotron_streaming(weights_path, audio_path, vocab_path,
                                          use_gpu, show_timestamps,
                                          latency_frames);
        } else if (model_type == "sortformer") {
            return run_sortformer(weights_path, audio_path, use_gpu);
        } else if (model_type == "diarized") {
            return run_diarized(weights_path, audio_path, vocab_path,
                                sortformer_weights_path, use_ctc, use_gpu);
        } else {
            std::cerr << "Unknown model type: " << model_type << std::endl;
            print_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
