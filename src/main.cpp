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
        << "Usage: " << prog
        << " <model.safetensors> <audio.wav> [audio2.wav ...] [options]\n"
        << "\nModel types:\n"
        << "  --model TYPE   Model type (default: tdt-ctc-110m)\n"
        << "                 Types: tdt-ctc-110m, tdt-600m, rnnt-600m,\n"
        << "                        eou-120m, nemotron-600m, sortformer,\n"
        << "                        diarized\n"
        << "\nDecoder options:\n"
        << "  --ctc          Use CTC greedy decoder (default: TDT)\n"
        << "  --tdt          Use TDT decoder\n"
        << "  --ctc-beam     Use CTC beam search decoder\n"
        << "  --tdt-beam     Use TDT beam search decoder\n"
        << "  --beam-width N Beam width for beam search (default: 8)\n"
        << "  --lm PATH      ARPA language model for beam search\n"
        << "  --lm-weight N  LM interpolation weight (default: 0.5)\n"
        << "\nPhrase boost:\n"
        << "  --boost PHRASE   Boost a phrase (repeatable)\n"
        << "  --boost-score N  Boost score (default: 5.0)\n"
        << "\nOther options:\n"
        << "  --vocab PATH   SentencePiece vocab file for detokenization\n"
        << "  --sortformer-weights PATH  Sortformer weights (for diarized "
           "mode)\n"
        << "  --features PATH  Load pre-computed features from .npy file\n"
        << "  --gpu          Run on Metal GPU\n"
        << "  --fp16         Use half-precision (fp16) inference\n"
        << "  --timestamps   Show word-level timestamps\n"
        << "  --streaming    Use streaming mode (eou/nemotron models)\n"
        << "  --latency N    Right context frames for nemotron (0/1/6/13)\n"
        << "  --vad PATH     Enable Silero VAD with given weights file\n"
        << "  --vad-threshold F  VAD speech threshold (default: 0.5)\n"
        << "\nBatch mode:\n"
        << "  Multiple audio files can be specified for batch inference.\n"
        << "  Supported for tdt-ctc-110m and tdt-600m models.\n"
        << std::endl;
}

// ─── TDT-CTC 110M mode (original) ───────────────────────────────────────────

static int run_tdt_ctc_110m(const std::string &weights_path,
                            const std::string &audio_path,
                            const std::string &vocab_path,
                            const std::string &features_path, bool use_ctc,
                            bool use_gpu, bool use_fp16, bool show_timestamps,
                            const std::vector<std::string> &boost_phrases = {},
                            float boost_score = 5.0f, bool use_ctc_beam = false,
                            int beam_width = 8, const std::string &lm_path = "",
                            float lm_weight = 0.5f, bool use_tdt_beam = false,
                            const std::string &vad_weights_path = {},
                            float vad_threshold = 0.5f) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-ctc-110m" << std::endl;
    TDTCTCConfig cfg = make_110m_config();
    ParakeetTDTCTC model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_fp16) {
        model.to(axiom::DType::Float16);
        std::cout << "Model cast to fp16" << std::endl;
    }

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

    // VAD preprocessing
    axiom::Tensor audio_samples = audio.samples;
    std::unique_ptr<parakeet::audio::TimestampRemapper> remapper;
    if (!vad_weights_path.empty()) {
        parakeet::audio::SileroVAD silero_vad(vad_weights_path);
        parakeet::audio::VADConfig vad_cfg;
        vad_cfg.threshold = vad_threshold;
        vad_cfg.neg_threshold = vad_threshold - 0.15f;
        auto segments =
            silero_vad.detect(audio.samples, audio.sample_rate, vad_cfg);
        if (!segments.empty()) {
            float speech_dur = 0;
            for (const auto &s : segments)
                speech_dur +=
                    static_cast<float>(s.end_sample - s.start_sample) /
                    audio.sample_rate;
            std::cout << "VAD: " << segments.size() << " speech segments, "
                      << std::fixed << std::setprecision(1) << speech_dur
                      << "s speech / " << audio.duration << "s total ("
                      << static_cast<int>(100 *
                                          (1 - speech_dur / audio.duration))
                      << "% filtered)" << std::endl;
            audio_samples =
                parakeet::audio::collect_speech(audio.samples, segments);
            if (show_timestamps) {
                remapper = std::make_unique<parakeet::audio::TimestampRemapper>(
                    segments);
            }
        }
    }

    auto t0 = Clock::now();
    axiom::Tensor features;
    if (!features_path.empty()) {
        features = axiom::io::numpy::load(features_path);
    } else {
        features = preprocess_audio(audio_samples);
    }
    auto t1 = Clock::now();
    std::cout << "Preprocessing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    if (use_fp16)
        features = features.half();
    if (use_gpu)
        features = features.gpu();

    t0 = Clock::now();
    auto encoder_out = model.encoder()(features);
    t1 = Clock::now();
    std::cout << "Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    // Build phrase boost trie if needed
    ContextTrie trie;
    bool use_boost = !boost_phrases.empty() && tokenizer.loaded();
    if (use_boost) {
        trie.build(boost_phrases, tokenizer);
        std::cout << "Phrase boost: " << boost_phrases.size() << " phrases, "
                  << trie.size() << " trie nodes" << std::endl;
    }

    // Load LM if beam search with LM requested
    ArpaLM arpa_lm;
    if ((use_ctc_beam || use_tdt_beam) && !lm_path.empty()) {
        arpa_lm.load(lm_path);
        std::cout << "LM loaded: " << arpa_lm.size() << " n-grams, order "
                  << arpa_lm.order() << std::endl;
    }

    t0 = Clock::now();
    std::vector<std::vector<int>> token_ids;
    std::vector<std::vector<TimestampedToken>> timestamped_tokens;

    if (use_tdt_beam) {
        TDTBeamSearchOptions tdt_bs_opts;
        tdt_bs_opts.beam_width = beam_width;
        tdt_bs_opts.lm = arpa_lm.loaded() ? &arpa_lm : nullptr;
        tdt_bs_opts.lm_weight = lm_weight;
        tdt_bs_opts.pieces = tokenizer.loaded() ? &tokenizer.pieces() : nullptr;
        if (show_timestamps) {
            timestamped_tokens = tdt_beam_decode_with_timestamps(
                model, encoder_out, cfg.durations, tdt_bs_opts);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids =
                tdt_beam_decode(model, encoder_out, cfg.durations, tdt_bs_opts);
        }
        std::cout << "Decoder: TDT beam (width=" << beam_width
                  << (arpa_lm.loaded() ? ", +LM" : "") << ")" << std::endl;
    } else if (use_ctc_beam) {
        auto log_probs = model.ctc_decoder()(encoder_out);
        auto cpu_lp = log_probs.cpu();
        BeamSearchOptions bs_opts;
        bs_opts.beam_width = beam_width;
        bs_opts.lm = arpa_lm.loaded() ? &arpa_lm : nullptr;
        bs_opts.lm_weight = lm_weight;
        bs_opts.pieces = tokenizer.loaded() ? &tokenizer.pieces() : nullptr;
        if (show_timestamps) {
            timestamped_tokens =
                ctc_beam_decode_with_timestamps(cpu_lp, bs_opts);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids = ctc_beam_decode(cpu_lp, bs_opts);
        }
        std::cout << "Decoder: CTC beam (width=" << beam_width
                  << (arpa_lm.loaded() ? ", +LM" : "") << ")" << std::endl;
    } else if (use_ctc) {
        auto log_probs = model.ctc_decoder()(encoder_out);
        auto cpu_lp = log_probs.cpu();
        if (show_timestamps) {
            timestamped_tokens =
                use_boost ? ctc_greedy_decode_with_timestamps_boosted(
                                cpu_lp, trie, boost_score)
                          : ctc_greedy_decode_with_timestamps(cpu_lp);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids =
                use_boost ? ctc_greedy_decode_boosted(cpu_lp, trie, boost_score)
                          : ctc_greedy_decode(cpu_lp);
        }
        std::cout << "Decoder: CTC" << (use_boost ? " (boosted)" : "")
                  << std::endl;
    } else {
        if (show_timestamps) {
            timestamped_tokens =
                use_boost
                    ? tdt_greedy_decode_with_timestamps_boosted(
                          model, encoder_out, cfg.durations, trie, boost_score)
                    : tdt_greedy_decode_with_timestamps(model, encoder_out,
                                                        cfg.durations);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids =
                use_boost
                    ? tdt_greedy_decode_boosted(
                          model, encoder_out, cfg.durations, trie, boost_score)
                    : tdt_greedy_decode(model, encoder_out, cfg.durations);
        }
        std::cout << "Decoder: TDT" << (use_boost ? " (boosted)" : "")
                  << std::endl;
    }
    t1 = Clock::now();
    std::cout << "Decode: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    // Remap timestamps if VAD was used
    if (remapper && show_timestamps) {
        for (auto &ts_vec : timestamped_tokens) {
            ts_vec = remapper->remap_tokens(ts_vec);
        }
    }

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

// ─── TDT-CTC 110M batch mode ────────────────────────────────────────────────

static int run_tdt_ctc_110m_batch(
    const std::string &weights_path,
    const std::vector<std::string> &audio_paths, const std::string &vocab_path,
    bool use_ctc, bool use_gpu, bool use_fp16, bool show_timestamps,
    const std::vector<std::string> &boost_phrases = {},
    float boost_score = 5.0f, const std::string &vad_weights_path = {},
    float vad_threshold = 0.5f) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-ctc-110m (batch mode, "
              << audio_paths.size() << " files)" << std::endl;
    Transcriber transcriber(weights_path, vocab_path);
    if (use_fp16)
        transcriber.to_half();
    if (use_gpu)
        transcriber.to_gpu();
    if (!vad_weights_path.empty()) {
        transcriber.enable_vad(vad_weights_path);
        std::cout << "VAD enabled (threshold=" << vad_threshold << ")"
                  << std::endl;
    }

    TranscribeOptions opts;
    opts.decoder = use_ctc ? Decoder::CTC : Decoder::TDT;
    opts.timestamps = show_timestamps;
    opts.boost_phrases = boost_phrases;
    opts.boost_score = boost_score;
    opts.use_vad = !vad_weights_path.empty();

    auto t0 = Clock::now();
    auto results = transcriber.transcribe_batch(audio_paths, opts);
    auto t1 = Clock::now();

    auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Batch transcription: " << total_ms << " ms ("
              << audio_paths.size() << " files)" << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "\n--- [" << (i + 1) << "/" << results.size() << "] "
                  << audio_paths[i] << " (" << results[i].token_ids.size()
                  << " tokens) ---" << std::endl;
        std::cout << results[i].text << std::endl;

        if (show_timestamps && !results[i].word_timestamps.empty()) {
            std::cout << "\n--- Word Timestamps ---" << std::endl;
            for (const auto &w : results[i].word_timestamps) {
                std::cout << "  [" << std::fixed << std::setprecision(2)
                          << w.start << "s - " << w.end << "s] ("
                          << w.confidence << ") " << w.word << std::endl;
            }
        }
    }
    return 0;
}

// ─── TDT 600M mode ──────────────────────────────────────────────────────────

static int run_tdt_600m(
    const std::string &weights_path, const std::string &audio_path,
    const std::string &vocab_path, bool use_gpu, bool use_fp16,
    bool show_timestamps, const std::vector<std::string> &boost_phrases = {},
    float boost_score = 5.0f, bool use_tdt_beam = false, int beam_width = 4,
    const std::string &lm_path = "", float lm_weight = 0.5f,
    const std::string &vad_weights_path = {}, float vad_threshold = 0.5f) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-600m" << std::endl;
    TDTConfig cfg = make_tdt_600m_config();
    ParakeetTDT model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_fp16) {
        model.to(axiom::DType::Float16);
        std::cout << "Model cast to fp16" << std::endl;
    }

    if (use_gpu) {
        model.to(axiom::Device::GPU);
        std::cout << "Model moved to GPU" << std::endl;
    }

    Tokenizer tokenizer;
    if (!vocab_path.empty()) {
        tokenizer.load(vocab_path);
    }

    auto audio = read_audio(audio_path);

    // VAD preprocessing
    axiom::Tensor audio_samples = audio.samples;
    std::unique_ptr<parakeet::audio::TimestampRemapper> remapper;
    if (!vad_weights_path.empty()) {
        parakeet::audio::SileroVAD silero_vad(vad_weights_path);
        parakeet::audio::VADConfig vad_cfg;
        vad_cfg.threshold = vad_threshold;
        vad_cfg.neg_threshold = vad_threshold - 0.15f;
        auto segments =
            silero_vad.detect(audio.samples, audio.sample_rate, vad_cfg);
        if (!segments.empty()) {
            float speech_dur = 0;
            for (const auto &s : segments)
                speech_dur +=
                    static_cast<float>(s.end_sample - s.start_sample) /
                    audio.sample_rate;
            std::cout << "VAD: " << segments.size() << " speech segments, "
                      << std::fixed << std::setprecision(1) << speech_dur
                      << "s speech / " << audio.duration << "s total ("
                      << static_cast<int>(100 *
                                          (1 - speech_dur / audio.duration))
                      << "% filtered)" << std::endl;
            audio_samples =
                parakeet::audio::collect_speech(audio.samples, segments);
            if (show_timestamps) {
                remapper = std::make_unique<parakeet::audio::TimestampRemapper>(
                    segments);
            }
        }
    }

    AudioConfig audio_cfg;
    audio_cfg.n_mels = cfg.encoder.mel_bins;
    auto features = preprocess_audio(audio_samples, audio_cfg);
    if (use_fp16)
        features = features.half();
    if (use_gpu)
        features = features.gpu();

    auto t0 = Clock::now();
    auto encoder_out = model.encoder()(features);
    auto t1 = Clock::now();
    std::cout << "Encoder: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << " ms" << std::endl;

    // Build phrase boost trie if needed
    ContextTrie trie;
    bool use_boost = !boost_phrases.empty() && tokenizer.loaded();
    if (use_boost) {
        trie.build(boost_phrases, tokenizer);
        std::cout << "Phrase boost: " << boost_phrases.size() << " phrases, "
                  << trie.size() << " trie nodes" << std::endl;
    }

    // Load LM if beam search with LM requested
    ArpaLM arpa_lm;
    if (use_tdt_beam && !lm_path.empty()) {
        arpa_lm.load(lm_path);
        std::cout << "LM loaded: " << arpa_lm.size() << " n-grams, order "
                  << arpa_lm.order() << std::endl;
    }

    std::vector<std::vector<int>> token_ids;
    std::vector<std::vector<TimestampedToken>> timestamped_tokens;

    int blank_id = cfg.prediction.vocab_size - 1;

    if (use_tdt_beam) {
        TDTBeamSearchOptions tdt_bs_opts;
        tdt_bs_opts.beam_width = beam_width;
        tdt_bs_opts.blank_id = blank_id;
        tdt_bs_opts.lm = arpa_lm.loaded() ? &arpa_lm : nullptr;
        tdt_bs_opts.lm_weight = lm_weight;
        tdt_bs_opts.pieces = tokenizer.loaded() ? &tokenizer.pieces() : nullptr;
        if (show_timestamps) {
            timestamped_tokens = tdt_beam_decode_with_timestamps(
                model, encoder_out, cfg.durations, tdt_bs_opts);
            token_ids.resize(timestamped_tokens.size());
            for (size_t b = 0; b < timestamped_tokens.size(); ++b)
                for (const auto &t : timestamped_tokens[b])
                    token_ids[b].push_back(t.token_id);
        } else {
            token_ids =
                tdt_beam_decode(model, encoder_out, cfg.durations, tdt_bs_opts);
        }
        std::cout << "Decoder: TDT beam (width=" << beam_width
                  << (arpa_lm.loaded() ? ", +LM" : "") << ")" << std::endl;
    } else if (show_timestamps) {
        timestamped_tokens =
            use_boost ? tdt_greedy_decode_with_timestamps_boosted(
                            model, encoder_out, cfg.durations, trie,
                            boost_score, blank_id)
                      : tdt_greedy_decode_with_timestamps(
                            model, encoder_out, cfg.durations, blank_id);
        token_ids.resize(timestamped_tokens.size());
        for (size_t b = 0; b < timestamped_tokens.size(); ++b)
            for (const auto &t : timestamped_tokens[b])
                token_ids[b].push_back(t.token_id);
    } else {
        token_ids = use_boost ? tdt_greedy_decode_boosted(model, encoder_out,
                                                          cfg.durations, trie,
                                                          boost_score, blank_id)
                              : tdt_greedy_decode(model, encoder_out,
                                                  cfg.durations, blank_id);
    }

    // Remap timestamps if VAD was used
    if (remapper && show_timestamps) {
        for (auto &ts_vec : timestamped_tokens) {
            ts_vec = remapper->remap_tokens(ts_vec);
        }
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

// ─── TDT 600M batch mode ────────────────────────────────────────────────────

static int run_tdt_600m_batch(
    const std::string &weights_path,
    const std::vector<std::string> &audio_paths, const std::string &vocab_path,
    bool use_gpu, bool use_fp16, bool show_timestamps,
    const std::vector<std::string> &boost_phrases = {},
    float boost_score = 5.0f, const std::string &vad_weights_path = {},
    float vad_threshold = 0.5f) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: tdt-600m (batch mode, " << audio_paths.size()
              << " files)" << std::endl;
    TDTTranscriber transcriber(weights_path, vocab_path);
    if (use_fp16)
        transcriber.to_half();
    if (use_gpu)
        transcriber.to_gpu();
    if (!vad_weights_path.empty()) {
        transcriber.enable_vad(vad_weights_path);
        std::cout << "VAD enabled (threshold=" << vad_threshold << ")"
                  << std::endl;
    }

    TranscribeOptions opts;
    opts.decoder = Decoder::TDT;
    opts.timestamps = show_timestamps;
    opts.boost_phrases = boost_phrases;
    opts.boost_score = boost_score;
    opts.use_vad = !vad_weights_path.empty();

    auto t0 = Clock::now();
    auto results = transcriber.transcribe_batch(audio_paths, opts);
    auto t1 = Clock::now();

    auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Batch transcription: " << total_ms << " ms ("
              << audio_paths.size() << " files)" << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "\n--- [" << (i + 1) << "/" << results.size() << "] "
                  << audio_paths[i] << " (" << results[i].token_ids.size()
                  << " tokens) ---" << std::endl;
        std::cout << results[i].text << std::endl;

        if (show_timestamps && !results[i].word_timestamps.empty()) {
            std::cout << "\n--- Word Timestamps ---" << std::endl;
            for (const auto &w : results[i].word_timestamps) {
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
                         bool use_fp16, bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: rnnt-600m" << std::endl;
    RNNTConfig cfg = make_rnnt_600m_config();
    ParakeetRNNT model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);
    std::cout << "Model loaded (" << weights.size() << " tensors)" << std::endl;

    if (use_fp16) {
        model.to(axiom::DType::Float16);
        std::cout << "Model cast to fp16" << std::endl;
    }

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
    if (use_fp16)
        features = features.half();
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
                             bool use_fp16, bool show_timestamps) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: eou-120m (streaming)" << std::endl;
    StreamingTranscriber transcriber(weights_path, vocab_path,
                                     make_eou_120m_config());
    if (use_fp16)
        transcriber.to_half();
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
                                  bool use_fp16, bool show_timestamps,
                                  int latency_frames) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: nemotron-600m (streaming, latency="
              << latency_frames << " frames)" << std::endl;
    NemotronTranscriber transcriber(weights_path, vocab_path,
                                    make_nemotron_600m_config(latency_frames));
    if (use_fp16)
        transcriber.to_half();
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
                          const std::string &audio_path, bool use_gpu,
                          bool use_fp16) {
    using namespace parakeet;
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "Loading model: sortformer" << std::endl;
    SortformerConfig cfg = make_sortformer_117m_config();
    Sortformer model(cfg);

    auto weights = axiom::io::safetensors::load(weights_path);
    model.load_state_dict(weights, "", false);

    if (use_fp16) {
        model.to(axiom::DType::Float16);
        std::cout << "Model cast to fp16" << std::endl;
    }

    if (use_gpu) {
        model.to(axiom::Device::GPU);
    }

    auto audio = read_audio(audio_path);
    AudioConfig audio_cfg;
    audio_cfg.n_mels = cfg.nest_encoder.mel_bins;
    audio_cfg.normalize = false; // sortformer uses normalize: NA
    auto features = preprocess_audio(audio.samples, audio_cfg);
    if (use_fp16)
        features = features.half();
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
                        bool use_ctc, bool use_gpu, bool use_fp16,
                        const std::string &vad_weights_path = {}) {
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

    if (use_fp16) {
        dt.to_half();
        std::cout << "Models cast to fp16" << std::endl;
    }

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

    if (!vad_weights_path.empty()) {
        dt.enable_vad(vad_weights_path);
        std::cout << "VAD enabled" << std::endl;
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

    // Print speaker-grouped output: merge consecutive runs of same speaker
    std::cout << "\n--- Diarized Transcription ---" << std::endl;
    for (size_t i = 0; i < result.words.size();) {
        int spk = result.words[i].speaker_id;
        float run_start = result.words[i].start;
        std::string run_text = result.words[i].word;
        size_t j = i + 1;
        while (j < result.words.size() && result.words[j].speaker_id == spk) {
            run_text += " " + result.words[j].word;
            ++j;
        }
        float run_end = result.words[j - 1].end;

        if (spk >= 0) {
            std::cout << "Speaker " << spk;
        } else {
            std::cout << "Unknown";
        }
        std::cout << " [" << std::fixed << std::setprecision(2) << run_start
                  << "s - " << run_end << "s]: " << run_text << std::endl;
        i = j;
    }

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
        std::string model_type = "tdt-ctc-110m";
        bool use_ctc = false;
        bool use_ctc_beam = false;
        bool use_tdt_beam = false;
        bool use_gpu = false;
        bool use_fp16 = false;
        bool show_timestamps = false;
        bool streaming = false;
        int latency_frames = 0;
        int beam_width = 8;
        std::string lm_path;
        float lm_weight = 0.5f;
        std::string vocab_path;
        std::string features_path;
        std::string sortformer_weights_path;
        std::vector<std::string> boost_phrases;
        float boost_score = 5.0f;
        std::string vad_weights_path;
        float vad_threshold = 0.5f;
        std::vector<std::string> audio_paths;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                model_type = argv[++i];
            } else if (arg == "--ctc") {
                use_ctc = true;
                use_ctc_beam = false;
            } else if (arg == "--tdt") {
                use_ctc = false;
                use_ctc_beam = false;
            } else if (arg == "--ctc-beam") {
                use_ctc_beam = true;
                use_ctc = false;
                use_tdt_beam = false;
            } else if (arg == "--tdt-beam") {
                use_tdt_beam = true;
                use_ctc = false;
                use_ctc_beam = false;
            } else if (arg == "--beam-width" && i + 1 < argc) {
                beam_width = std::stoi(argv[++i]);
            } else if (arg == "--lm" && i + 1 < argc) {
                lm_path = argv[++i];
            } else if (arg == "--lm-weight" && i + 1 < argc) {
                lm_weight = std::stof(argv[++i]);
            } else if (arg == "--gpu") {
                use_gpu = true;
            } else if (arg == "--fp16") {
                use_fp16 = true;
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
            } else if (arg == "--boost" && i + 1 < argc) {
                boost_phrases.push_back(argv[++i]);
            } else if (arg == "--boost-score" && i + 1 < argc) {
                boost_score = std::stof(argv[++i]);
            } else if (arg == "--vad" && i + 1 < argc) {
                vad_weights_path = argv[++i];
            } else if (arg == "--vad-threshold" && i + 1 < argc) {
                vad_threshold = std::stof(argv[++i]);
            } else if (arg.substr(0, 2) == "--") {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            } else {
                // Positional argument = audio file
                audio_paths.push_back(arg);
            }
        }

        if (audio_paths.empty()) {
            std::cerr << "Error: at least one audio file required" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        if (use_gpu && !axiom::system::is_metal_available()) {
            std::cerr << "Error: Metal GPU not available" << std::endl;
            return 1;
        }

        std::string audio_path =
            audio_paths[0]; // first audio for single-file modes

        if (model_type == "tdt-ctc-110m") {
            if (audio_paths.size() > 1) {
                return run_tdt_ctc_110m_batch(
                    weights_path, audio_paths, vocab_path, use_ctc, use_gpu,
                    use_fp16, show_timestamps, boost_phrases, boost_score,
                    vad_weights_path, vad_threshold);
            }
            return run_tdt_ctc_110m(
                weights_path, audio_path, vocab_path, features_path, use_ctc,
                use_gpu, use_fp16, show_timestamps, boost_phrases, boost_score,
                use_ctc_beam, beam_width, lm_path, lm_weight, use_tdt_beam,
                vad_weights_path, vad_threshold);
        } else if (model_type == "tdt-600m") {
            if (audio_paths.size() > 1) {
                return run_tdt_600m_batch(weights_path, audio_paths, vocab_path,
                                          use_gpu, use_fp16, show_timestamps,
                                          boost_phrases, boost_score,
                                          vad_weights_path, vad_threshold);
            }
            return run_tdt_600m(weights_path, audio_path, vocab_path, use_gpu,
                                use_fp16, show_timestamps, boost_phrases,
                                boost_score, use_tdt_beam, beam_width, lm_path,
                                lm_weight, vad_weights_path, vad_threshold);
        } else if (model_type == "rnnt-600m") {
            return run_rnnt_600m(weights_path, audio_path, vocab_path, use_gpu,
                                 use_fp16, show_timestamps);
        } else if (model_type == "eou-120m") {
            return run_eou_streaming(weights_path, audio_path, vocab_path,
                                     use_gpu, use_fp16, show_timestamps);
        } else if (model_type == "nemotron-600m") {
            return run_nemotron_streaming(weights_path, audio_path, vocab_path,
                                          use_gpu, use_fp16, show_timestamps,
                                          latency_frames);
        } else if (model_type == "sortformer") {
            return run_sortformer(weights_path, audio_path, use_gpu, use_fp16);
        } else if (model_type == "diarized") {
            return run_diarized(weights_path, audio_path, vocab_path,
                                sortformer_weights_path, use_ctc, use_gpu,
                                use_fp16, vad_weights_path);
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
