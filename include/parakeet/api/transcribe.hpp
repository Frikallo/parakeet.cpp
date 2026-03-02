#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>
#include <axiom/io/safetensors.hpp>

#include "parakeet/audio/audio.hpp"
#include "parakeet/audio/audio_io.hpp"
#include "parakeet/audio/vad.hpp"
#include "parakeet/decode/arpa_lm.hpp"
#include "parakeet/decode/beam_search.hpp"
#include "parakeet/decode/phrase_boost.hpp"
#include "parakeet/decode/tdt_beam_search.hpp"
#include "parakeet/decode/timestamp.hpp"
#include "parakeet/decode/vocab.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/ctc.hpp"
#include "parakeet/models/eou.hpp"
#include "parakeet/models/nemotron.hpp"
#include "parakeet/models/tdt.hpp"
#include "parakeet/models/tdt_ctc.hpp"

namespace parakeet::api {

using namespace models;
using namespace decode;
using namespace audio;

// ─── Transcription Result ───────────────────────────────────────────────────

struct TranscribeResult {
    std::string text;           // Decoded text
    std::vector<int> token_ids; // Raw token IDs before detokenization

    // Optional timestamps (populated when timestamps=true)
    std::vector<TimestampedToken> timestamped_tokens;
    std::vector<WordTimestamp> word_timestamps;
};

// ─── High-Level Transcription API ───────────────────────────────────────────

enum class Decoder { CTC, TDT, CTC_BEAM, TDT_BEAM };

// ─── Transcription Options ──────────────────────────────────────────────────

struct TranscribeOptions {
    Decoder decoder = Decoder::TDT;
    bool timestamps = false;
    std::vector<std::string> boost_phrases;
    float boost_score = 5.0f;

    // Beam search options (used when decoder == CTC_BEAM or TDT_BEAM)
    int beam_width = 8;
    std::string lm_path;    // Path to ARPA language model (optional)
    float lm_weight = 0.5f; // LM interpolation weight
    bool use_vad = false;   // Enable Silero VAD preprocessing
};

/// Transcribe an audio file to text using a ParakeetTDTCTC model.
/// Supports WAV, FLAC, MP3, OGG. Automatically resamples to 16kHz.
///
/// This is the simplest entry point — loads audio, preprocesses, encodes,
/// decodes, and detokenizes in one call.
///
///   parakeet::Transcriber t("model.safetensors", "vocab.txt");
///   auto result = t.transcribe("audio.wav");
///   std::cout << result.text << std::endl;
///
class Transcriber {
  public:
    /// Construct from a safetensors weights file and SentencePiece vocab file.
    /// Uses the 110M TDT-CTC config by default.
    Transcriber(const std::string &weights_path, const std::string &vocab_path,
                const TDTCTCConfig &config = make_110m_config())
        : config_(config), model_(config) {
        auto weights = axiom::io::safetensors::load(weights_path);
        model_.load_state_dict(weights, "", false);
        tokenizer_.load(vocab_path);
    }

    /// Move model to GPU (Metal). Call once after construction.
    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    /// Cast model to fp16. Call before to_gpu() for efficient transfer.
    void to_half() {
        model_.to(axiom::DType::Float16);
        use_fp16_ = true;
    }

    /// Transcribe an audio file (WAV, FLAC, MP3, OGG).
    TranscribeResult transcribe(const std::string &audio_path,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        auto audio_data = read_audio(audio_path);
        return transcribe(audio_data.samples, decoder, timestamps);
    }

    /// Transcribe from raw float32 samples (16kHz mono).
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                Decoder decoder = Decoder::TDT,
                                bool timestamps = false) {
        TranscribeOptions opts;
        opts.decoder = decoder;
        opts.timestamps = timestamps;
        return transcribe(samples, opts);
    }

    /// Transcribe an audio file with options (boost phrases, etc.).
    TranscribeResult transcribe(const std::string &audio_path,
                                const TranscribeOptions &opts) {
        auto audio_data = read_audio(audio_path);
        return transcribe(audio_data.samples, opts);
    }

    /// Transcribe from raw float32 samples with options.
    TranscribeResult transcribe(const axiom::Tensor &samples,
                                const TranscribeOptions &opts) {
        // VAD preprocessing: strip silence, build remapper
        axiom::Tensor audio_for_asr = samples;
        std::unique_ptr<audio::TimestampRemapper> remapper;

        if (opts.use_vad && vad_) {
            auto segments = vad_->detect(samples);
            if (!segments.empty()) {
                audio_for_asr = audio::collect_speech(samples, segments);
                if (opts.timestamps) {
                    remapper =
                        std::make_unique<audio::TimestampRemapper>(segments);
                }
            }
        }

        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto features = preprocess_audio(audio_for_asr, audio_cfg);
        if (use_fp16_)
            features = features.half();
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        // Build trie if boost phrases provided
        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        // Load LM if beam search with LM requested
        ArpaLM lm;
        if ((opts.decoder == Decoder::CTC_BEAM ||
             opts.decoder == Decoder::TDT_BEAM) &&
            !opts.lm_path.empty()) {
            lm.load(opts.lm_path);
        }

        TranscribeResult result;

        if (opts.timestamps) {
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = tdt_beam_decode_with_timestamps(
                    model_, encoder_out, config_.durations, bs_opts);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else if (opts.decoder == Decoder::CTC_BEAM) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                BeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = ctc_beam_decode_with_timestamps(cpu_lp, bs_opts);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                auto all_ts = use_boost
                                  ? ctc_greedy_decode_with_timestamps_boosted(
                                        cpu_lp, trie, opts.boost_score)
                                  : ctc_greedy_decode_with_timestamps(cpu_lp);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else {
                auto all_ts = use_boost
                                  ? tdt_greedy_decode_with_timestamps_boosted(
                                        model_, encoder_out, config_.durations,
                                        trie, opts.boost_score)
                                  : tdt_greedy_decode_with_timestamps(
                                        model_, encoder_out, config_.durations);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            }

            // Remap timestamps if VAD was used
            if (remapper) {
                result.timestamped_tokens =
                    remapper->remap_tokens(result.timestamped_tokens);
            }

            if (tokenizer_.loaded()) {
                result.text = tokenizer_.decode(result.token_ids);
                result.word_timestamps = group_timestamps(
                    result.timestamped_tokens, tokenizer_.pieces());
            }
        } else {
            std::vector<std::vector<int>> all_tokens;
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                all_tokens = tdt_beam_decode(model_, encoder_out,
                                             config_.durations, bs_opts);
            } else if (opts.decoder == Decoder::CTC_BEAM) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                BeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                all_tokens = ctc_beam_decode(cpu_lp, bs_opts);
            } else if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                all_tokens = use_boost ? ctc_greedy_decode_boosted(
                                             cpu_lp, trie, opts.boost_score)
                                       : ctc_greedy_decode(cpu_lp);
            } else {
                all_tokens = use_boost
                                 ? tdt_greedy_decode_boosted(
                                       model_, encoder_out, config_.durations,
                                       trie, opts.boost_score)
                                 : tdt_greedy_decode(model_, encoder_out,
                                                     config_.durations);
            }

            if (!all_tokens.empty()) {
                result.token_ids = all_tokens[0];
                if (tokenizer_.loaded()) {
                    result.text = tokenizer_.decode(result.token_ids);
                }
            }
        }

        return result;
    }

    /// Batch transcribe multiple audio files.
    std::vector<TranscribeResult>
    transcribe_batch(const std::vector<std::string> &audio_paths,
                     const TranscribeOptions &opts = {}) {
        std::vector<axiom::Tensor> waveforms;
        waveforms.reserve(audio_paths.size());
        for (const auto &path : audio_paths) {
            auto audio_data = read_audio(path);
            waveforms.push_back(audio_data.samples);
        }
        return transcribe_batch(waveforms, opts);
    }

    /// Batch transcribe from multiple raw sample tensors.
    std::vector<TranscribeResult>
    transcribe_batch(const std::vector<axiom::Tensor> &samples,
                     const TranscribeOptions &opts = {}) {
        if (samples.empty())
            return {};
        if (samples.size() == 1) {
            return {transcribe(samples[0], opts)};
        }

        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto batch = preprocess_audio_batch(samples, audio_cfg);

        // Compute subsampled lengths
        std::vector<int> sub_lengths;
        int max_sub_len = 0;
        sub_lengths.reserve(batch.feature_lengths.size());
        for (int fl : batch.feature_lengths) {
            int sl = compute_subsampled_length(fl);
            sub_lengths.push_back(sl);
            if (sl > max_sub_len)
                max_sub_len = sl;
        }

        auto mask = create_padding_mask(sub_lengths, max_sub_len);
        auto features = batch.features;

        if (use_fp16_) {
            features = features.half();
            mask = mask.half();
        }
        if (use_gpu_) {
            features = features.gpu();
            mask = mask.gpu();
        }

        auto encoder_out = model_.encoder()(features, mask);

        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        // Load LM if beam search with LM requested
        ArpaLM batch_lm;
        if ((opts.decoder == Decoder::CTC_BEAM ||
             opts.decoder == Decoder::TDT_BEAM) &&
            !opts.lm_path.empty()) {
            batch_lm.load(opts.lm_path);
        }

        std::vector<TranscribeResult> results(samples.size());

        if (opts.timestamps) {
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = tdt_beam_decode_with_timestamps(
                    model_, encoder_out, config_.durations, bs_opts,
                    sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            } else if (opts.decoder == Decoder::CTC_BEAM) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                BeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = ctc_beam_decode_with_timestamps(cpu_lp, bs_opts,
                                                              sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            } else if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                auto all_ts =
                    use_boost
                        ? ctc_greedy_decode_with_timestamps_boosted(
                              cpu_lp, trie, opts.boost_score, 1024, sub_lengths)
                        : ctc_greedy_decode_with_timestamps(cpu_lp, 1024,
                                                            sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            } else {
                auto all_ts =
                    use_boost
                        ? tdt_greedy_decode_with_timestamps_boosted(
                              model_, encoder_out, config_.durations, trie,
                              opts.boost_score, 1024, 10, sub_lengths)
                        : tdt_greedy_decode_with_timestamps(
                              model_, encoder_out, config_.durations, 1024, 10,
                              sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            }
            for (size_t b = 0; b < results.size(); ++b) {
                if (tokenizer_.loaded()) {
                    results[b].text = tokenizer_.decode(results[b].token_ids);
                    results[b].word_timestamps = group_timestamps(
                        results[b].timestamped_tokens, tokenizer_.pieces());
                }
            }
        } else {
            std::vector<std::vector<int>> all_tokens;
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                all_tokens =
                    tdt_beam_decode(model_, encoder_out, config_.durations,
                                    bs_opts, sub_lengths);
            } else if (opts.decoder == Decoder::CTC_BEAM) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                BeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                all_tokens = ctc_beam_decode(cpu_lp, bs_opts, sub_lengths);
            } else if (opts.decoder == Decoder::CTC) {
                auto log_probs = model_.ctc_decoder()(encoder_out);
                auto cpu_lp = log_probs.cpu();
                all_tokens = use_boost
                                 ? ctc_greedy_decode_boosted(cpu_lp, trie,
                                                             opts.boost_score,
                                                             1024, sub_lengths)
                                 : ctc_greedy_decode(cpu_lp, 1024, sub_lengths);
            } else {
                all_tokens =
                    use_boost
                        ? tdt_greedy_decode_boosted(
                              model_, encoder_out, config_.durations, trie,
                              opts.boost_score, 1024, 10, sub_lengths)
                        : tdt_greedy_decode(model_, encoder_out,
                                            config_.durations, 1024, 10,
                                            sub_lengths);
            }
            for (size_t b = 0; b < all_tokens.size(); ++b) {
                results[b].token_ids = all_tokens[b];
                if (tokenizer_.loaded()) {
                    results[b].text = tokenizer_.decode(results[b].token_ids);
                }
            }
        }

        return results;
    }

    /// Enable VAD preprocessing. Call after to_half()/to_gpu().
    void enable_vad(const std::string &vad_weights_path) {
        vad_ = std::make_unique<audio::SileroVAD>(vad_weights_path);
        if (use_fp16_)
            vad_->to_half();
        if (use_gpu_)
            vad_->to_gpu();
    }

    /// Access the underlying model for advanced use.
    ParakeetTDTCTC &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    TDTCTCConfig config_;
    ParakeetTDTCTC model_;
    Tokenizer tokenizer_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    std::unique_ptr<audio::SileroVAD> vad_;
};

// ─── TDT-Only Transcriber (for 600M multilingual etc.) ─────────────────────

/// Transcribe using a TDT-only model (no CTC head).
///
///   parakeet::TDTTranscriber t("model.safetensors", "vocab.txt",
///                               parakeet::make_tdt_600m_config());
///   auto result = t.transcribe("audio.wav");
///
class TDTTranscriber {
  public:
    TDTTranscriber(const std::string &weights_path,
                   const std::string &vocab_path,
                   const TDTConfig &config = make_tdt_600m_config())
        : config_(config), model_(config) {
        auto weights = axiom::io::safetensors::load(weights_path);
        model_.load_state_dict(weights, "", false);
        tokenizer_.load(vocab_path);
    }

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    void to_half() {
        model_.to(axiom::DType::Float16);
        use_fp16_ = true;
    }

    TranscribeResult transcribe(const std::string &audio_path,
                                bool timestamps = false) {
        auto audio_data = read_audio(audio_path);
        return transcribe(audio_data.samples, timestamps);
    }

    TranscribeResult transcribe(const axiom::Tensor &samples,
                                bool timestamps = false) {
        TranscribeOptions opts;
        opts.decoder = Decoder::TDT;
        opts.timestamps = timestamps;
        return transcribe(samples, opts);
    }

    TranscribeResult transcribe(const std::string &audio_path,
                                const TranscribeOptions &opts) {
        auto audio_data = read_audio(audio_path);
        return transcribe(audio_data.samples, opts);
    }

    TranscribeResult transcribe(const axiom::Tensor &samples,
                                const TranscribeOptions &opts) {
        // VAD preprocessing
        axiom::Tensor audio_for_asr = samples;
        std::unique_ptr<audio::TimestampRemapper> remapper;

        if (opts.use_vad && vad_) {
            auto segments = vad_->detect(samples);
            if (!segments.empty()) {
                audio_for_asr = audio::collect_speech(samples, segments);
                if (opts.timestamps) {
                    remapper =
                        std::make_unique<audio::TimestampRemapper>(segments);
                }
            }
        }

        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto features = preprocess_audio(audio_for_asr, audio_cfg);
        if (use_fp16_)
            features = features.half();
        if (use_gpu_) {
            features = features.gpu();
        }

        auto encoder_out = model_.encoder()(features);

        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        // Load LM if beam search with LM requested
        ArpaLM lm;
        if (opts.decoder == Decoder::TDT_BEAM && !opts.lm_path.empty()) {
            lm.load(opts.lm_path);
        }

        TranscribeResult result;

        if (opts.timestamps) {
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = tdt_beam_decode_with_timestamps(
                    model_, encoder_out, config_.durations, bs_opts);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            } else {
                auto all_ts = use_boost
                                  ? tdt_greedy_decode_with_timestamps_boosted(
                                        model_, encoder_out, config_.durations,
                                        trie, opts.boost_score)
                                  : tdt_greedy_decode_with_timestamps(
                                        model_, encoder_out, config_.durations);
                if (!all_ts.empty()) {
                    result.timestamped_tokens = all_ts[0];
                    for (const auto &t : result.timestamped_tokens) {
                        result.token_ids.push_back(t.token_id);
                    }
                }
            }

            if (remapper) {
                result.timestamped_tokens =
                    remapper->remap_tokens(result.timestamped_tokens);
            }

            if (tokenizer_.loaded()) {
                result.text = tokenizer_.decode(result.token_ids);
                result.word_timestamps = group_timestamps(
                    result.timestamped_tokens, tokenizer_.pieces());
            }
        } else {
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.lm = lm.loaded() ? &lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_tokens = tdt_beam_decode(model_, encoder_out,
                                                  config_.durations, bs_opts);
                if (!all_tokens.empty()) {
                    result.token_ids = all_tokens[0];
                    if (tokenizer_.loaded()) {
                        result.text = tokenizer_.decode(result.token_ids);
                    }
                }
            } else {
                auto all_tokens =
                    use_boost
                        ? tdt_greedy_decode_boosted(model_, encoder_out,
                                                    config_.durations, trie,
                                                    opts.boost_score)
                        : tdt_greedy_decode(model_, encoder_out,
                                            config_.durations);
                if (!all_tokens.empty()) {
                    result.token_ids = all_tokens[0];
                    if (tokenizer_.loaded()) {
                        result.text = tokenizer_.decode(result.token_ids);
                    }
                }
            }
        }

        return result;
    }

    /// Batch transcribe multiple audio files.
    std::vector<TranscribeResult>
    transcribe_batch(const std::vector<std::string> &audio_paths,
                     const TranscribeOptions &opts = {}) {
        std::vector<axiom::Tensor> waveforms;
        waveforms.reserve(audio_paths.size());
        for (const auto &path : audio_paths) {
            auto audio_data = read_audio(path);
            waveforms.push_back(audio_data.samples);
        }
        return transcribe_batch(waveforms, opts);
    }

    /// Batch transcribe from multiple raw sample tensors.
    std::vector<TranscribeResult>
    transcribe_batch(const std::vector<axiom::Tensor> &samples,
                     const TranscribeOptions &opts = {}) {
        if (samples.empty())
            return {};
        if (samples.size() == 1) {
            return {transcribe(samples[0], opts)};
        }

        AudioConfig audio_cfg;
        audio_cfg.n_mels = config_.encoder.mel_bins;
        auto batch = preprocess_audio_batch(samples, audio_cfg);

        std::vector<int> sub_lengths;
        int max_sub_len = 0;
        sub_lengths.reserve(batch.feature_lengths.size());
        for (int fl : batch.feature_lengths) {
            int sl = compute_subsampled_length(fl);
            sub_lengths.push_back(sl);
            if (sl > max_sub_len)
                max_sub_len = sl;
        }

        auto mask = create_padding_mask(sub_lengths, max_sub_len);
        auto features = batch.features;

        if (use_fp16_) {
            features = features.half();
            mask = mask.half();
        }
        if (use_gpu_) {
            features = features.gpu();
            mask = mask.gpu();
        }

        auto encoder_out = model_.encoder()(features, mask);

        ContextTrie trie;
        bool use_boost = !opts.boost_phrases.empty();
        if (use_boost) {
            trie.build(opts.boost_phrases, tokenizer_);
        }

        // Load LM if beam search with LM requested
        ArpaLM batch_lm;
        if (opts.decoder == Decoder::TDT_BEAM && !opts.lm_path.empty()) {
            batch_lm.load(opts.lm_path);
        }

        int blank_id = config_.prediction.vocab_size - 1;
        std::vector<TranscribeResult> results(samples.size());

        if (opts.timestamps) {
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.blank_id = blank_id;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                auto all_ts = tdt_beam_decode_with_timestamps(
                    model_, encoder_out, config_.durations, bs_opts,
                    sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            } else {
                auto all_ts =
                    use_boost
                        ? tdt_greedy_decode_with_timestamps_boosted(
                              model_, encoder_out, config_.durations, trie,
                              opts.boost_score, blank_id, 10, sub_lengths)
                        : tdt_greedy_decode_with_timestamps(
                              model_, encoder_out, config_.durations, blank_id,
                              10, sub_lengths);
                for (size_t b = 0; b < all_ts.size(); ++b) {
                    results[b].timestamped_tokens = all_ts[b];
                    for (const auto &t : all_ts[b])
                        results[b].token_ids.push_back(t.token_id);
                }
            }
            for (size_t b = 0; b < results.size(); ++b) {
                if (tokenizer_.loaded()) {
                    results[b].text = tokenizer_.decode(results[b].token_ids);
                    results[b].word_timestamps = group_timestamps(
                        results[b].timestamped_tokens, tokenizer_.pieces());
                }
            }
        } else {
            std::vector<std::vector<int>> all_tokens;
            if (opts.decoder == Decoder::TDT_BEAM) {
                TDTBeamSearchOptions bs_opts;
                bs_opts.beam_width = opts.beam_width;
                bs_opts.blank_id = blank_id;
                bs_opts.lm = batch_lm.loaded() ? &batch_lm : nullptr;
                bs_opts.lm_weight = opts.lm_weight;
                bs_opts.pieces =
                    tokenizer_.loaded() ? &tokenizer_.pieces() : nullptr;
                all_tokens =
                    tdt_beam_decode(model_, encoder_out, config_.durations,
                                    bs_opts, sub_lengths);
            } else {
                all_tokens =
                    use_boost
                        ? tdt_greedy_decode_boosted(
                              model_, encoder_out, config_.durations, trie,
                              opts.boost_score, blank_id, 10, sub_lengths)
                        : tdt_greedy_decode(model_, encoder_out,
                                            config_.durations, blank_id, 10,
                                            sub_lengths);
            }
            for (size_t b = 0; b < all_tokens.size(); ++b) {
                results[b].token_ids = all_tokens[b];
                if (tokenizer_.loaded()) {
                    results[b].text = tokenizer_.decode(results[b].token_ids);
                }
            }
        }

        return results;
    }

    /// Enable VAD preprocessing. Call after to_half()/to_gpu().
    void enable_vad(const std::string &vad_weights_path) {
        vad_ = std::make_unique<audio::SileroVAD>(vad_weights_path);
        if (use_fp16_)
            vad_->to_half();
        if (use_gpu_)
            vad_->to_gpu();
    }

    ParakeetTDT &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    TDTConfig config_;
    ParakeetTDT model_;
    Tokenizer tokenizer_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    std::unique_ptr<audio::SileroVAD> vad_;
};

// ─── Streaming Transcriber ───────────────────────────────────────────────────

// Callback for partial transcription results.
using PartialResultCallback = std::function<void(const std::string &partial)>;

class StreamingTranscriber {
  public:
    StreamingTranscriber(const std::string &weights_path,
                         const std::string &vocab_path,
                         const EOUConfig &config = make_eou_120m_config());

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    void to_half() {
        model_.to(axiom::DType::Float16);
        use_fp16_ = true;
    }

    // Process a chunk of raw audio samples.
    // Returns any new text produced by this chunk.
    std::string transcribe_chunk(const axiom::Tensor &samples);

    // Convenience: process raw float32 PCM buffer.
    std::string transcribe_chunk(const float *data, size_t num_samples) {
        auto t =
            axiom::Tensor::from_data(data, axiom::Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Convenience: process raw int16 PCM buffer (converted to float32).
    std::string transcribe_chunk(const int16_t *data, size_t num_samples) {
        std::vector<float> f(num_samples);
        for (size_t i = 0; i < num_samples; ++i)
            f[i] = static_cast<float>(data[i]) / 32768.0f;
        auto t =
            axiom::Tensor::from_data(f.data(), axiom::Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Reset state for a new utterance.
    void reset();

    // Set callback for partial results (called each time new tokens are
    // emitted).
    void set_partial_callback(PartialResultCallback cb) {
        partial_callback_ = std::move(cb);
    }

    // Get full transcription so far.
    std::string get_text() const;

    // Get accumulated timestamped tokens across all chunks.
    const std::vector<TimestampedToken> &get_timestamped_tokens() const {
        return decode_state_.timestamped_tokens;
    }

    ParakeetEOU &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    EOUConfig config_;
    ParakeetEOU model_;
    Tokenizer tokenizer_;
    StreamingAudioPreprocessor preprocessor_;
    EncoderCache encoder_cache_;
    StreamingDecodeState decode_state_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    PartialResultCallback partial_callback_;
};

// ─── Nemotron Streaming Transcriber ──────────────────────────────────────────

class NemotronTranscriber {
  public:
    NemotronTranscriber(
        const std::string &weights_path, const std::string &vocab_path,
        const NemotronConfig &config = make_nemotron_600m_config());

    void to_gpu() {
        model_.to(axiom::Device::GPU);
        use_gpu_ = true;
    }

    void to_half() {
        model_.to(axiom::DType::Float16);
        use_fp16_ = true;
    }

    // Process a chunk of raw audio → returns new text from this chunk
    std::string transcribe_chunk(const axiom::Tensor &samples);

    // Convenience: process raw float32 PCM buffer.
    std::string transcribe_chunk(const float *data, size_t num_samples) {
        auto t =
            axiom::Tensor::from_data(data, axiom::Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Convenience: process raw int16 PCM buffer (converted to float32).
    std::string transcribe_chunk(const int16_t *data, size_t num_samples) {
        std::vector<float> f(num_samples);
        for (size_t i = 0; i < num_samples; ++i)
            f[i] = static_cast<float>(data[i]) / 32768.0f;
        auto t =
            axiom::Tensor::from_data(f.data(), axiom::Shape{num_samples}, true);
        return transcribe_chunk(t);
    }

    // Reset for a new utterance
    void reset();

    // Get full transcription so far
    std::string get_text() const;

    void set_partial_callback(PartialResultCallback cb) {
        partial_callback_ = std::move(cb);
    }

    const std::vector<TimestampedToken> &get_timestamped_tokens() const {
        return decode_state_.timestamped_tokens;
    }

    ParakeetNemotron &model() { return model_; }
    const Tokenizer &tokenizer() const { return tokenizer_; }

  private:
    NemotronConfig config_;
    ParakeetNemotron model_;
    Tokenizer tokenizer_;
    StreamingAudioPreprocessor preprocessor_;
    EncoderCache encoder_cache_;
    StreamingDecodeState decode_state_;
    bool use_gpu_ = false;
    bool use_fp16_ = false;
    PartialResultCallback partial_callback_;
};

} // namespace parakeet::api
