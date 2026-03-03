#include "parakeet/api/diarize.hpp"

#include <algorithm>
#include <memory>
#include <unordered_map>

namespace parakeet::api {

// ─── Alignment: assign speakers to words by max temporal overlap ────────────

std::vector<DiarizedWord>
diarize_transcription(const std::vector<WordTimestamp> &words,
                      const std::vector<DiarizationSegment> &segments) {
    std::vector<DiarizedWord> result;
    result.reserve(words.size());

    for (const auto &w : words) {
        DiarizedWord dw;
        dw.word = w.word;
        dw.start = w.start;
        dw.end = w.end;
        dw.confidence = w.confidence;
        dw.speaker_id = -1;

        // Accumulate overlap per speaker
        std::unordered_map<int, float> overlap_by_speaker;

        for (const auto &seg : segments) {
            float overlap =
                std::min(w.end, seg.end) - std::max(w.start, seg.start);
            if (overlap > 0.0f) {
                overlap_by_speaker[seg.speaker_id] += overlap;
            }
        }

        // Pick speaker with maximum total overlap
        float best_overlap = 0.0f;
        for (const auto &[spk, ovl] : overlap_by_speaker) {
            if (ovl > best_overlap) {
                best_overlap = ovl;
                dw.speaker_id = spk;
            }
        }

        result.push_back(std::move(dw));
    }

    return result;
}

// ─── DiarizedTranscriber ────────────────────────────────────────────────────

DiarizedTranscriber::DiarizedTranscriber(const std::string &asr_weights,
                                         const std::string &sortformer_weights,
                                         const std::string &vocab_path,
                                         const TDTCTCConfig &config,
                                         const SortformerConfig &sf_config)
    : transcriber_(asr_weights, vocab_path, config), sortformer_(sf_config),
      sf_config_(sf_config) {
    auto weights = axiom::io::safetensors::load(sortformer_weights);
    sortformer_.load_state_dict(weights, "", false);
}

void DiarizedTranscriber::to_gpu() {
    transcriber_.to_gpu();
    sortformer_.to(axiom::Device::GPU);
    use_gpu_ = true;
}

void DiarizedTranscriber::to_half() {
    transcriber_.to_half();
    api::module_to_dtype(sortformer_, axiom::DType::Float16);
    use_fp16_ = true;
}

void DiarizedTranscriber::enable_vad(const std::string &vad_weights_path) {
    vad_ = std::make_unique<audio::SileroVAD>(vad_weights_path);
    if (use_fp16_)
        vad_->to_half();
    if (use_gpu_)
        vad_->to_gpu();
}

DiarizedResult DiarizedTranscriber::transcribe(const std::string &audio_path,
                                               Decoder decoder) {
    auto audio_data = read_audio(audio_path);
    return transcribe(audio_data.samples, decoder);
}

DiarizedResult DiarizedTranscriber::transcribe(const axiom::Tensor &samples,
                                               Decoder decoder) {
    // VAD preprocessing: filter audio for both ASR and Sortformer
    axiom::Tensor audio_for_processing = samples;
    std::unique_ptr<audio::TimestampRemapper> remapper;

    if (vad_) {
        auto vad_segments = vad_->detect(samples);
        if (!vad_segments.empty()) {
            audio_for_processing = audio::collect_speech(samples, vad_segments);
            remapper = std::make_unique<audio::TimestampRemapper>(vad_segments);
        }
    }

    // 1. Run ASR with timestamps (VAD already handled at transcriber level
    //    if enabled, but we pass the filtered audio directly here)
    TranscribeOptions opts;
    opts.decoder = decoder;
    opts.timestamps = true;
    opts.use_vad = false; // We handle VAD at this level
    auto asr_result = transcriber_.transcribe(audio_for_processing, opts);

    // 2. Run Sortformer diarization (128 mel, no normalization)
    AudioConfig sf_audio_cfg;
    sf_audio_cfg.n_mels = sf_config_.nest_encoder.mel_bins;
    sf_audio_cfg.normalize = false;
    auto sf_features = preprocess_audio(audio_for_processing, sf_audio_cfg);
    if (use_fp16_)
        sf_features = sf_features.half();
    if (use_gpu_) {
        sf_features = sf_features.gpu();
    }
    auto segments = sortformer_.diarize(sf_features);

    // 3. Remap timestamps if VAD was used
    if (remapper) {
        asr_result.timestamped_tokens =
            remapper->remap_tokens(asr_result.timestamped_tokens);
        asr_result.word_timestamps =
            remapper->remap(asr_result.word_timestamps);
        // Remap diarization segments too
        for (auto &seg : segments) {
            seg.start = remapper->remap(seg.start);
            seg.end = remapper->remap(seg.end);
        }
    }

    // 4. Fuse: assign speakers to words
    auto diarized_words =
        diarize_transcription(asr_result.word_timestamps, segments);

    // 5. Build result
    DiarizedResult result;
    result.text = asr_result.text;
    result.words = std::move(diarized_words);
    result.segments = std::move(segments);
    result.word_timestamps = std::move(asr_result.word_timestamps);
    return result;
}

} // namespace parakeet::api
