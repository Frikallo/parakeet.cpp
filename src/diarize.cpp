#include "parakeet/diarize.hpp"

#include <algorithm>
#include <unordered_map>

namespace parakeet {

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

DiarizedResult DiarizedTranscriber::transcribe(const std::string &audio_path,
                                               Decoder decoder) {
    auto audio = read_audio(audio_path);
    return transcribe(audio.samples, decoder);
}

DiarizedResult DiarizedTranscriber::transcribe(const axiom::Tensor &samples,
                                               Decoder decoder) {
    // 1. Run ASR with timestamps
    auto asr_result = transcriber_.transcribe(samples, decoder,
                                              /*timestamps=*/true);

    // 2. Run Sortformer diarization (128 mel, no normalization)
    AudioConfig sf_audio_cfg;
    sf_audio_cfg.n_mels = sf_config_.nest_encoder.mel_bins;
    sf_audio_cfg.normalize = false;
    auto sf_features = preprocess_audio(samples, sf_audio_cfg);
    if (use_gpu_) {
        sf_features = sf_features.gpu();
    }
    auto segments = sortformer_.diarize(sf_features);

    // 3. Fuse: assign speakers to words
    auto diarized_words =
        diarize_transcription(asr_result.word_timestamps, segments);

    // 4. Build result
    DiarizedResult result;
    result.text = asr_result.text;
    result.words = std::move(diarized_words);
    result.segments = std::move(segments);
    result.word_timestamps = std::move(asr_result.word_timestamps);
    return result;
}

} // namespace parakeet
