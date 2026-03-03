#include "parakeet/audio/vad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace parakeet::audio {

// ─── SileroVAD ───────────────────────────────────────────────────────────────

SileroVAD::SileroVAD(const std::string &weights_path) {
    auto weights = axiom::io::safetensors::load(weights_path);
    model_.load_state_dict(weights, "", false);
}

void SileroVAD::to_gpu() {
    model_.to(axiom::Device::GPU);
    use_gpu_ = true;
    initialized_ = false; // re-init states on next use
}

void SileroVAD::to_half() {
    model_.to(axiom::DType::Float16);
    use_fp16_ = true;
    initialized_ = false;
}

void SileroVAD::ensure_states() {
    if (!initialized_) {
        auto [h, c] = model_.get_initial_states(1);
        h_ = h;
        c_ = c;
        context_ = axiom::Tensor::zeros(
            {static_cast<size_t>(model_.config().context_size)});
        initialized_ = true;
    }
}

void SileroVAD::reset() { initialized_ = false; }

float SileroVAD::process_chunk(const axiom::Tensor &chunk, int sample_rate) {
    if (sample_rate != model_.config().sample_rate) {
        throw std::runtime_error("SileroVAD: sample rate mismatch (expected " +
                                 std::to_string(model_.config().sample_rate) +
                                 ", got " + std::to_string(sample_rate) + ")");
    }

    ensure_states();

    int window = model_.config().num_samples;
    int context = model_.config().context_size;

    // Pad if needed
    axiom::Tensor padded = chunk;
    int chunk_len = static_cast<int>(chunk.shape()[0]);
    if (chunk_len < window) {
        auto pad =
            axiom::Tensor::zeros({static_cast<size_t>(window - chunk_len)});
        padded = axiom::Tensor::cat({chunk, pad}, 0);
    } else if (chunk_len > window) {
        padded = chunk.slice({axiom::Slice(0, window)});
    }

    // Prepend context → (576,)
    auto input = axiom::Tensor::cat({context_, padded}, 0);
    context_ = padded.slice({axiom::Slice(window - context)}); // update context

    // Add batch dim
    input = input.unsqueeze(0);

    auto [prob, h_new, c_new] = model_.forward(input, h_, c_);
    h_ = h_new;
    c_ = c_new;

    // Extract scalar probability
    auto cpu_prob = prob.to_contiguous_cpu();
    return cpu_prob.typed_data<float>()[0];
}

// ─── get_speech_timestamps algorithm ─────────────────────────────────────────
//
// Ported from the Python Silero VAD utils:
// 1. Run VAD on all windows → per-window probabilities
// 2. Hysteresis thresholding (threshold for onset, neg_threshold for offset)
// 3. Collect raw speech segments
// 4. Merge short silence gaps, enforce min/max duration
// 5. Apply speech padding

std::vector<SpeechSegment> SileroVAD::detect(const axiom::Tensor &audio,
                                             int sample_rate,
                                             const VADConfig &config) {
    if (sample_rate != model_.config().sample_rate) {
        throw std::runtime_error("SileroVAD: sample rate mismatch (expected " +
                                 std::to_string(model_.config().sample_rate) +
                                 ", got " + std::to_string(sample_rate) + ")");
    }

    int total_samples = static_cast<int>(audio.shape()[0]);
    int window = model_.config().num_samples; // 512

    // Reset state for fresh detection
    reset();
    ensure_states();

    // Compute per-window speech probabilities
    auto probs_tensor = model_.predict(audio);
    auto cpu_probs = probs_tensor.to_contiguous_cpu();
    int num_windows = static_cast<int>(cpu_probs.shape()[0]);
    const float *probs = cpu_probs.typed_data<float>();

    // Convert config durations to samples
    int min_speech_samples = config.min_speech_duration_ms * sample_rate / 1000;
    int min_silence_samples =
        config.min_silence_duration_ms * sample_rate / 1000;
    int speech_pad_samples = config.speech_pad_ms * sample_rate / 1000;

    // Hysteresis-based speech detection
    bool triggered = false;
    int speech_start = 0;
    int temp_end = 0;

    struct RawSegment {
        int start;
        int end;
    };
    std::vector<RawSegment> speeches;

    for (int i = 0; i < num_windows; ++i) {
        float p = probs[i];
        int current_sample = i * window;

        if (p >= config.threshold && !triggered) {
            // Speech onset
            triggered = true;
            speech_start = current_sample;
            temp_end = 0;
        }

        if (p < config.neg_threshold && triggered) {
            // Potential speech offset
            if (temp_end == 0) {
                temp_end = current_sample;
            }

            // Check if silence is long enough
            if (current_sample - temp_end >= min_silence_samples) {
                // Confirm end of speech
                int speech_end = temp_end;

                // Check min speech duration
                if (speech_end - speech_start >= min_speech_samples) {
                    speeches.push_back({speech_start, speech_end});
                }

                triggered = false;
                temp_end = 0;
            }
        } else if (p >= config.neg_threshold && triggered) {
            // Reset temp_end if probability goes back up
            temp_end = 0;
        }
    }

    // Handle speech that extends to end of audio
    if (triggered) {
        int speech_end = total_samples;
        if (speech_end - speech_start >= min_speech_samples) {
            speeches.push_back({speech_start, speech_end});
        }
    }

    // Enforce max speech duration by splitting long segments
    if (std::isfinite(config.max_speech_duration_s)) {
        int max_samples =
            static_cast<int>(config.max_speech_duration_s * sample_rate);
        std::vector<RawSegment> split_speeches;
        for (auto &seg : speeches) {
            if (seg.end - seg.start > max_samples) {
                // Split into chunks
                for (int s = seg.start; s < seg.end; s += max_samples) {
                    split_speeches.push_back(
                        {s, std::min(s + max_samples, seg.end)});
                }
            } else {
                split_speeches.push_back(seg);
            }
        }
        speeches = std::move(split_speeches);
    }

    // Apply speech padding and clamp to audio bounds
    std::vector<SpeechSegment> result;
    result.reserve(speeches.size());
    for (auto &seg : speeches) {
        int64_t start = std::max(0, seg.start - speech_pad_samples);
        int64_t end = std::min(total_samples, seg.end + speech_pad_samples);
        result.push_back({start, end});
    }

    // Merge overlapping segments (can happen after padding)
    if (result.size() > 1) {
        std::vector<SpeechSegment> merged;
        merged.push_back(result[0]);
        for (size_t i = 1; i < result.size(); ++i) {
            if (result[i].start_sample <= merged.back().end_sample) {
                merged.back().end_sample =
                    std::max(merged.back().end_sample, result[i].end_sample);
            } else {
                merged.push_back(result[i]);
            }
        }
        result = std::move(merged);
    }

    return result;
}

// ─── collect_speech ──────────────────────────────────────────────────────────

axiom::Tensor collect_speech(const axiom::Tensor &audio,
                             const std::vector<SpeechSegment> &segments) {
    if (segments.empty()) {
        return audio; // no segments = return full audio
    }

    std::vector<axiom::Tensor> parts;
    parts.reserve(segments.size());

    for (const auto &seg : segments) {
        parts.push_back(
            audio.slice({axiom::Slice(seg.start_sample, seg.end_sample)}));
    }

    return axiom::Tensor::cat(parts, 0);
}

// ─── TimestampRemapper ───────────────────────────────────────────────────────

TimestampRemapper::TimestampRemapper(const std::vector<SpeechSegment> &segments,
                                     int sample_rate)
    : segments_(segments), sample_rate_(sample_rate), total_compressed_(0.0f) {
    compressed_starts_.reserve(segments.size());

    float cumulative = 0.0f;
    for (const auto &seg : segments) {
        compressed_starts_.push_back(cumulative);
        float duration = static_cast<float>(seg.end_sample - seg.start_sample) /
                         static_cast<float>(sample_rate);
        cumulative += duration;
    }
    total_compressed_ = cumulative;
}

float TimestampRemapper::remap(float compressed_seconds) const {
    if (segments_.empty()) {
        return compressed_seconds;
    }

    // Clamp to valid range
    compressed_seconds = std::max(0.0f, compressed_seconds);
    if (compressed_seconds >= total_compressed_) {
        // Past the end → return end of last segment
        return static_cast<float>(segments_.back().end_sample) /
               static_cast<float>(sample_rate_);
    }

    // Binary search: find which segment this compressed time falls in
    // compressed_starts_[i] <= compressed_seconds < compressed_starts_[i+1]
    auto it = std::upper_bound(compressed_starts_.begin(),
                               compressed_starts_.end(), compressed_seconds);
    int idx = static_cast<int>(it - compressed_starts_.begin()) - 1;
    idx = std::max(0, idx);

    // Offset within this segment
    float offset_in_segment = compressed_seconds - compressed_starts_[idx];

    // Map to original timeline
    float original_start = static_cast<float>(segments_[idx].start_sample) /
                           static_cast<float>(sample_rate_);

    return original_start + offset_in_segment;
}

WordTimestamp TimestampRemapper::remap(const WordTimestamp &w) const {
    WordTimestamp out = w;
    out.start = remap(w.start);
    out.end = remap(w.end);
    return out;
}

std::vector<WordTimestamp>
TimestampRemapper::remap(const std::vector<WordTimestamp> &words) const {
    std::vector<WordTimestamp> out;
    out.reserve(words.size());
    for (const auto &w : words) {
        out.push_back(remap(w));
    }
    return out;
}

TimestampedToken
TimestampRemapper::remap_token(const TimestampedToken &t) const {
    // Convert frame→seconds, remap, convert back to frame
    float start_s = decode::frame_to_seconds(t.start_frame);
    float end_s = decode::frame_to_seconds(t.end_frame);

    float remapped_start = remap(start_s);
    float remapped_end = remap(end_s);

    TimestampedToken out = t;
    out.start_frame =
        static_cast<int>(remapped_start / decode::FRAME_DURATION_S + 0.5f);
    out.end_frame =
        static_cast<int>(remapped_end / decode::FRAME_DURATION_S + 0.5f);
    return out;
}

std::vector<TimestampedToken> TimestampRemapper::remap_tokens(
    const std::vector<TimestampedToken> &tokens) const {
    std::vector<TimestampedToken> out;
    out.reserve(tokens.size());
    for (const auto &t : tokens) {
        out.push_back(remap_token(t));
    }
    return out;
}

} // namespace parakeet::audio
