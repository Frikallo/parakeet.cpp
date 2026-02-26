#include "parakeet/sortformer.hpp"
#include "parakeet/timestamp.hpp"

#include <algorithm>
#include <cmath>

namespace parakeet {

// ─── AOSCCache ───────────────────────────────────────────────────────────────

AOSCCache::AOSCCache(int max_speakers) : max_speakers_(max_speakers) {
    speaker_active_.resize(max_speakers, false);
}

void AOSCCache::update(const Tensor &probs) {
    // probs: (T, max_speakers) — sigmoid probabilities
    auto p = probs.cpu().ascontiguousarray();
    int T = static_cast<int>(p.shape()[0]);
    int S = static_cast<int>(p.shape()[1]);
    const float *data = p.typed_data<float>();

    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < S && s < max_speakers_; ++s) {
            float prob = data[t * S + s];
            if (prob > 0.5f && !speaker_active_[s]) {
                speaker_active_[s] = true;
                arrival_order_.push_back(s);
            }
        }
    }
}

std::vector<int> AOSCCache::speaker_order() const { return arrival_order_; }

void AOSCCache::reset() {
    std::fill(speaker_active_.begin(), speaker_active_.end(), false);
    arrival_order_.clear();
}

// ─── Sortformer ──────────────────────────────────────────────────────────────

Sortformer::Sortformer(const SortformerConfig &config)
    : config_(config), nest_encoder_(config.nest_encoder), projection_(true),
      transformer_(config.transformer), output_proj_(true) {
    AX_REGISTER_MODULES(nest_encoder_, projection_, transformer_, output_proj_);
}

Tensor Sortformer::forward(const Tensor &features) const {
    // 1. NEST encoder: (batch, mel_len, mel_bins) → (batch, T, encoder_hidden)
    auto enc_out = nest_encoder_(features);

    // 2. Project to transformer dim
    auto proj = projection_(enc_out); // (batch, T, transformer_hidden)

    // 3. Transformer encoder
    auto trans_out = transformer_(proj);

    // 4. Output projection → (batch, T, max_speakers)
    auto logits = output_proj_(trans_out);

    // 5. Sigmoid activation (multi-label: each speaker independently)
    return ops::sigmoid(logits);
}

std::vector<DiarizationSegment>
Sortformer::probs_to_segments(const Tensor &probs) const {
    // probs: (T, max_speakers) on CPU
    auto p = probs.cpu().ascontiguousarray();
    int T = static_cast<int>(p.shape()[0]);
    int S = static_cast<int>(p.shape()[1]);
    const float *data = p.typed_data<float>();

    std::vector<DiarizationSegment> segments;
    float threshold = config_.activity_threshold;

    // Per speaker: find contiguous active regions
    for (int s = 0; s < S; ++s) {
        bool in_segment = false;
        int seg_start = 0;

        for (int t = 0; t < T; ++t) {
            float prob = data[t * S + s];
            bool active = prob > threshold;

            if (active && !in_segment) {
                seg_start = t;
                in_segment = true;
            } else if (!active && in_segment) {
                segments.push_back(
                    {s, frame_to_seconds(seg_start), frame_to_seconds(t - 1)});
                in_segment = false;
            }
        }

        if (in_segment) {
            segments.push_back(
                {s, frame_to_seconds(seg_start), frame_to_seconds(T - 1)});
        }
    }

    // Sort by start time
    std::sort(segments.begin(), segments.end(),
              [](const DiarizationSegment &a, const DiarizationSegment &b) {
                  return a.start < b.start;
              });

    return segments;
}

std::vector<DiarizationSegment>
Sortformer::diarize(const Tensor &features) const {
    auto probs = forward(features); // (1, T, max_speakers)

    // Remove batch dim: (T, max_speakers)
    auto p = probs.squeeze(0);
    return probs_to_segments(p);
}

std::vector<DiarizationSegment>
Sortformer::diarize_chunk(const Tensor &features, EncoderCache &enc_cache,
                          AOSCCache &aosc_cache) const {
    // 1. Encode chunk
    auto enc_out = nest_encoder_.forward_chunk(features, enc_cache);
    if (!enc_out.storage()) {
        return {};
    }

    // 2. Project + transform
    auto proj = projection_(enc_out);
    auto trans_out = transformer_(proj);
    auto logits = output_proj_(trans_out);
    auto probs = ops::sigmoid(logits);

    // 3. Update AOSC
    auto p = probs.squeeze(0);
    aosc_cache.update(p);

    // 4. Convert to segments
    return probs_to_segments(p);
}

} // namespace parakeet
