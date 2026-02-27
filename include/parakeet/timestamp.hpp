#pragma once

#include <string>
#include <vector>

namespace parakeet {

// ─── Timestamp Types
// ──────────────────────────────────────────────────────────

struct TimestampedToken {
    int token_id;
    int start_frame; // encoder frame index
    int end_frame;   // encoder frame index (inclusive)
};

struct WordTimestamp {
    std::string word;
    float start; // seconds
    float end;   // seconds
};

// ─── Frame ↔ Time Conversion ─────────────────────────────────────────────────

// Encoder frames → seconds.
// Each encoder frame = subsampling_factor (8) * hop_length (160) / sample_rate
// (16000)
//                    = 8 * 160 / 16000 = 0.08s
constexpr float FRAME_DURATION_S = 0.08f;

inline float frame_to_seconds(int frame) {
    return static_cast<float>(frame) * FRAME_DURATION_S;
}

// ─── Word-Level Grouping
// ──────────────────────────────────────────────────────

enum class TimestampMode {
    Words,     // Group by SentencePiece ▁ boundary marker
    Sentences, // Group by sentence-ending punctuation (.?!)
};

// Group timestamped tokens into word-level timestamps.
// Uses the tokenizer's piece table to detect word boundaries (▁ prefix).
std::vector<WordTimestamp>
group_timestamps(const std::vector<TimestampedToken> &tokens,
                 const std::vector<std::string> &pieces,
                 TimestampMode mode = TimestampMode::Words);

} // namespace parakeet
