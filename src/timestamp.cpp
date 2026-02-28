#include "parakeet/timestamp.hpp"

#include <algorithm>

namespace parakeet {

// ─── Word-Level Grouping
// ──────────────────────────────────────────────────────

// SentencePiece word boundary marker: U+2581 (▁) encoded as 3 bytes
static const std::string SP_MARKER = "\xe2\x96\x81";

static bool starts_with_marker(const std::string &piece) {
    return piece.size() >= 3 && piece.compare(0, 3, SP_MARKER) == 0;
}

static bool is_sentence_end(const std::string &word) {
    if (word.empty())
        return false;
    char last = word.back();
    return last == '.' || last == '?' || last == '!';
}

std::vector<WordTimestamp>
group_timestamps(const std::vector<TimestampedToken> &tokens,
                 const std::vector<std::string> &pieces, TimestampMode mode) {
    if (tokens.empty())
        return {};

    std::vector<WordTimestamp> words;
    std::string current_word;
    int word_start_frame = tokens[0].start_frame;
    int word_end_frame = tokens[0].end_frame;
    float word_min_confidence = 1.0f;

    for (const auto &tok : tokens) {
        if (tok.token_id < 0 ||
            tok.token_id >= static_cast<int>(pieces.size())) {
            continue;
        }

        const std::string &piece = pieces[tok.token_id];

        // Check if this token starts a new word (has ▁ prefix)
        bool new_word = starts_with_marker(piece);

        if (new_word && !current_word.empty()) {
            // Flush the current word
            words.push_back({current_word, frame_to_seconds(word_start_frame),
                             frame_to_seconds(word_end_frame),
                             word_min_confidence});
            current_word.clear();
            word_start_frame = tok.start_frame;
            word_min_confidence = 1.0f;
        }

        // Append piece to current word (strip ▁ if present)
        if (starts_with_marker(piece)) {
            current_word += piece.substr(3);
        } else {
            current_word += piece;
        }
        word_end_frame = tok.end_frame;
        word_min_confidence = std::min(word_min_confidence, tok.confidence);
    }

    // Flush last word
    if (!current_word.empty()) {
        words.push_back({current_word, frame_to_seconds(word_start_frame),
                         frame_to_seconds(word_end_frame),
                         word_min_confidence});
    }

    // If sentence mode, merge words into sentences
    if (mode == TimestampMode::Sentences) {
        std::vector<WordTimestamp> sentences;
        std::string current_sentence;
        float sent_start = 0.0f;
        float sent_end = 0.0f;

        float sent_min_confidence = 1.0f;

        for (const auto &w : words) {
            if (current_sentence.empty()) {
                sent_start = w.start;
            } else {
                current_sentence += ' ';
            }
            current_sentence += w.word;
            sent_end = w.end;
            sent_min_confidence = std::min(sent_min_confidence, w.confidence);

            if (is_sentence_end(w.word)) {
                sentences.push_back({current_sentence, sent_start, sent_end,
                                     sent_min_confidence});
                current_sentence.clear();
                sent_min_confidence = 1.0f;
            }
        }

        // Flush remaining
        if (!current_sentence.empty()) {
            sentences.push_back(
                {current_sentence, sent_start, sent_end, sent_min_confidence});
        }

        return sentences;
    }

    return words;
}

} // namespace parakeet
