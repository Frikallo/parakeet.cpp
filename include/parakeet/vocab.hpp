#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet {

// SentencePiece BPE tokenizer for Parakeet models.
// Token IDs 0..1023 = BPE pieces, token ID 1024 = blank.
class Tokenizer {
  public:
    // Load SentencePiece .vocab file (piece\tscore per line).
    // Pieces are stored 0-indexed: file line 0 -> token ID 0, etc.
    void load(const std::string &vocab_path);

    // Decode token IDs to text.
    // Maps IDs to pieces, joins, replaces U+2581 with space.
    std::string decode(const std::vector<int> &token_ids) const;

    // Encode text to token IDs using greedy longest-match.
    // Prepends ▁ to the input and replaces spaces with ▁.
    std::vector<int> encode(const std::string &text) const;

    bool loaded() const { return !pieces_.empty(); }
    size_t vocab_size() const { return pieces_.size() + 1; } // +1 for blank
    const std::vector<std::string> &pieces() const { return pieces_; }

  private:
    std::vector<std::string> pieces_; // index i = piece for token ID i

    // Lazy-built lookup table for encode()
    mutable std::unordered_map<std::string, int> piece_to_id_;
    mutable size_t max_piece_len_ = 0;
    void build_encode_table_() const;
};

} // namespace parakeet
