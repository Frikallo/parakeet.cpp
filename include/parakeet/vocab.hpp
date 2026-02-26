#pragma once

#include <string>
#include <vector>

namespace parakeet {

// SentencePiece BPE tokenizer for Parakeet models.
// Token ID 0 = blank, IDs 1..vocab_size = BPE pieces.
class Tokenizer {
  public:
    // Load SentencePiece .vocab file (piece\tscore per line).
    // Pieces are stored 1-indexed: file line 0 -> token ID 1, etc.
    void load(const std::string &vocab_path);

    // Decode token IDs to text.
    // Removes blank (0), maps IDs to pieces, joins, replaces U+2581 with space.
    std::string decode(const std::vector<int> &token_ids) const;

    bool loaded() const { return !pieces_.empty(); }
    size_t vocab_size() const { return pieces_.size() + 1; } // +1 for blank

  private:
    std::vector<std::string> pieces_; // index i = piece for token ID i+1
};

} // namespace parakeet
