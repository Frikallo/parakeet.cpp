#include "parakeet/vocab.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace parakeet {

void Tokenizer::load(const std::string &vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_path);
    }

    pieces_.clear();
    std::string line;
    while (std::getline(file, line)) {
        // SentencePiece .vocab format: piece<tab>score
        auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            pieces_.push_back(line.substr(0, tab_pos));
        } else if (!line.empty()) {
            pieces_.push_back(line);
        }
    }
}

std::string Tokenizer::decode(const std::vector<int> &token_ids) const {
    std::string result;

    for (int id : token_ids) {
        // Skip blank token (ID 0)
        if (id == 0)
            continue;

        // Token IDs are 1-indexed: ID N -> pieces_[N-1]
        int piece_idx = id - 1;
        if (piece_idx < 0 || piece_idx >= static_cast<int>(pieces_.size())) {
            result += "[" + std::to_string(id) + "]";
            continue;
        }

        const std::string &piece = pieces_[piece_idx];
        result += piece;
    }

    // Replace SentencePiece word boundary marker (U+2581 = \xe2\x96\x81) with
    // space
    const std::string marker = "\xe2\x96\x81";
    std::string output;
    size_t pos = 0;
    while (pos < result.size()) {
        if (pos + 3 <= result.size() && result.compare(pos, 3, marker) == 0) {
            output += ' ';
            pos += 3;
        } else {
            output += result[pos];
            ++pos;
        }
    }

    // Strip leading space
    if (!output.empty() && output[0] == ' ') {
        output.erase(0, 1);
    }

    return output;
}

} // namespace parakeet
