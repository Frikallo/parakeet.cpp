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
        // Token IDs are 0-indexed: ID N -> pieces_[N]
        if (id < 0 || id >= static_cast<int>(pieces_.size())) {
            result += "[" + std::to_string(id) + "]";
            continue;
        }

        const std::string &piece = pieces_[id];
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

void Tokenizer::build_encode_table_() const {
    if (!piece_to_id_.empty())
        return;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        piece_to_id_[pieces_[i]] = static_cast<int>(i);
        if (pieces_[i].size() > max_piece_len_)
            max_piece_len_ = pieces_[i].size();
    }
}

std::vector<int> Tokenizer::encode(const std::string &text) const {
    if (pieces_.empty() || text.empty())
        return {};

    build_encode_table_();

    // Prepare input: prepend ▁, replace spaces with ▁
    const std::string marker = "\xe2\x96\x81"; // U+2581
    std::string input = marker;
    for (char c : text) {
        if (c == ' ') {
            input += marker;
        } else {
            input += c;
        }
    }

    std::vector<int> result;
    size_t pos = 0;
    while (pos < input.size()) {
        // Greedy longest match
        size_t best_len = 0;
        int best_id = -1;
        size_t max_len = std::min(max_piece_len_, input.size() - pos);
        for (size_t len = max_len; len >= 1; --len) {
            auto it = piece_to_id_.find(input.substr(pos, len));
            if (it != piece_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            result.push_back(best_id);
            pos += best_len;
        } else {
            // Skip unknown byte
            ++pos;
        }
    }
    return result;
}

} // namespace parakeet
