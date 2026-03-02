#include "parakeet/decode/arpa_lm.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace parakeet::decode {

int ArpaLM::get_or_create_word_id(const std::string &word) {
    auto it = word_to_id_.find(word);
    if (it != word_to_id_.end())
        return it->second;
    int id = static_cast<int>(word_to_id_.size());
    word_to_id_[word] = id;
    return id;
}

int ArpaLM::get_word_id(const std::string &word) const {
    auto it = word_to_id_.find(word);
    return it != word_to_id_.end() ? it->second : -1;
}

int ArpaLM::find_child(int node_id, int word_id) const {
    const auto &children = nodes_[node_id].children;
    auto it = children.find(word_id);
    return it != children.end() ? it->second : -1;
}

float ArpaLM::score_with_backoff(int node_id, int word_id) const {
    // Try direct lookup
    int child = find_child(node_id, word_id);
    if (child >= 0 && nodes_[child].has_entry) {
        return nodes_[child].log_prob;
    }

    // Backoff: use the backoff weight at the current node
    // and try from unigram (root's children)
    if (node_id != 0) {
        // Back off to root and look up the unigram
        float bo = nodes_[node_id].backoff;
        int unigram_child = find_child(0, word_id);
        if (unigram_child >= 0 && nodes_[unigram_child].has_entry) {
            return bo + nodes_[unigram_child].log_prob;
        }
    }

    // Unknown word — return a large negative value
    return -99.0f;
}

void ArpaLM::load(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ARPA file: " + path);
    }

    nodes_.clear();
    word_to_id_.clear();
    num_ngrams_ = 0;
    order_ = 0;

    // Create root node
    nodes_.push_back(TrieNode{});

    std::string line;
    int current_order = 0;
    bool in_data = false;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        // Trim trailing whitespace/CR
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                 line.back() == '\t'))
            line.pop_back();

        if (line == "\\data\\") {
            in_data = true;
            continue;
        }

        if (line == "\\end\\") {
            break;
        }

        if (in_data && line.substr(0, 5) == "ngram") {
            // Parse "ngram N=COUNT"
            auto eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                int n = std::stoi(line.substr(6, eq_pos - 6));
                if (n > order_)
                    order_ = n;
            }
            continue;
        }

        // Check for "\N-grams:" section header
        if (line.size() > 2 && line[0] == '\\' &&
            line.find("-grams:") != std::string::npos) {
            current_order = std::stoi(line.substr(1, line.find('-') - 1));
            in_data = false;
            continue;
        }

        if (current_order == 0)
            continue;

        // Parse n-gram entry: log_prob word1 [word2 ...] [backoff]
        std::istringstream iss(line);
        float log_prob;
        if (!(iss >> log_prob))
            continue;

        std::vector<std::string> words;
        std::string token;
        while (iss >> token) {
            words.push_back(token);
        }

        if (words.empty())
            continue;

        // Last element might be a backoff weight (for non-highest-order
        // n-grams)
        float backoff = 0.0f;
        bool has_backoff = false;
        if (current_order<order_ &&static_cast<int>(words.size())>
                current_order) {
            // The extra token is the backoff weight
            try {
                backoff = std::stof(words.back());
                has_backoff = true;
                words.pop_back();
            } catch (...) {
                // Not a number, keep it as a word
            }
        }

        if (static_cast<int>(words.size()) != current_order)
            continue;

        // Insert into trie
        int node_id = 0;
        for (int i = 0; i < current_order; ++i) {
            int wid = get_or_create_word_id(words[i]);
            int child = find_child(node_id, wid);
            if (child < 0) {
                child = static_cast<int>(nodes_.size());
                nodes_.push_back(TrieNode{});
                nodes_[node_id].children[wid] = child;
            }
            node_id = child;
        }

        nodes_[node_id].log_prob = log_prob;
        nodes_[node_id].has_entry = true;
        if (has_backoff) {
            nodes_[node_id].backoff = backoff;
        }
        ++num_ngrams_;
    }
}

float ArpaLM::score(State &state, const std::string &word) const {
    int wid = get_word_id(word);
    if (wid < 0) {
        // Unknown word — stay in current state, return OOV penalty
        return -99.0f;
    }

    float lp = score_with_backoff(state.node_id, wid);

    // Advance state: try to find the longest matching context
    // Try current_context + word first
    int child = find_child(state.node_id, wid);
    if (child >= 0) {
        state.node_id = child;
    } else {
        // Fall back to unigram context
        int unigram = find_child(0, wid);
        state.node_id = unigram >= 0 ? unigram : 0;
    }

    return lp;
}

} // namespace parakeet::decode
