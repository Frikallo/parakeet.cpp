#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet::decode {

// ─── ARPA N-gram Language Model ─────────────────────────────────────────────
//
// Loads a standard .arpa language model file and provides n-gram log-prob
// queries. Uses trie-based storage for efficient prefix lookup.
//
// ARPA format:
//   \data\
//   ngram 1=...
//   ngram 2=...
//   \1-grams:
//   log_prob  word  [backoff_weight]
//   \2-grams:
//   log_prob  word1 word2  [backoff_weight]
//   ...
//   \end\

class ArpaLM {
  public:
    // LM state for tracking context during beam search.
    // Stores the trie node index for the current n-gram context.
    struct State {
        int node_id = 0; // root = 0
    };

    ArpaLM() = default;

    // Load an ARPA-format language model file.
    void load(const std::string &path);

    bool loaded() const { return !nodes_.empty(); }

    // Query the log10 probability of a word given the current state.
    // Returns the LM score and advances the state.
    float score(State &state, const std::string &word) const;

    // Get the initial (empty-context) state.
    State initial_state() const { return State{0}; }

    // Number of n-gram entries loaded.
    size_t size() const { return num_ngrams_; }

    // Maximum n-gram order.
    int order() const { return order_; }

  private:
    struct TrieNode {
        std::unordered_map<int, int> children; // word_id -> child node index
        float log_prob = -99.0f;               // log10 probability at this node
        float backoff = 0.0f;                  // backoff weight (log10)
        bool has_entry = false; // whether this node has an explicit n-gram
    };

    std::vector<TrieNode> nodes_;
    std::unordered_map<std::string, int> word_to_id_;
    int order_ = 0;
    size_t num_ngrams_ = 0;

    int get_or_create_word_id(const std::string &word);
    int get_word_id(const std::string &word) const;

    // Find child node, returns -1 if not found.
    int find_child(int node_id, int word_id) const;

    // Backoff: try shorter context until we find a match.
    float score_with_backoff(int node_id, int word_id) const;
};

} // namespace parakeet::decode
