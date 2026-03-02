#include "parakeet/decode/beam_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace parakeet::decode {

// ─── Internal Types ─────────────────────────────────────────────────────────

namespace {

// Log-space addition: log(exp(a) + exp(b))
inline float log_add(float a, float b) {
    if (a == -std::numeric_limits<float>::infinity())
        return b;
    if (b == -std::numeric_limits<float>::infinity())
        return a;
    float max_val = std::max(a, b);
    return max_val + std::log1pf(std::expf(std::min(a, b) - max_val));
}

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// SentencePiece word boundary marker: U+2581 (▁)
const std::string WORD_BOUNDARY = "\xe2\x96\x81";

// Check if a token starts a new word (has ▁ prefix).
bool is_word_start(int token_id, const std::vector<std::string> *pieces) {
    if (!pieces || token_id < 0 || token_id >= static_cast<int>(pieces->size()))
        return false;
    const auto &piece = (*pieces)[token_id];
    return piece.size() >= 3 && piece.compare(0, 3, WORD_BOUNDARY) == 0;
}

// Extract the word text from a piece (strip ▁ prefix if present).
std::string piece_to_word(int token_id,
                          const std::vector<std::string> *pieces) {
    if (!pieces || token_id < 0 || token_id >= static_cast<int>(pieces->size()))
        return "";
    const auto &piece = (*pieces)[token_id];
    if (piece.size() >= 3 && piece.compare(0, 3, WORD_BOUNDARY) == 0) {
        return piece.substr(3);
    }
    return piece;
}

// Collect the current word being built from the last ▁-starting token.
// Returns the decoded word for LM scoring.
std::string get_current_word(const std::vector<int> &prefix,
                             const std::vector<std::string> *pieces) {
    if (!pieces || prefix.empty())
        return "";

    // Find last word-start token
    std::string word;
    for (int i = static_cast<int>(prefix.size()) - 1; i >= 0; --i) {
        if (is_word_start(prefix[i], pieces)) {
            // Build word from this token to end
            word.clear();
            for (int j = i; j < static_cast<int>(prefix.size()); ++j) {
                word += piece_to_word(prefix[j], pieces);
            }
            return word;
        }
    }
    // No word boundary found — whole prefix is one word
    for (int id : prefix) {
        word += piece_to_word(id, pieces);
    }
    return word;
}

struct Hypothesis {
    std::vector<int> prefix;
    float p_blank;     // log prob of prefix ending in blank
    float p_non_blank; // log prob of prefix ending in non-blank
    float lm_score;    // accumulated LM score (log10, weighted)
    ArpaLM::State lm_state;

    // For timestamps
    std::vector<int> emission_frames; // frame at which each token was emitted

    float total_score() const { return log_add(p_blank, p_non_blank); }

    // Combined score with LM
    float combined_score() const { return total_score() + lm_score; }
};

// Hash for prefix vectors — used as map key
struct PrefixHash {
    size_t operator()(const std::vector<int> &v) const {
        size_t seed = v.size();
        for (auto &i : v) {
            seed ^=
                static_cast<size_t>(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Run beam search on a single sequence.
// log_probs_data: pointer to (T, V) contiguous float data
// T: sequence length, V: vocab size
Hypothesis beam_search_single(const float *log_probs_data, int T, int V,
                              const BeamSearchOptions &opts) {
    int beam_width = opts.beam_width;
    int blank_id = opts.blank_id;
    const ArpaLM *lm = opts.lm;
    float lm_weight = opts.lm_weight;
    const auto *pieces = opts.pieces;
    bool use_lm = lm && lm->loaded() && pieces;

    // Initialize beam with empty prefix
    std::vector<Hypothesis> beam(1);
    beam[0].p_blank = 0.0f; // log(1)
    beam[0].p_non_blank = NEG_INF;
    beam[0].lm_score = 0.0f;
    if (use_lm)
        beam[0].lm_state = lm->initial_state();

    for (int t = 0; t < T; ++t) {
        const float *frame = log_probs_data + t * V;

        // Map from prefix -> merged hypothesis
        std::unordered_map<std::vector<int>, Hypothesis, PrefixHash> next_beam;

        for (const auto &hyp : beam) {
            float p_total = hyp.total_score();
            int last_token = hyp.prefix.empty() ? -1 : hyp.prefix.back();

            // 1. Extend with blank
            {
                float p = p_total + frame[blank_id];
                auto it = next_beam.find(hyp.prefix);
                if (it != next_beam.end()) {
                    it->second.p_blank = log_add(it->second.p_blank, p);
                } else {
                    Hypothesis h;
                    h.prefix = hyp.prefix;
                    h.p_blank = p;
                    h.p_non_blank = NEG_INF;
                    h.lm_score = hyp.lm_score;
                    h.lm_state = hyp.lm_state;
                    h.emission_frames = hyp.emission_frames;
                    next_beam[hyp.prefix] = std::move(h);
                }
            }

            // 2. Extend with each non-blank token
            // Only consider top-k tokens per frame to limit computation
            // Find top tokens by scanning the frame
            struct TokenScore {
                int id;
                float score;
            };

            // Collect top beam_width tokens (excluding blank)
            std::vector<TokenScore> top_tokens;
            top_tokens.reserve(V);
            for (int v = 0; v < V; ++v) {
                if (v == blank_id)
                    continue;
                // Pre-filter: skip tokens with very low probability
                if (frame[v] < -30.0f)
                    continue;
                top_tokens.push_back({v, frame[v]});
            }
            // Partial sort to get top beam_width * 2 candidates
            int k =
                std::min(static_cast<int>(top_tokens.size()), beam_width * 2);
            if (k < static_cast<int>(top_tokens.size())) {
                std::partial_sort(top_tokens.begin(), top_tokens.begin() + k,
                                  top_tokens.end(),
                                  [](const TokenScore &a, const TokenScore &b) {
                                      return a.score > b.score;
                                  });
                top_tokens.resize(k);
            }

            for (const auto &[token_id, log_p] : top_tokens) {
                if (token_id == last_token) {
                    // Same as last token in prefix — two sub-cases:

                    // Case A: from non-blank ending → continued emission
                    // of the same token, keeps prefix y unchanged.
                    // p_nb(y) += p_nb(y) * P(c|t)
                    {
                        float p = hyp.p_non_blank + log_p;
                        auto it = next_beam.find(hyp.prefix);
                        if (it != next_beam.end()) {
                            it->second.p_non_blank =
                                log_add(it->second.p_non_blank, p);
                        } else {
                            Hypothesis h;
                            h.prefix = hyp.prefix;
                            h.p_blank = NEG_INF;
                            h.p_non_blank = p;
                            h.lm_score = hyp.lm_score;
                            h.lm_state = hyp.lm_state;
                            h.emission_frames = hyp.emission_frames;
                            next_beam[hyp.prefix] = std::move(h);
                        }
                    }

                    // Case B: from blank ending → blank-separated repeat,
                    // creates new prefix y+c.
                    // p_nb(y+c) += p_b(y) * P(c|t)
                    {
                        std::vector<int> new_prefix = hyp.prefix;
                        new_prefix.push_back(token_id);
                        float p = hyp.p_blank + log_p;

                        float lm_bonus = 0.0f;
                        ArpaLM::State new_lm_state = hyp.lm_state;
                        if (use_lm && is_word_start(token_id, pieces)) {
                            std::string word =
                                get_current_word(hyp.prefix, pieces);
                            if (!word.empty()) {
                                new_lm_state = hyp.lm_state;
                                float lm_lp = lm->score(new_lm_state, word);
                                lm_bonus = lm_weight * lm_lp * std::log(10.0f);
                            }
                        }

                        auto it = next_beam.find(new_prefix);
                        if (it != next_beam.end()) {
                            it->second.p_non_blank =
                                log_add(it->second.p_non_blank, p);
                            if (hyp.lm_score + lm_bonus > it->second.lm_score) {
                                it->second.lm_score = hyp.lm_score + lm_bonus;
                                it->second.lm_state = new_lm_state;
                            }
                        } else {
                            Hypothesis h;
                            h.prefix = new_prefix;
                            h.p_blank = NEG_INF;
                            h.p_non_blank = p;
                            h.lm_score = hyp.lm_score + lm_bonus;
                            h.lm_state = new_lm_state;
                            h.emission_frames = hyp.emission_frames;
                            h.emission_frames.push_back(t);
                            next_beam[new_prefix] = std::move(h);
                        }
                    }
                } else {
                    // Different token → always extends prefix
                    std::vector<int> new_prefix = hyp.prefix;
                    new_prefix.push_back(token_id);
                    float p = p_total + log_p;

                    float lm_bonus = 0.0f;
                    ArpaLM::State new_lm_state = hyp.lm_state;
                    if (use_lm && is_word_start(token_id, pieces)) {
                        // Score the completed word before this new word start
                        std::string word = get_current_word(hyp.prefix, pieces);
                        if (!word.empty()) {
                            new_lm_state = hyp.lm_state;
                            float lm_lp = lm->score(new_lm_state, word);
                            lm_bonus = lm_weight * lm_lp * std::log(10.0f);
                        }
                    }

                    auto it = next_beam.find(new_prefix);
                    if (it != next_beam.end()) {
                        it->second.p_non_blank =
                            log_add(it->second.p_non_blank, p);
                        if (hyp.lm_score + lm_bonus > it->second.lm_score) {
                            it->second.lm_score = hyp.lm_score + lm_bonus;
                            it->second.lm_state = new_lm_state;
                        }
                    } else {
                        Hypothesis h;
                        h.prefix = new_prefix;
                        h.p_blank = NEG_INF;
                        h.p_non_blank = p;
                        h.lm_score = hyp.lm_score + lm_bonus;
                        h.lm_state = new_lm_state;
                        h.emission_frames = hyp.emission_frames;
                        h.emission_frames.push_back(t);
                        next_beam[new_prefix] = std::move(h);
                    }
                }
            }
        }

        // Prune to beam_width
        beam.clear();
        beam.reserve(next_beam.size());
        for (auto &[prefix, hyp] : next_beam) {
            beam.push_back(std::move(hyp));
        }
        std::partial_sort(
            beam.begin(),
            beam.begin() + std::min(beam_width, static_cast<int>(beam.size())),
            beam.end(), [](const Hypothesis &a, const Hypothesis &b) {
                return a.combined_score() > b.combined_score();
            });
        if (static_cast<int>(beam.size()) > beam_width) {
            beam.resize(beam_width);
        }
    }

    // Return best hypothesis
    if (beam.empty()) {
        return Hypothesis{};
    }

    auto best = std::max_element(
        beam.begin(), beam.end(), [](const Hypothesis &a, const Hypothesis &b) {
            return a.combined_score() < b.combined_score();
        });

    return *best;
}

} // anonymous namespace

// ─── Public API ─────────────────────────────────────────────────────────────

std::vector<std::vector<int>> ctc_beam_decode(const axiom::Tensor &log_probs,
                                              const BeamSearchOptions &opts,
                                              const std::vector<int> &lengths) {
    auto lp = log_probs.to_float().ascontiguousarray();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;
        const float *batch_data = data + b * seq_len * vocab_size;

        auto best = beam_search_single(batch_data, T, vocab_size, opts);
        results[b] = std::move(best.prefix);
    }

    return results;
}

std::vector<std::vector<TimestampedToken>>
ctc_beam_decode_with_timestamps(const axiom::Tensor &log_probs,
                                const BeamSearchOptions &opts,
                                const std::vector<int> &lengths) {
    auto lp = log_probs.to_float().ascontiguousarray();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;
        const float *batch_data = data + b * seq_len * vocab_size;

        auto best = beam_search_single(batch_data, T, vocab_size, opts);

        // Convert emission frames to TimestampedToken format
        auto &tokens = results[b];
        tokens.reserve(best.prefix.size());

        for (size_t i = 0; i < best.prefix.size(); ++i) {
            int frame =
                (i < best.emission_frames.size()) ? best.emission_frames[i] : 0;
            // Look up the log-prob at the emission frame for confidence
            float log_p = -1.0f;
            if (frame < T) {
                log_p = batch_data[frame * vocab_size + best.prefix[i]];
            }
            float confidence = std::exp(log_p);

            // Estimate end_frame: next token's start - 1, or T-1 for last
            int end_frame = (i + 1 < best.emission_frames.size())
                                ? best.emission_frames[i + 1] - 1
                                : T - 1;
            if (end_frame < frame)
                end_frame = frame;

            tokens.push_back({best.prefix[i], frame, end_frame, confidence});
        }
    }

    return results;
}

} // namespace parakeet::decode
