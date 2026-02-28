#include "parakeet/phrase_boost.hpp"

#include <cmath>

namespace parakeet {

// ─── ContextTrie ────────────────────────────────────────────────────────────

ContextTrie::ContextTrie() { nodes_.push_back(TrieNode{}); } // root node

void ContextTrie::insert(const std::vector<int> &token_ids) {
    if (token_ids.empty())
        return;
    int node = 0;
    for (int tid : token_ids) {
        auto it = nodes_[node].children.find(tid);
        if (it == nodes_[node].children.end()) {
            int next = static_cast<int>(nodes_.size());
            nodes_[node].children[tid] = next;
            nodes_.push_back(TrieNode{});
            node = next;
        } else {
            node = it->second;
        }
    }
    nodes_[node].is_end = true;
}

void ContextTrie::build(const std::vector<std::string> &phrases,
                        const Tokenizer &tokenizer) {
    for (const auto &phrase : phrases) {
        auto ids = tokenizer.encode(phrase);
        if (!ids.empty()) {
            insert(ids);
        }
    }
}

std::unordered_set<int> ContextTrie::get_boosted_tokens(
    const std::unordered_set<int> &active_states) const {
    std::unordered_set<int> boosted;
    for (int state : active_states) {
        if (state < 0 || state >= static_cast<int>(nodes_.size()))
            continue;
        for (const auto &[tid, _] : nodes_[state].children) {
            boosted.insert(tid);
        }
    }
    return boosted;
}

std::unordered_set<int>
ContextTrie::advance(const std::unordered_set<int> &active_states,
                     int token_id) const {
    std::unordered_set<int> next_states;
    next_states.insert(0); // always include root
    for (int state : active_states) {
        if (state < 0 || state >= static_cast<int>(nodes_.size()))
            continue;
        auto it = nodes_[state].children.find(token_id);
        if (it != nodes_[state].children.end()) {
            next_states.insert(it->second);
        }
    }
    return next_states;
}

// ─── Boosted CTC Greedy Decode ─────────────────────────────────────────────

std::vector<std::vector<int>> ctc_greedy_decode_boosted(const Tensor &log_probs,
                                                        const ContextTrie &trie,
                                                        float boost_score,
                                                        int blank_id) {
    auto lp = log_probs.ascontiguousarray();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> &tokens = results[b];
        int prev = -1;
        std::unordered_set<int> active_states = {0};

        for (int t = 0; t < seq_len; ++t) {
            const float *frame = data + (b * seq_len + t) * vocab_size;

            // Get boosted tokens from trie
            auto boosted = trie.get_boosted_tokens(active_states);

            // Argmax with boost
            int best = 0;
            float best_val = frame[0] + (boosted.count(0) ? boost_score : 0.0f);
            for (int v = 1; v < vocab_size; ++v) {
                float val = frame[v] + (boosted.count(v) ? boost_score : 0.0f);
                if (val > best_val) {
                    best_val = val;
                    best = v;
                }
            }

            if (best != blank_id && best != prev) {
                tokens.push_back(best);
                // Advance trie on actual emission
                active_states = trie.advance(active_states, best);
            } else if (best == blank_id || best == prev) {
                // No emission — don't advance trie
            }
            prev = best;
        }
    }

    return results;
}

std::vector<std::vector<TimestampedToken>>
ctc_greedy_decode_with_timestamps_boosted(const Tensor &log_probs,
                                          const ContextTrie &trie,
                                          float boost_score, int blank_id) {
    auto lp = log_probs.ascontiguousarray();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto &tokens = results[b];
        int prev = -1;
        std::unordered_set<int> active_states = {0};

        for (int t = 0; t < seq_len; ++t) {
            const float *frame = data + (b * seq_len + t) * vocab_size;

            auto boosted = trie.get_boosted_tokens(active_states);

            int best = 0;
            float best_val = frame[0] + (boosted.count(0) ? boost_score : 0.0f);
            for (int v = 1; v < vocab_size; ++v) {
                float val = frame[v] + (boosted.count(v) ? boost_score : 0.0f);
                if (val > best_val) {
                    best_val = val;
                    best = v;
                }
            }

            // Use unboosted log-prob for confidence
            float raw_lp = frame[best];

            if (best != prev) {
                if (prev != -1 && prev != blank_id && !tokens.empty()) {
                    tokens.back().end_frame = t - 1;
                }
                if (best != blank_id) {
                    tokens.push_back({best, t, t, std::exp(raw_lp)});
                    active_states = trie.advance(active_states, best);
                }
            }
            prev = best;
        }

        if (!tokens.empty()) {
            tokens.back().end_frame = seq_len - 1;
        }
    }

    return results;
}

// ─── Boosted TDT Greedy Decode ─────────────────────────────────────────────

std::vector<std::vector<int>> tdt_greedy_decode_boosted(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score, int blank_id, int max_symbols_per_step) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)});

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}), Tensor::zeros({1, hs})};
        }

        auto token = Tensor({1}, DType::Int32);
        token.fill(blank_id);
        int t = 0;
        std::unordered_set<int> active_states = {0};

        while (t < seq_len) {
            auto enc_t = enc.slice({Slice(), Slice(t, t + 1)});

            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto saved_states = states;
                auto pred = prediction.step(token, states);
                pred = pred.unsqueeze(1);

                auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

                // Manual argmax with boost on label log-probs
                auto label_1d =
                    label_lp.squeeze(0).squeeze(0).cpu().ascontiguousarray();
                const float *label_data = label_1d.typed_data<float>();
                int vocab_size = static_cast<int>(label_1d.shape()[0]);

                auto boosted = trie.get_boosted_tokens(active_states);

                int token_id = 0;
                float best_lp =
                    label_data[0] + (boosted.count(0) ? boost_score : 0.0f);
                for (int v = 1; v < vocab_size; ++v) {
                    float val =
                        label_data[v] + (boosted.count(v) ? boost_score : 0.0f);
                    if (val > best_lp) {
                        best_lp = val;
                        token_id = v;
                    }
                }

                auto best_dur = ops::argmax(dur_lp.squeeze(0).squeeze(0), -1);
                int dur_idx = best_dur.item<int>();
                int skip = (dur_idx < static_cast<int>(durations.size()))
                               ? durations[dur_idx]
                               : 1;

                if (token_id == blank_id) {
                    states = saved_states;
                    t += std::max(skip, 1);
                    break;
                }

                results[b].push_back(token_id);
                active_states = trie.advance(active_states, token_id);

                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                if (skip > 0) {
                    t += skip;
                    break;
                }
            }
        }
    }

    return results;
}

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, const ContextTrie &trie,
    float boost_score, int blank_id, int max_symbols_per_step) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)});

        int num_layers = prediction.config().num_lstm_layers;
        size_t hs = prediction.config().pred_hidden;
        std::vector<LSTMState> states(num_layers);
        for (int l = 0; l < num_layers; ++l) {
            states[l] = {Tensor::zeros({1, hs}), Tensor::zeros({1, hs})};
        }

        auto token = Tensor({1}, DType::Int32);
        token.fill(blank_id);
        int t = 0;
        std::unordered_set<int> active_states = {0};

        while (t < seq_len) {
            auto enc_t = enc.slice({Slice(), Slice(t, t + 1)});

            for (int sym = 0; sym < max_symbols_per_step; ++sym) {
                auto saved_states = states;
                auto pred = prediction.step(token, states);
                pred = pred.unsqueeze(1);

                auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

                auto label_1d =
                    label_lp.squeeze(0).squeeze(0).cpu().ascontiguousarray();
                const float *label_data = label_1d.typed_data<float>();
                int vocab_size = static_cast<int>(label_1d.shape()[0]);

                auto boosted = trie.get_boosted_tokens(active_states);

                int token_id = 0;
                float best_lp =
                    label_data[0] + (boosted.count(0) ? boost_score : 0.0f);
                for (int v = 1; v < vocab_size; ++v) {
                    float val =
                        label_data[v] + (boosted.count(v) ? boost_score : 0.0f);
                    if (val > best_lp) {
                        best_lp = val;
                        token_id = v;
                    }
                }
                // Use raw (unboosted) log-prob for confidence
                float raw_lp = label_data[token_id];
                float confidence = std::exp(raw_lp);

                auto best_dur = ops::argmax(dur_lp.squeeze(0).squeeze(0), -1);
                int dur_idx = best_dur.item<int>();
                int skip = (dur_idx < static_cast<int>(durations.size()))
                               ? durations[dur_idx]
                               : 1;

                if (token_id == blank_id) {
                    states = saved_states;
                    t += std::max(skip, 1);
                    break;
                }

                int end_frame = t + std::max(skip, 1) - 1;
                if (end_frame >= seq_len)
                    end_frame = seq_len - 1;
                results[b].push_back({token_id, t, end_frame, confidence});

                active_states = trie.advance(active_states, token_id);

                token = Tensor({1}, DType::Int32);
                token.fill(token_id);

                if (skip > 0) {
                    t += skip;
                    break;
                }
            }
        }
    }

    return results;
}

// ─── Convenience wrappers ──────────────────────────────────────────────────

std::vector<std::vector<int>>
tdt_greedy_decode_boosted(ParakeetTDT &model, const Tensor &encoder_out,
                          const std::vector<int> &durations,
                          const ContextTrie &trie, float boost_score,
                          int blank_id, int max_symbols_per_step) {
    return tdt_greedy_decode_boosted(model.prediction(), model.joint(),
                                     encoder_out, durations, trie, boost_score,
                                     blank_id, max_symbols_per_step);
}

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(ParakeetTDT &model,
                                          const Tensor &encoder_out,
                                          const std::vector<int> &durations,
                                          const ContextTrie &trie,
                                          float boost_score, int blank_id,
                                          int max_symbols_per_step) {
    return tdt_greedy_decode_with_timestamps_boosted(
        model.prediction(), model.joint(), encoder_out, durations, trie,
        boost_score, blank_id, max_symbols_per_step);
}

std::vector<std::vector<int>>
tdt_greedy_decode_boosted(ParakeetTDTCTC &model, const Tensor &encoder_out,
                          const std::vector<int> &durations,
                          const ContextTrie &trie, float boost_score,
                          int blank_id, int max_symbols_per_step) {
    return tdt_greedy_decode_boosted(model.prediction(), model.tdt_joint(),
                                     encoder_out, durations, trie, boost_score,
                                     blank_id, max_symbols_per_step);
}

std::vector<std::vector<TimestampedToken>>
tdt_greedy_decode_with_timestamps_boosted(ParakeetTDTCTC &model,
                                          const Tensor &encoder_out,
                                          const std::vector<int> &durations,
                                          const ContextTrie &trie,
                                          float boost_score, int blank_id,
                                          int max_symbols_per_step) {
    return tdt_greedy_decode_with_timestamps_boosted(
        model.prediction(), model.tdt_joint(), encoder_out, durations, trie,
        boost_score, blank_id, max_symbols_per_step);
}

} // namespace parakeet
