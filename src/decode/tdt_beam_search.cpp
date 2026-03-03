#include "parakeet/decode/tdt_beam_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace parakeet::decode {

using namespace axiom;
using namespace models;

// ─── Internal Types ─────────────────────────────────────────────────────────

namespace {

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// SentencePiece word boundary marker: U+2581 (▁)
const std::string WORD_BOUNDARY = "\xe2\x96\x81";

bool is_word_start(int token_id, const std::vector<std::string> *pieces) {
    if (!pieces || token_id < 0 || token_id >= static_cast<int>(pieces->size()))
        return false;
    const auto &piece = (*pieces)[token_id];
    return piece.size() >= 3 && piece.compare(0, 3, WORD_BOUNDARY) == 0;
}

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

std::string get_current_word(const std::vector<int> &prefix,
                             const std::vector<std::string> *pieces) {
    if (!pieces || prefix.empty())
        return "";
    std::string word;
    for (int i = static_cast<int>(prefix.size()) - 1; i >= 0; --i) {
        if (is_word_start(prefix[i], pieces)) {
            word.clear();
            for (int j = i; j < static_cast<int>(prefix.size()); ++j) {
                word += piece_to_word(prefix[j], pieces);
            }
            return word;
        }
    }
    for (int id : prefix) {
        word += piece_to_word(id, pieces);
    }
    return word;
}

struct TDTHypothesis {
    std::vector<int> tokens;
    float score = 0.0f;
    std::vector<LSTMState> states;
    int last_token;
    int next_frame = 0;

    // LM state
    float lm_score = 0.0f;
    ArpaLM::State lm_state;

    // For timestamps
    std::vector<int> emission_frames;
    std::vector<float> token_log_probs;

    float combined_score() const { return score + lm_score; }
};

// Expand a single hypothesis at encoder frame t.
// Adds resulting candidates to `candidates`.
void expand_hypothesis(TDTHypothesis &hyp, const Tensor &enc_t,
                       RNNTPrediction &prediction, TDTJoint &joint,
                       const std::vector<int> &durations, int t, int T,
                       int remaining, const TDTBeamSearchOptions &opts,
                       std::vector<TDTHypothesis> &candidates) {
    if (remaining <= 0)
        return;

    int blank_id = opts.blank_id;
    bool use_lm = opts.lm && opts.lm->loaded() && opts.pieces;

    // Save LSTM states before prediction step
    auto saved_states = hyp.states;

    auto token_tensor = Tensor({1}, DType::Int32);
    token_tensor.fill(hyp.last_token);

    auto pred = prediction.step(token_tensor, hyp.states);
    pred = pred.unsqueeze(1); // (1, 1, pred_hidden)

    auto [label_lp, dur_lp] = joint.forward(enc_t, pred);

    // Flatten and move to CPU for reading
    auto label_1d = label_lp.squeeze(0).squeeze(0).to_contiguous_cpu();
    auto dur_1d = dur_lp.squeeze(0).squeeze(0).to_contiguous_cpu();

    const float *label_data = label_1d.typed_data<float>();
    const float *dur_data = dur_1d.typed_data<float>();
    int vocab_size = static_cast<int>(label_1d.shape()[0]);
    int num_durs = static_cast<int>(dur_1d.shape()[0]);

    // Greedy duration: argmax on duration head
    int best_dur_idx = 0;
    float best_dur_lp = dur_data[0];
    for (int d = 1; d < num_durs; d++) {
        if (dur_data[d] > best_dur_lp) {
            best_dur_lp = dur_data[d];
            best_dur_idx = d;
        }
    }
    int skip = (best_dur_idx < static_cast<int>(durations.size()))
                   ? durations[best_dur_idx]
                   : 1;

    // Collect top-k tokens
    struct TokenScore {
        int id;
        float score;
    };
    std::vector<TokenScore> top_tokens;
    top_tokens.reserve(vocab_size);
    for (int v = 0; v < vocab_size; ++v) {
        if (label_data[v] < -30.0f)
            continue;
        top_tokens.push_back({v, label_data[v]});
    }
    int k = std::min(static_cast<int>(top_tokens.size()), opts.beam_width * 2);
    if (k < static_cast<int>(top_tokens.size())) {
        std::partial_sort(top_tokens.begin(), top_tokens.begin() + k,
                          top_tokens.end(),
                          [](const TokenScore &a, const TokenScore &b) {
                              return a.score > b.score;
                          });
        top_tokens.resize(k);
    }

    for (const auto &[token_id, log_p] : top_tokens) {
        if (token_id == blank_id) {
            // Blank: revert LSTM states, advance by max(skip, 1)
            TDTHypothesis new_hyp;
            new_hyp.tokens = hyp.tokens;
            new_hyp.score = hyp.score + log_p;
            new_hyp.states = saved_states;
            new_hyp.last_token = hyp.last_token;
            new_hyp.next_frame = t + std::max(skip, 1);
            new_hyp.lm_score = hyp.lm_score;
            new_hyp.lm_state = hyp.lm_state;
            new_hyp.emission_frames = hyp.emission_frames;
            new_hyp.token_log_probs = hyp.token_log_probs;
            candidates.push_back(std::move(new_hyp));
        } else if (skip > 0) {
            // Non-blank with skip > 0: emit token, advance frames
            TDTHypothesis new_hyp;
            new_hyp.tokens = hyp.tokens;
            new_hyp.tokens.push_back(token_id);
            new_hyp.score = hyp.score + log_p;
            new_hyp.states = hyp.states; // keep updated LSTM states
            new_hyp.last_token = token_id;
            new_hyp.next_frame = t + skip;
            new_hyp.emission_frames = hyp.emission_frames;
            new_hyp.emission_frames.push_back(t);
            new_hyp.token_log_probs = hyp.token_log_probs;
            new_hyp.token_log_probs.push_back(log_p);

            // LM scoring at word boundaries
            new_hyp.lm_score = hyp.lm_score;
            new_hyp.lm_state = hyp.lm_state;
            if (use_lm && is_word_start(token_id, opts.pieces)) {
                std::string word = get_current_word(hyp.tokens, opts.pieces);
                if (!word.empty()) {
                    new_hyp.lm_state = hyp.lm_state;
                    float lm_lp = opts.lm->score(new_hyp.lm_state, word);
                    new_hyp.lm_score +=
                        opts.lm_weight * lm_lp * std::log(10.0f);
                }
            }

            candidates.push_back(std::move(new_hyp));
        } else {
            // Non-blank with skip == 0: emit token, recurse on same frame
            TDTHypothesis recurse_hyp;
            recurse_hyp.tokens = hyp.tokens;
            recurse_hyp.tokens.push_back(token_id);
            recurse_hyp.score = hyp.score + log_p;
            recurse_hyp.states = hyp.states;
            recurse_hyp.last_token = token_id;
            recurse_hyp.next_frame = t;
            recurse_hyp.emission_frames = hyp.emission_frames;
            recurse_hyp.emission_frames.push_back(t);
            recurse_hyp.token_log_probs = hyp.token_log_probs;
            recurse_hyp.token_log_probs.push_back(log_p);

            recurse_hyp.lm_score = hyp.lm_score;
            recurse_hyp.lm_state = hyp.lm_state;
            if (use_lm && is_word_start(token_id, opts.pieces)) {
                std::string word = get_current_word(hyp.tokens, opts.pieces);
                if (!word.empty()) {
                    recurse_hyp.lm_state = hyp.lm_state;
                    float lm_lp = opts.lm->score(recurse_hyp.lm_state, word);
                    recurse_hyp.lm_score +=
                        opts.lm_weight * lm_lp * std::log(10.0f);
                }
            }

            // Recursively expand this new hypothesis on the same frame
            expand_hypothesis(recurse_hyp, enc_t, prediction, joint, durations,
                              t, T, remaining - 1, opts, candidates);
        }
    }
}

// Run TDT beam search on a single batch element.
TDTHypothesis tdt_beam_search_single(const Tensor &enc, int T,
                                     RNNTPrediction &prediction,
                                     TDTJoint &joint,
                                     const std::vector<int> &durations,
                                     const TDTBeamSearchOptions &opts) {
    int beam_width = opts.beam_width;
    int blank_id = opts.blank_id;
    bool use_lm = opts.lm && opts.lm->loaded() && opts.pieces;

    int num_layers = prediction.config().num_lstm_layers;
    size_t hs = prediction.config().pred_hidden;

    // Initialize beam with a single empty hypothesis
    std::vector<TDTHypothesis> beam(1);
    beam[0].score = 0.0f;
    beam[0].last_token = blank_id;
    beam[0].next_frame = 0;
    beam[0].states.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        beam[0].states[l] = {Tensor::zeros({1, hs}, enc.dtype()),
                             Tensor::zeros({1, hs}, enc.dtype())};
    }
    if (use_lm)
        beam[0].lm_state = opts.lm->initial_state();

    for (int t = 0; t < T; ++t) {
        auto enc_t = enc.slice({Slice(), Slice(t, t + 1)}); // (1, 1, hidden)

        // Partition beam into ready (next_frame == t) and waiting (> t)
        std::vector<TDTHypothesis> candidates;

        for (auto &hyp : beam) {
            if (hyp.next_frame == t) {
                expand_hypothesis(hyp, enc_t, prediction, joint, durations, t,
                                  T, opts.max_symbols_per_step, opts,
                                  candidates);
            } else {
                // Carry forward: not ready yet
                candidates.push_back(std::move(hyp));
            }
        }

        // Prune to beam_width
        if (static_cast<int>(candidates.size()) > beam_width) {
            std::partial_sort(
                candidates.begin(), candidates.begin() + beam_width,
                candidates.end(),
                [](const TDTHypothesis &a, const TDTHypothesis &b) {
                    return a.combined_score() > b.combined_score();
                });
            candidates.resize(beam_width);
        }

        beam = std::move(candidates);

        if (beam.empty())
            break;
    }

    if (beam.empty()) {
        return TDTHypothesis{};
    }

    auto best =
        std::max_element(beam.begin(), beam.end(),
                         [](const TDTHypothesis &a, const TDTHypothesis &b) {
                             return a.combined_score() < b.combined_score();
                         });

    return *best;
}

} // anonymous namespace

// ─── Public API ─────────────────────────────────────────────────────────────

std::vector<std::vector<int>> tdt_beam_decode(RNNTPrediction &prediction,
                                              TDTJoint &joint,
                                              const Tensor &encoder_out,
                                              const std::vector<int> &durations,
                                              const TDTBeamSearchOptions &opts,
                                              const std::vector<int> &lengths) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)});
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        auto best =
            tdt_beam_search_single(enc, T, prediction, joint, durations, opts);
        results[b] = std::move(best.tokens);
    }

    return results;
}

std::vector<std::vector<TimestampedToken>> tdt_beam_decode_with_timestamps(
    RNNTPrediction &prediction, TDTJoint &joint, const Tensor &encoder_out,
    const std::vector<int> &durations, const TDTBeamSearchOptions &opts,
    const std::vector<int> &lengths) {
    auto shape = encoder_out.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto enc = encoder_out.slice({Slice(b, b + 1)});
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        auto best =
            tdt_beam_search_single(enc, T, prediction, joint, durations, opts);

        auto &tokens = results[b];
        tokens.reserve(best.tokens.size());

        for (size_t i = 0; i < best.tokens.size(); ++i) {
            int frame =
                (i < best.emission_frames.size()) ? best.emission_frames[i] : 0;
            float log_p = (i < best.token_log_probs.size())
                              ? best.token_log_probs[i]
                              : -1.0f;
            float confidence = std::exp(log_p);

            // Estimate end_frame from next token or T-1
            int end_frame = (i + 1 < best.emission_frames.size())
                                ? best.emission_frames[i + 1] - 1
                                : T - 1;
            if (end_frame < frame)
                end_frame = frame;

            tokens.push_back({best.tokens[i], frame, end_frame, confidence});
        }
    }

    return results;
}

// ─── ParakeetTDT convenience overloads ───────────────────────────────────────

std::vector<std::vector<int>> tdt_beam_decode(ParakeetTDT &model,
                                              const Tensor &encoder_out,
                                              const std::vector<int> &durations,
                                              const TDTBeamSearchOptions &opts,
                                              const std::vector<int> &lengths) {
    return tdt_beam_decode(model.prediction(), model.joint(), encoder_out,
                           durations, opts, lengths);
}

std::vector<std::vector<TimestampedToken>>
tdt_beam_decode_with_timestamps(ParakeetTDT &model, const Tensor &encoder_out,
                                const std::vector<int> &durations,
                                const TDTBeamSearchOptions &opts,
                                const std::vector<int> &lengths) {
    return tdt_beam_decode_with_timestamps(model.prediction(), model.joint(),
                                           encoder_out, durations, opts,
                                           lengths);
}

// ─── ParakeetTDTCTC convenience overloads ────────────────────────────────────

std::vector<std::vector<int>> tdt_beam_decode(ParakeetTDTCTC &model,
                                              const Tensor &encoder_out,
                                              const std::vector<int> &durations,
                                              const TDTBeamSearchOptions &opts,
                                              const std::vector<int> &lengths) {
    return tdt_beam_decode(model.prediction(), model.tdt_joint(), encoder_out,
                           durations, opts, lengths);
}

std::vector<std::vector<TimestampedToken>> tdt_beam_decode_with_timestamps(
    ParakeetTDTCTC &model, const Tensor &encoder_out,
    const std::vector<int> &durations, const TDTBeamSearchOptions &opts,
    const std::vector<int> &lengths) {
    return tdt_beam_decode_with_timestamps(model.prediction(),
                                           model.tdt_joint(), encoder_out,
                                           durations, opts, lengths);
}

} // namespace parakeet::decode
