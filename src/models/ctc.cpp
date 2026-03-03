#include "parakeet/models/ctc.hpp"

#include <cmath>
#include <iostream>

namespace parakeet::models {

// ─── CTCDecoder ─────────────────────────────────────────────────────────────

CTCDecoder::CTCDecoder() { AX_REGISTER_MODULE(proj_); }

Tensor CTCDecoder::forward(const Tensor &input) const {
    // input: (batch, seq, hidden) → Conv1d expects (batch, hidden, seq)
    auto x = input.transpose({0, 2, 1});
    x = proj_(x);
    // output: (batch, vocab, seq) → (batch, seq, vocab)
    x = x.transpose({0, 2, 1});

    // Move to CPU for log_softmax (GPU dispatch differs).
    // Contiguity is now handled inside axiom's softmax implementation.
    x = x.cpu();
    return ops::log_softmax(x, /*axis=*/-1);
}

// ─── ParakeetCTC ────────────────────────────────────────────────────────────

ParakeetCTC::ParakeetCTC(const CTCConfig &config)
    : config_(config), encoder_(config.encoder) {
    AX_REGISTER_MODULES(encoder_, decoder_);
}

Tensor ParakeetCTC::forward(const Tensor &input, const Tensor &mask) const {
    return decoder_(encoder_(input, mask));
}

// ─── CTC Greedy Decode ─────────────────────────────────────────────────────

std::vector<std::vector<int>>
ctc_greedy_decode(const Tensor &log_probs, int blank_id,
                  const std::vector<int> &lengths) {
    // log_probs: (batch, seq, vocab)
    auto lp = log_probs.to_contiguous_cpu();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> &tokens = results[b];
        int prev = -1;
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        for (int t = 0; t < T; ++t) {
            // Manual argmax along vocab dimension
            const float *frame = data + (b * seq_len + t) * vocab_size;
            int best = 0;
            float best_val = frame[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (frame[v] > best_val) {
                    best_val = frame[v];
                    best = v;
                }
            }
            if (best != blank_id && best != prev) {
                tokens.push_back(best);
            }
            prev = best;
        }
    }

    return results;
}

// ─── Timestamped CTC Greedy Decode ────────────────────────────────────────

std::vector<std::vector<TimestampedToken>>
ctc_greedy_decode_with_timestamps(const Tensor &log_probs, int blank_id,
                                  const std::vector<int> &lengths) {
    auto lp = log_probs.to_contiguous_cpu();
    auto shape = lp.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);
    int vocab_size = static_cast<int>(shape[2]);
    const float *data = lp.typed_data<float>();

    std::vector<std::vector<TimestampedToken>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto &tokens = results[b];
        int prev = -1;
        int token_start_frame = 0;
        int T = (!lengths.empty() && b < static_cast<int>(lengths.size()))
                    ? lengths[b]
                    : seq_len;

        for (int t = 0; t < T; ++t) {
            const float *frame = data + (b * seq_len + t) * vocab_size;
            int best = 0;
            float best_val = frame[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (frame[v] > best_val) {
                    best_val = frame[v];
                    best = v;
                }
            }

            if (best != prev) {
                // Token changed — if previous was non-blank, close its span
                if (prev != -1 && prev != blank_id && !tokens.empty()) {
                    tokens.back().end_frame = t - 1;
                }
                // If new token is non-blank, start a new span
                if (best != blank_id) {
                    tokens.push_back({best, t, t, std::exp(best_val)});
                }
                token_start_frame = t;
            }
            prev = best;
        }

        // Close last token span
        if (!tokens.empty()) {
            tokens.back().end_frame = T - 1;
        }
    }

    return results;
}

} // namespace parakeet::models
