#include "parakeet/ctc.hpp"

namespace parakeet {

// ─── CTCDecoder ─────────────────────────────────────────────────────────────

CTCDecoder::CTCDecoder() : proj_(true) { AX_REGISTER_MODULE(proj_); }

Tensor CTCDecoder::forward(const Tensor &input) const {
    return ops::log_softmax(proj_(input), /*axis=*/-1);
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

std::vector<std::vector<int>> ctc_greedy_decode(const Tensor &log_probs,
                                                int blank_id) {
    // log_probs: (batch, seq, vocab)
    auto predictions = ops::argmax(log_probs, /*axis=*/-1); // (batch, seq)
    auto shape = predictions.shape();
    int batch_size = static_cast<int>(shape[0]);
    int seq_len = static_cast<int>(shape[1]);

    std::vector<std::vector<int>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        auto batch_preds =
            predictions.slice({Slice(b, b + 1)}).squeeze(0); // (seq,)
        std::vector<int> &tokens = results[b];
        int prev = -1;

        for (int t = 0; t < seq_len; ++t) {
            int token = batch_preds.slice({Slice(t, t + 1)}).item<int>();
            if (token != blank_id && token != prev) {
                tokens.push_back(token);
            }
            prev = token;
        }
    }

    return results;
}

} // namespace parakeet
