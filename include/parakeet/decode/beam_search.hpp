#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>

#include "parakeet/decode/arpa_lm.hpp"
#include "parakeet/decode/timestamp.hpp"

namespace parakeet::decode {

// ─── CTC Prefix Beam Search ─────────────────────────────────────────────────
//
// Standard CTC prefix beam search algorithm. Maintains a beam of hypothesis
// prefixes, each with separate blank/non-blank ending probabilities.
// Merges duplicate prefixes and prunes to top-k at each frame.
//
// Optional n-gram LM fusion: applies LM score at word boundaries
// (SentencePiece ▁ tokens) during beam expansion.

struct BeamSearchOptions {
    int beam_width = 8;
    int blank_id = 1024;

    // Language model (optional)
    const ArpaLM *lm = nullptr;
    float lm_weight = 0.5f;

    // Piece table for word boundary detection during LM scoring.
    // Required when lm is non-null.
    const std::vector<std::string> *pieces = nullptr;
};

// CTC prefix beam search decode.
// log_probs: (batch, seq, vocab) — log probabilities from CTC head.
// Returns per-batch token sequences (same format as ctc_greedy_decode).
std::vector<std::vector<int>>
ctc_beam_decode(const axiom::Tensor &log_probs,
                const BeamSearchOptions &opts = {},
                const std::vector<int> &lengths = {});

// CTC prefix beam search with timestamps.
// Returns per-batch timestamped token sequences.
std::vector<std::vector<TimestampedToken>>
ctc_beam_decode_with_timestamps(const axiom::Tensor &log_probs,
                                const BeamSearchOptions &opts = {},
                                const std::vector<int> &lengths = {});

} // namespace parakeet::decode
