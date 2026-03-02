#pragma once

#include <string>
#include <vector>

#include <axiom/axiom.hpp>

#include "parakeet/decode/arpa_lm.hpp"
#include "parakeet/decode/timestamp.hpp"
#include "parakeet/models/rnnt.hpp"
#include "parakeet/models/tdt.hpp"
#include "parakeet/models/tdt_ctc.hpp"

namespace parakeet::decode {

// ─── TDT Beam Search ─────────────────────────────────────────────────────────
//
// Time-synchronous beam search for TDT models. Each hypothesis carries its
// own autoregressive LSTM decoder states and a `next_frame` pointer that
// tracks which encoder frame it needs to process next (because TDT's
// duration model allows variable frame advancement).
//
// At each frame t, only hypotheses with next_frame == t are expanded;
// others are carried forward unchanged.
//
// Optional n-gram LM fusion: applies LM score at word boundaries
// (SentencePiece ▁ tokens) during beam expansion.

struct TDTBeamSearchOptions {
    int beam_width = 4;
    int blank_id = 1024;
    int max_symbols_per_step = 10;

    // Language model (optional)
    const ArpaLM *lm = nullptr;
    float lm_weight = 0.5f;

    // Piece table for word boundary detection during LM scoring.
    // Required when lm is non-null.
    const std::vector<std::string> *pieces = nullptr;
};

// ─── Component-based API ─────────────────────────────────────────────────────

// TDT beam search decode using prediction + joint components directly.
// encoder_out: (batch, seq, hidden)
// Returns per-batch token sequences.
std::vector<std::vector<int>>
tdt_beam_decode(models::RNNTPrediction &prediction, models::TDTJoint &joint,
                const axiom::Tensor &encoder_out,
                const std::vector<int> &durations,
                const TDTBeamSearchOptions &opts = {},
                const std::vector<int> &lengths = {});

// TDT beam search with timestamps.
std::vector<std::vector<TimestampedToken>> tdt_beam_decode_with_timestamps(
    models::RNNTPrediction &prediction, models::TDTJoint &joint,
    const axiom::Tensor &encoder_out, const std::vector<int> &durations,
    const TDTBeamSearchOptions &opts = {},
    const std::vector<int> &lengths = {});

// ─── ParakeetTDT convenience overloads ───────────────────────────────────────

std::vector<std::vector<int>>
tdt_beam_decode(models::ParakeetTDT &model, const axiom::Tensor &encoder_out,
                const std::vector<int> &durations,
                const TDTBeamSearchOptions &opts = {},
                const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>> tdt_beam_decode_with_timestamps(
    models::ParakeetTDT &model, const axiom::Tensor &encoder_out,
    const std::vector<int> &durations, const TDTBeamSearchOptions &opts = {},
    const std::vector<int> &lengths = {});

// ─── ParakeetTDTCTC convenience overloads ────────────────────────────────────

std::vector<std::vector<int>>
tdt_beam_decode(models::ParakeetTDTCTC &model, const axiom::Tensor &encoder_out,
                const std::vector<int> &durations,
                const TDTBeamSearchOptions &opts = {},
                const std::vector<int> &lengths = {});

std::vector<std::vector<TimestampedToken>> tdt_beam_decode_with_timestamps(
    models::ParakeetTDTCTC &model, const axiom::Tensor &encoder_out,
    const std::vector<int> &durations, const TDTBeamSearchOptions &opts = {},
    const std::vector<int> &lengths = {});

} // namespace parakeet::decode
