#pragma once

// Umbrella header — includes all parakeet modules

// Audio
#include "parakeet/audio/audio.hpp"
#include "parakeet/audio/audio_io.hpp"
#include "parakeet/audio/vad.hpp"

// Models
#include "parakeet/models/config.hpp"
#include "parakeet/models/ctc.hpp"
#include "parakeet/models/encoder.hpp"
#include "parakeet/models/eou.hpp"
#include "parakeet/models/lstm.hpp"
#include "parakeet/models/nemotron.hpp"
#include "parakeet/models/rnnt.hpp"
#include "parakeet/models/silero_vad.hpp"
#include "parakeet/models/sortformer.hpp"
#include "parakeet/models/streaming_encoder.hpp"
#include "parakeet/models/tdt.hpp"
#include "parakeet/models/tdt_ctc.hpp"
#include "parakeet/models/transformer.hpp"

// Decode
#include "parakeet/decode/arpa_lm.hpp"
#include "parakeet/decode/beam_search.hpp"
#include "parakeet/decode/phrase_boost.hpp"
#include "parakeet/decode/tdt_beam_search.hpp"
#include "parakeet/decode/timestamp.hpp"
#include "parakeet/decode/vocab.hpp"

// API
#include "parakeet/api/diarize.hpp"
#include "parakeet/api/transcribe.hpp"

// Backward compatibility: re-export sub-namespaces into parakeet::
namespace parakeet {
using namespace audio;
using namespace models;
using namespace decode;
using namespace api;
} // namespace parakeet
