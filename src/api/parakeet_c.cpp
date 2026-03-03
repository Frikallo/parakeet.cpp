/**
 * parakeet_c.cpp — C API implementation for parakeet.cpp
 *
 * Each opaque handle wraps a C++ object. Thread-local error string for
 * error reporting. All fallible functions use PARAKEET_TRY/PARAKEET_CATCH.
 */

#include "parakeet/api/parakeet_c.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <axiom/axiom.hpp>

#include "parakeet/api/diarize.hpp"
#include "parakeet/api/transcribe.hpp"
#include "parakeet/audio/audio_io.hpp"
#include "parakeet/audio/vad.hpp"
#include "parakeet/models/config.hpp"
#include "parakeet/models/eou.hpp"
#include "parakeet/models/nemotron.hpp"

// Re-export sub-namespaces into parakeet:: for backward compat in this file
namespace parakeet {
using namespace audio;
using namespace models;
using namespace decode;
using namespace api;
} // namespace parakeet

// ── Thread-local error ──────────────────────────────────────────────────────

static thread_local std::string g_last_error;

#define PARAKEET_TRY try {
#define PARAKEET_CATCH(ret)                                                    \
    }                                                                          \
    catch (const std::exception &e) {                                          \
        g_last_error = e.what();                                               \
        return (ret);                                                          \
    }

#define NULL_CHECK(h, ret)                                                     \
    if (!(h)) {                                                                \
        g_last_error = "null handle";                                          \
        return (ret);                                                          \
    }

// ── Internal wrapper structs ────────────────────────────────────────────────

struct parakeet_config_s {
    enum Type {
        TDT_CTC_110M,
        TDT_600M,
        EOU_120M,
        NEMOTRON_600M,
        SORTFORMER_117M
    } type;
    void *ptr; // heap-allocated config
    ~parakeet_config_s() {
        switch (type) {
        case TDT_CTC_110M:
            delete static_cast<parakeet::TDTCTCConfig *>(ptr);
            break;
        case TDT_600M:
            delete static_cast<parakeet::TDTConfig *>(ptr);
            break;
        case EOU_120M:
            delete static_cast<parakeet::EOUConfig *>(ptr);
            break;
        case NEMOTRON_600M:
            delete static_cast<parakeet::NemotronConfig *>(ptr);
            break;
        case SORTFORMER_117M:
            delete static_cast<parakeet::SortformerConfig *>(ptr);
            break;
        }
    }
};

struct parakeet_options_s {
    parakeet::TranscribeOptions opts;
};

struct parakeet_transcriber_s {
    std::unique_ptr<parakeet::Transcriber> impl;
};

struct parakeet_tdt_transcriber_s {
    std::unique_ptr<parakeet::TDTTranscriber> impl;
};

struct parakeet_streaming_transcriber_s {
    std::unique_ptr<parakeet::StreamingTranscriber> impl;
    std::string last_chunk_text;
    std::string full_text_cache;
};

struct parakeet_nemotron_transcriber_s {
    std::unique_ptr<parakeet::NemotronTranscriber> impl;
    std::string last_chunk_text;
    std::string full_text_cache;
};

struct parakeet_diarized_transcriber_s {
    std::unique_ptr<parakeet::DiarizedTranscriber> impl;
};

struct parakeet_result_s {
    parakeet::TranscribeResult result;
};

struct parakeet_diarized_result_s {
    parakeet::DiarizedResult result;
};

struct parakeet_audio_s {
    parakeet::AudioData audio;
    std::vector<float> cached_samples;
    bool samples_cached = false;
};

// ── Helpers ─────────────────────────────────────────────────────────────────

static parakeet::Decoder to_cpp_decoder(parakeet_decoder_t d) {
    return d == PARAKEET_DECODER_CTC ? parakeet::Decoder::CTC
                                     : parakeet::Decoder::TDT;
}

static parakeet_audio_format_t to_c_format(parakeet::AudioFormat f) {
    switch (f) {
    case parakeet::AudioFormat::WAV:
        return PARAKEET_AUDIO_FORMAT_WAV;
    case parakeet::AudioFormat::FLAC:
        return PARAKEET_AUDIO_FORMAT_FLAC;
    case parakeet::AudioFormat::MP3:
        return PARAKEET_AUDIO_FORMAT_MP3;
    case parakeet::AudioFormat::OGG:
        return PARAKEET_AUDIO_FORMAT_OGG;
    default:
        return PARAKEET_AUDIO_FORMAT_UNKNOWN;
    }
}

static const parakeet::TranscribeOptions &
resolve_opts(parakeet_options_t opts) {
    static const parakeet::TranscribeOptions default_opts;
    return opts ? opts->opts : default_opts;
}

// ── Version ─────────────────────────────────────────────────────────────────

extern "C" const char *parakeet_version(void) { return "0.1.0"; }

// ── Error Handling ──────────────────────────────────────────────────────────

extern "C" const char *parakeet_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

extern "C" void parakeet_clear_error(void) { g_last_error.clear(); }

// ── Config Presets ──────────────────────────────────────────────────────────

extern "C" parakeet_config_t parakeet_config_110m(void) {
    auto c = new parakeet_config_s;
    c->type = parakeet_config_s::TDT_CTC_110M;
    c->ptr = new parakeet::TDTCTCConfig(parakeet::make_110m_config());
    return c;
}

extern "C" parakeet_config_t parakeet_config_tdt_600m(void) {
    auto c = new parakeet_config_s;
    c->type = parakeet_config_s::TDT_600M;
    c->ptr = new parakeet::TDTConfig(parakeet::make_tdt_600m_config());
    return c;
}

extern "C" parakeet_config_t parakeet_config_eou_120m(void) {
    auto c = new parakeet_config_s;
    c->type = parakeet_config_s::EOU_120M;
    c->ptr = new parakeet::EOUConfig(parakeet::make_eou_120m_config());
    return c;
}

extern "C" parakeet_config_t parakeet_config_nemotron_600m(int latency_frames) {
    auto c = new parakeet_config_s;
    c->type = parakeet_config_s::NEMOTRON_600M;
    c->ptr = new parakeet::NemotronConfig(
        parakeet::make_nemotron_600m_config(latency_frames));
    return c;
}

extern "C" parakeet_config_t parakeet_config_sortformer_117m(void) {
    auto c = new parakeet_config_s;
    c->type = parakeet_config_s::SORTFORMER_117M;
    c->ptr =
        new parakeet::SortformerConfig(parakeet::make_sortformer_117m_config());
    return c;
}

extern "C" void parakeet_config_free(parakeet_config_t config) {
    delete config;
}

// ── Transcribe Options ──────────────────────────────────────────────────────

extern "C" parakeet_options_t parakeet_options_create(void) {
    return new parakeet_options_s;
}

extern "C" void parakeet_options_free(parakeet_options_t opts) { delete opts; }

extern "C" parakeet_error_t
parakeet_options_set_decoder(parakeet_options_t opts,
                             parakeet_decoder_t decoder) {
    NULL_CHECK(opts, PARAKEET_ERROR_NULL_HANDLE);
    opts->opts.decoder = to_cpp_decoder(decoder);
    return PARAKEET_OK;
}

extern "C" parakeet_error_t
parakeet_options_set_timestamps(parakeet_options_t opts, int enable) {
    NULL_CHECK(opts, PARAKEET_ERROR_NULL_HANDLE);
    opts->opts.timestamps = (enable != 0);
    return PARAKEET_OK;
}

extern "C" parakeet_error_t
parakeet_options_set_boost_score(parakeet_options_t opts, float score) {
    NULL_CHECK(opts, PARAKEET_ERROR_NULL_HANDLE);
    opts->opts.boost_score = score;
    return PARAKEET_OK;
}

extern "C" parakeet_error_t
parakeet_options_add_boost_phrase(parakeet_options_t opts, const char *phrase) {
    NULL_CHECK(opts, PARAKEET_ERROR_NULL_HANDLE);
    if (!phrase) {
        g_last_error = "phrase is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    opts->opts.boost_phrases.emplace_back(phrase);
    return PARAKEET_OK;
}

// ── Audio I/O ───────────────────────────────────────────────────────────────

extern "C" parakeet_audio_t parakeet_audio_read_file(const char *path,
                                                     int target_sample_rate) {
    if (!path) {
        g_last_error = "path is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto a = new parakeet_audio_s;
    a->audio = parakeet::read_audio(path, target_sample_rate);
    return a;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_audio_t parakeet_audio_read_f32(const float *pcm, size_t n,
                                                    int rate, int target_rate) {
    if (!pcm) {
        g_last_error = "pcm is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto a = new parakeet_audio_s;
    a->audio = parakeet::read_audio(pcm, n, rate, target_rate);
    return a;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_audio_t parakeet_audio_read_i16(const int16_t *pcm,
                                                    size_t n, int rate,
                                                    int target_rate) {
    if (!pcm) {
        g_last_error = "pcm is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto a = new parakeet_audio_s;
    a->audio = parakeet::read_audio(pcm, n, rate, target_rate);
    return a;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_audio_t
parakeet_audio_read_encoded(const uint8_t *data, size_t len, int target_rate) {
    if (!data) {
        g_last_error = "data is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto a = new parakeet_audio_s;
    a->audio = parakeet::read_audio(data, len, target_rate);
    return a;
    PARAKEET_CATCH(nullptr)
}

extern "C" void parakeet_audio_free(parakeet_audio_t audio) { delete audio; }

extern "C" int parakeet_audio_sample_rate(parakeet_audio_t audio) {
    return audio ? audio->audio.sample_rate : 0;
}

extern "C" int parakeet_audio_num_samples(parakeet_audio_t audio) {
    return audio ? audio->audio.num_samples : 0;
}

extern "C" float parakeet_audio_duration(parakeet_audio_t audio) {
    return audio ? audio->audio.duration : 0.0f;
}

extern "C" parakeet_audio_format_t
parakeet_audio_format(parakeet_audio_t audio) {
    return audio ? to_c_format(audio->audio.format)
                 : PARAKEET_AUDIO_FORMAT_UNKNOWN;
}

extern "C" const float *parakeet_audio_samples(parakeet_audio_t audio,
                                               size_t *out_count) {
    if (!audio) {
        if (out_count)
            *out_count = 0;
        return nullptr;
    }
    if (!audio->samples_cached) {
        // Materialize contiguous float copy from axiom tensor
        auto &t = audio->audio.samples;
        auto cpu_t = t.to_contiguous_cpu();
        size_t n = static_cast<size_t>(cpu_t.shape()[0]);
        audio->cached_samples.resize(n);
        auto *src = cpu_t.typed_data<float>();
        std::copy(src, src + n, audio->cached_samples.data());
        audio->samples_cached = true;
    }
    if (out_count)
        *out_count = audio->cached_samples.size();
    return audio->cached_samples.data();
}

extern "C" float parakeet_audio_file_duration(const char *path) {
    if (!path)
        return 0.0f;
    PARAKEET_TRY
    return parakeet::get_audio_duration(path);
    PARAKEET_CATCH(0.0f)
}

// ── Transcriber (110M TDT-CTC) ─────────────────────────────────────────────

extern "C" parakeet_transcriber_t
parakeet_transcriber_create(const char *weights, const char *vocab,
                            parakeet_config_t config) {
    if (!weights || !vocab) {
        g_last_error = "weights or vocab path is null";
        return nullptr;
    }
    if (config && config->type != parakeet_config_s::TDT_CTC_110M) {
        g_last_error = "config type mismatch: expected 110M TDT-CTC config";
        return nullptr;
    }
    PARAKEET_TRY
    auto t = new parakeet_transcriber_s;
    if (config) {
        auto &cfg = *static_cast<parakeet::TDTCTCConfig *>(config->ptr);
        t->impl = std::make_unique<parakeet::Transcriber>(weights, vocab, cfg);
    } else {
        t->impl = std::make_unique<parakeet::Transcriber>(weights, vocab);
    }
    return t;
    PARAKEET_CATCH(nullptr)
}

extern "C" void parakeet_transcriber_free(parakeet_transcriber_t t) {
    delete t;
}

extern "C" parakeet_error_t
parakeet_transcriber_to_gpu(parakeet_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_transcriber_to_half(parakeet_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_result_t
parakeet_transcriber_transcribe_file(parakeet_transcriber_t t, const char *path,
                                     parakeet_options_t opts) {
    if (!t || !path) {
        g_last_error = !t ? "null transcriber handle" : "path is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto r = new parakeet_result_s;
    r->result = t->impl->transcribe(std::string(path), resolve_opts(opts));
    return r;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_result_t
parakeet_transcriber_transcribe_pcm(parakeet_transcriber_t t,
                                    const float *samples, size_t n,
                                    parakeet_options_t opts) {
    if (!t || !samples) {
        g_last_error = !t ? "null transcriber handle" : "samples is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto tensor = axiom::Tensor::from_data(samples, axiom::Shape{n}, true);
    auto r = new parakeet_result_s;
    r->result = t->impl->transcribe(tensor, resolve_opts(opts));
    return r;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_result_t *
parakeet_transcriber_transcribe_batch(parakeet_transcriber_t t,
                                      const char **paths, size_t count,
                                      parakeet_options_t opts) {
    if (!t || !paths || count == 0) {
        g_last_error = !t ? "null transcriber handle" : "invalid paths/count";
        return nullptr;
    }
    PARAKEET_TRY
    std::vector<std::string> path_vec;
    path_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (!paths[i]) {
            g_last_error = "null path at index " + std::to_string(i);
            return nullptr;
        }
        path_vec.emplace_back(paths[i]);
    }
    auto results = t->impl->transcribe_batch(path_vec, resolve_opts(opts));
    auto *arr = new parakeet_result_t[count];
    for (size_t i = 0; i < results.size(); ++i) {
        arr[i] = new parakeet_result_s{std::move(results[i])};
    }
    return arr;
    PARAKEET_CATCH(nullptr)
}

// ── TDT Transcriber (600M) ─────────────────────────────────────────────────

extern "C" parakeet_tdt_transcriber_t
parakeet_tdt_transcriber_create(const char *weights, const char *vocab,
                                parakeet_config_t config) {
    if (!weights || !vocab) {
        g_last_error = "weights or vocab path is null";
        return nullptr;
    }
    if (config && config->type != parakeet_config_s::TDT_600M) {
        g_last_error = "config type mismatch: expected TDT 600M config";
        return nullptr;
    }
    PARAKEET_TRY
    auto t = new parakeet_tdt_transcriber_s;
    if (config) {
        auto &cfg = *static_cast<parakeet::TDTConfig *>(config->ptr);
        t->impl =
            std::make_unique<parakeet::TDTTranscriber>(weights, vocab, cfg);
    } else {
        t->impl = std::make_unique<parakeet::TDTTranscriber>(weights, vocab);
    }
    return t;
    PARAKEET_CATCH(nullptr)
}

extern "C" void parakeet_tdt_transcriber_free(parakeet_tdt_transcriber_t t) {
    delete t;
}

extern "C" parakeet_error_t
parakeet_tdt_transcriber_to_gpu(parakeet_tdt_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_tdt_transcriber_to_half(parakeet_tdt_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_result_t parakeet_tdt_transcriber_transcribe_file(
    parakeet_tdt_transcriber_t t, const char *path, parakeet_options_t opts) {
    if (!t || !path) {
        g_last_error = !t ? "null transcriber handle" : "path is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto r = new parakeet_result_s;
    r->result = t->impl->transcribe(std::string(path), resolve_opts(opts));
    return r;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_result_t
parakeet_tdt_transcriber_transcribe_pcm(parakeet_tdt_transcriber_t t,
                                        const float *samples, size_t n,
                                        parakeet_options_t opts) {
    if (!t || !samples) {
        g_last_error = !t ? "null transcriber handle" : "samples is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto tensor = axiom::Tensor::from_data(samples, axiom::Shape{n}, true);
    auto r = new parakeet_result_s;
    r->result = t->impl->transcribe(tensor, resolve_opts(opts));
    return r;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_result_t *
parakeet_tdt_transcriber_transcribe_batch(parakeet_tdt_transcriber_t t,
                                          const char **paths, size_t count,
                                          parakeet_options_t opts) {
    if (!t || !paths || count == 0) {
        g_last_error = !t ? "null transcriber handle" : "invalid paths/count";
        return nullptr;
    }
    PARAKEET_TRY
    std::vector<std::string> path_vec;
    path_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (!paths[i]) {
            g_last_error = "null path at index " + std::to_string(i);
            return nullptr;
        }
        path_vec.emplace_back(paths[i]);
    }
    auto results = t->impl->transcribe_batch(path_vec, resolve_opts(opts));
    auto *arr = new parakeet_result_t[count];
    for (size_t i = 0; i < results.size(); ++i) {
        arr[i] = new parakeet_result_s{std::move(results[i])};
    }
    return arr;
    PARAKEET_CATCH(nullptr)
}

// ── Streaming Transcriber (EOU 120M) ────────────────────────────────────────

extern "C" parakeet_streaming_transcriber_t
parakeet_streaming_transcriber_create(const char *weights, const char *vocab,
                                      parakeet_config_t config) {
    if (!weights || !vocab) {
        g_last_error = "weights or vocab path is null";
        return nullptr;
    }
    if (config && config->type != parakeet_config_s::EOU_120M) {
        g_last_error = "config type mismatch: expected EOU 120M config";
        return nullptr;
    }
    PARAKEET_TRY
    auto t = new parakeet_streaming_transcriber_s;
    if (config) {
        auto &cfg = *static_cast<parakeet::EOUConfig *>(config->ptr);
        t->impl = std::make_unique<parakeet::StreamingTranscriber>(weights,
                                                                   vocab, cfg);
    } else {
        t->impl =
            std::make_unique<parakeet::StreamingTranscriber>(weights, vocab);
    }
    return t;
    PARAKEET_CATCH(nullptr)
}

extern "C" void
parakeet_streaming_transcriber_free(parakeet_streaming_transcriber_t t) {
    delete t;
}

extern "C" parakeet_error_t
parakeet_streaming_transcriber_to_gpu(parakeet_streaming_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_streaming_transcriber_to_half(parakeet_streaming_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_streaming_transcriber_feed_f32(parakeet_streaming_transcriber_t t,
                                        const float *samples, size_t n,
                                        const char **out_text) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!samples) {
        g_last_error = "samples is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->last_chunk_text = t->impl->transcribe_chunk(samples, n);
    if (out_text)
        *out_text = t->last_chunk_text.c_str();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_streaming_transcriber_feed_i16(parakeet_streaming_transcriber_t t,
                                        const int16_t *samples, size_t n,
                                        const char **out_text) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!samples) {
        g_last_error = "samples is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->last_chunk_text = t->impl->transcribe_chunk(samples, n);
    if (out_text)
        *out_text = t->last_chunk_text.c_str();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_streaming_transcriber_reset(parakeet_streaming_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->reset();
    t->last_chunk_text.clear();
    t->full_text_cache.clear();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" const char *
parakeet_streaming_transcriber_get_text(parakeet_streaming_transcriber_t t) {
    if (!t)
        return nullptr;
    PARAKEET_TRY
    t->full_text_cache = t->impl->get_text();
    return t->full_text_cache.c_str();
    PARAKEET_CATCH(nullptr)
}

// ── Nemotron Transcriber (600M) ─────────────────────────────────────────────

extern "C" parakeet_nemotron_transcriber_t
parakeet_nemotron_transcriber_create(const char *weights, const char *vocab,
                                     parakeet_config_t config) {
    if (!weights || !vocab) {
        g_last_error = "weights or vocab path is null";
        return nullptr;
    }
    if (config && config->type != parakeet_config_s::NEMOTRON_600M) {
        g_last_error = "config type mismatch: expected Nemotron 600M config";
        return nullptr;
    }
    PARAKEET_TRY
    auto t = new parakeet_nemotron_transcriber_s;
    if (config) {
        auto &cfg = *static_cast<parakeet::NemotronConfig *>(config->ptr);
        t->impl = std::make_unique<parakeet::NemotronTranscriber>(weights,
                                                                  vocab, cfg);
    } else {
        t->impl =
            std::make_unique<parakeet::NemotronTranscriber>(weights, vocab);
    }
    return t;
    PARAKEET_CATCH(nullptr)
}

extern "C" void
parakeet_nemotron_transcriber_free(parakeet_nemotron_transcriber_t t) {
    delete t;
}

extern "C" parakeet_error_t
parakeet_nemotron_transcriber_to_gpu(parakeet_nemotron_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_nemotron_transcriber_to_half(parakeet_nemotron_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_nemotron_transcriber_feed_f32(parakeet_nemotron_transcriber_t t,
                                       const float *samples, size_t n,
                                       const char **out_text) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!samples) {
        g_last_error = "samples is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->last_chunk_text = t->impl->transcribe_chunk(samples, n);
    if (out_text)
        *out_text = t->last_chunk_text.c_str();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_nemotron_transcriber_feed_i16(parakeet_nemotron_transcriber_t t,
                                       const int16_t *samples, size_t n,
                                       const char **out_text) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!samples) {
        g_last_error = "samples is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->last_chunk_text = t->impl->transcribe_chunk(samples, n);
    if (out_text)
        *out_text = t->last_chunk_text.c_str();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_nemotron_transcriber_reset(parakeet_nemotron_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->reset();
    t->last_chunk_text.clear();
    t->full_text_cache.clear();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" const char *
parakeet_nemotron_transcriber_get_text(parakeet_nemotron_transcriber_t t) {
    if (!t)
        return nullptr;
    PARAKEET_TRY
    t->full_text_cache = t->impl->get_text();
    return t->full_text_cache.c_str();
    PARAKEET_CATCH(nullptr)
}

// ── Diarized Transcriber ────────────────────────────────────────────────────

extern "C" parakeet_diarized_transcriber_t parakeet_diarized_transcriber_create(
    const char *asr_weights, const char *sortformer_weights, const char *vocab,
    parakeet_config_t config, parakeet_config_t sf_config) {
    if (!asr_weights || !sortformer_weights || !vocab) {
        g_last_error = "asr_weights, sortformer_weights, or vocab path is null";
        return nullptr;
    }
    if (config && config->type != parakeet_config_s::TDT_CTC_110M) {
        g_last_error =
            "config type mismatch: expected 110M TDT-CTC config for ASR";
        return nullptr;
    }
    if (sf_config && sf_config->type != parakeet_config_s::SORTFORMER_117M) {
        g_last_error =
            "sf_config type mismatch: expected Sortformer 117M config";
        return nullptr;
    }
    PARAKEET_TRY
    auto t = new parakeet_diarized_transcriber_s;
    auto asr_cfg = config ? *static_cast<parakeet::TDTCTCConfig *>(config->ptr)
                          : parakeet::make_110m_config();
    auto sf_cfg =
        sf_config ? *static_cast<parakeet::SortformerConfig *>(sf_config->ptr)
                  : parakeet::make_sortformer_117m_config();
    t->impl = std::make_unique<parakeet::DiarizedTranscriber>(
        asr_weights, sortformer_weights, vocab, asr_cfg, sf_cfg);
    return t;
    PARAKEET_CATCH(nullptr)
}

extern "C" void
parakeet_diarized_transcriber_free(parakeet_diarized_transcriber_t t) {
    delete t;
}

extern "C" parakeet_error_t
parakeet_diarized_transcriber_to_gpu(parakeet_diarized_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_diarized_transcriber_to_half(parakeet_diarized_transcriber_t t) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    t->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_diarized_result_t
parakeet_diarized_transcriber_transcribe_file(parakeet_diarized_transcriber_t t,
                                              const char *path,
                                              parakeet_decoder_t decoder) {
    if (!t || !path) {
        g_last_error = !t ? "null transcriber handle" : "path is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto r = new parakeet_diarized_result_s;
    r->result = t->impl->transcribe(std::string(path), to_cpp_decoder(decoder));
    return r;
    PARAKEET_CATCH(nullptr)
}

extern "C" parakeet_diarized_result_t
parakeet_diarized_transcriber_transcribe_pcm(parakeet_diarized_transcriber_t t,
                                             const float *samples, size_t n,
                                             parakeet_decoder_t decoder) {
    if (!t || !samples) {
        g_last_error = !t ? "null transcriber handle" : "samples is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto tensor = axiom::Tensor::from_data(samples, axiom::Shape{n}, true);
    auto r = new parakeet_diarized_result_s;
    r->result = t->impl->transcribe(tensor, to_cpp_decoder(decoder));
    return r;
    PARAKEET_CATCH(nullptr)
}

// ── VAD ─────────────────────────────────────────────────────────────────────

struct parakeet_vad_s {
    std::unique_ptr<parakeet::audio::SileroVAD> impl;
};

extern "C" parakeet_vad_config_t parakeet_vad_config_default(void) {
    parakeet_vad_config_t cfg;
    cfg.threshold = 0.5f;
    cfg.neg_threshold = 0.35f;
    cfg.min_speech_duration_ms = 250;
    cfg.min_silence_duration_ms = 100;
    cfg.speech_pad_ms = 30;
    cfg.max_speech_duration_s = std::numeric_limits<float>::infinity();
    return cfg;
}

extern "C" parakeet_vad_t parakeet_vad_create(const char *weights_path) {
    if (!weights_path) {
        g_last_error = "weights_path is null";
        return nullptr;
    }
    PARAKEET_TRY
    auto v = new parakeet_vad_s;
    v->impl = std::make_unique<parakeet::audio::SileroVAD>(weights_path);
    return v;
    PARAKEET_CATCH(nullptr)
}

extern "C" void parakeet_vad_destroy(parakeet_vad_t vad) { delete vad; }

extern "C" parakeet_error_t parakeet_vad_to_gpu(parakeet_vad_t vad) {
    NULL_CHECK(vad, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    vad->impl->to_gpu();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t parakeet_vad_to_half(parakeet_vad_t vad) {
    NULL_CHECK(vad, PARAKEET_ERROR_NULL_HANDLE);
    PARAKEET_TRY
    vad->impl->to_half();
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_vad_detect(parakeet_vad_t vad, parakeet_audio_t audio,
                    parakeet_vad_config_t config,
                    parakeet_speech_segment_t **segments_out, int *count_out) {
    NULL_CHECK(vad, PARAKEET_ERROR_NULL_HANDLE);
    NULL_CHECK(audio, PARAKEET_ERROR_NULL_HANDLE);
    if (!segments_out || !count_out) {
        g_last_error = "output pointers are null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    parakeet::audio::VADConfig cpp_config;
    cpp_config.threshold = config.threshold;
    cpp_config.neg_threshold = config.neg_threshold;
    cpp_config.min_speech_duration_ms = config.min_speech_duration_ms;
    cpp_config.min_silence_duration_ms = config.min_silence_duration_ms;
    cpp_config.speech_pad_ms = config.speech_pad_ms;
    cpp_config.max_speech_duration_s = config.max_speech_duration_s;

    auto segments = vad->impl->detect(audio->audio.samples,
                                      audio->audio.sample_rate, cpp_config);

    *count_out = static_cast<int>(segments.size());
    if (segments.empty()) {
        *segments_out = nullptr;
    } else {
        auto *arr = new parakeet_speech_segment_t[segments.size()];
        for (size_t i = 0; i < segments.size(); ++i) {
            arr[i].start_sample = segments[i].start_sample;
            arr[i].end_sample = segments[i].end_sample;
        }
        *segments_out = arr;
    }
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" void
parakeet_speech_segments_free(parakeet_speech_segment_t *segments) {
    delete[] segments;
}

extern "C" parakeet_error_t
parakeet_transcriber_enable_vad(parakeet_transcriber_t t,
                                const char *vad_weights_path) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!vad_weights_path) {
        g_last_error = "vad_weights_path is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->impl->enable_vad(vad_weights_path);
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_tdt_transcriber_enable_vad(parakeet_tdt_transcriber_t t,
                                    const char *vad_weights_path) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!vad_weights_path) {
        g_last_error = "vad_weights_path is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->impl->enable_vad(vad_weights_path);
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

extern "C" parakeet_error_t
parakeet_diarized_transcriber_enable_vad(parakeet_diarized_transcriber_t t,
                                         const char *vad_weights_path) {
    NULL_CHECK(t, PARAKEET_ERROR_NULL_HANDLE);
    if (!vad_weights_path) {
        g_last_error = "vad_weights_path is null";
        return PARAKEET_ERROR_INVALID_ARGUMENT;
    }
    PARAKEET_TRY
    t->impl->enable_vad(vad_weights_path);
    return PARAKEET_OK;
    PARAKEET_CATCH(PARAKEET_ERROR_RUNTIME)
}

// ── Result Accessors ────────────────────────────────────────────────────────

extern "C" void parakeet_result_batch_free(parakeet_result_t *results,
                                           size_t count) {
    if (!results)
        return;
    for (size_t i = 0; i < count; ++i) {
        delete results[i];
    }
    delete[] results;
}

extern "C" void parakeet_result_free(parakeet_result_t r) { delete r; }

extern "C" const char *parakeet_result_text(parakeet_result_t r) {
    return r ? r->result.text.c_str() : nullptr;
}

extern "C" size_t parakeet_result_token_count(parakeet_result_t r) {
    return r ? r->result.token_ids.size() : 0;
}

extern "C" const int *parakeet_result_token_ids(parakeet_result_t r) {
    return (r && !r->result.token_ids.empty()) ? r->result.token_ids.data()
                                               : nullptr;
}

extern "C" size_t parakeet_result_timestamped_token_count(parakeet_result_t r) {
    return r ? r->result.timestamped_tokens.size() : 0;
}

extern "C" parakeet_timestamped_token_t
parakeet_result_timestamped_token_at(parakeet_result_t r, size_t index) {
    parakeet_timestamped_token_t out = {0, 0, 0, 0.0f};
    if (!r || index >= r->result.timestamped_tokens.size())
        return out;
    auto &t = r->result.timestamped_tokens[index];
    out.token_id = t.token_id;
    out.start_frame = t.start_frame;
    out.end_frame = t.end_frame;
    out.confidence = t.confidence;
    return out;
}

extern "C" size_t parakeet_result_word_count(parakeet_result_t r) {
    return r ? r->result.word_timestamps.size() : 0;
}

extern "C" parakeet_word_timestamp_t
parakeet_result_word_at(parakeet_result_t r, size_t index) {
    parakeet_word_timestamp_t out = {nullptr, 0.0f, 0.0f, 0.0f};
    if (!r || index >= r->result.word_timestamps.size())
        return out;
    auto &w = r->result.word_timestamps[index];
    out.word = w.word.c_str();
    out.start = w.start;
    out.end = w.end;
    out.confidence = w.confidence;
    return out;
}

// ── Diarized Result Accessors ───────────────────────────────────────────────

extern "C" void parakeet_diarized_result_free(parakeet_diarized_result_t r) {
    delete r;
}

extern "C" const char *
parakeet_diarized_result_text(parakeet_diarized_result_t r) {
    return r ? r->result.text.c_str() : nullptr;
}

extern "C" size_t
parakeet_diarized_result_word_count(parakeet_diarized_result_t r) {
    return r ? r->result.words.size() : 0;
}

extern "C" parakeet_diarized_word_t
parakeet_diarized_result_word_at(parakeet_diarized_result_t r, size_t index) {
    parakeet_diarized_word_t out = {nullptr, 0.0f, 0.0f, -1, 0.0f};
    if (!r || index >= r->result.words.size())
        return out;
    auto &w = r->result.words[index];
    out.word = w.word.c_str();
    out.start = w.start;
    out.end = w.end;
    out.speaker_id = w.speaker_id;
    out.confidence = w.confidence;
    return out;
}

extern "C" size_t
parakeet_diarized_result_segment_count(parakeet_diarized_result_t r) {
    return r ? r->result.segments.size() : 0;
}

extern "C" parakeet_diarization_segment_t
parakeet_diarized_result_segment_at(parakeet_diarized_result_t r,
                                    size_t index) {
    parakeet_diarization_segment_t out = {-1, 0.0f, 0.0f};
    if (!r || index >= r->result.segments.size())
        return out;
    auto &s = r->result.segments[index];
    out.speaker_id = s.speaker_id;
    out.start = s.start;
    out.end = s.end;
    return out;
}
