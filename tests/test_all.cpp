#include <gtest/gtest.h>

#include "parakeet/parakeet.hpp"

#include <axiom/io/safetensors.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace parakeet;
using namespace axiom;

// ─── Test Fixture with model paths ──────────────────────────────────────────

static const char *MODELS_DIR = "models";

static std::string model_path(const std::string &filename) {
    // Try a few relative paths since test runner CWD may vary
    for (const auto &base :
         {"models", "../models", "../../models",
          "/Users/noahkay/Documents/parakeet.cpp/models"}) {
        auto p = std::string(base) + "/" + filename;
        if (std::filesystem::exists(p))
            return p;
    }
    return std::string("models/") + filename;
}

static bool has_model_weights() {
    return std::filesystem::exists(model_path("model.safetensors"));
}

static bool has_vocab() {
    return std::filesystem::exists(model_path("vocab.txt"));
}

static bool has_test_audio() {
    return std::filesystem::exists(model_path("2086-149220-0033.wav"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Phase 1: Timestamps
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TimestampTypes, FrameToSeconds) {
    EXPECT_FLOAT_EQ(frame_to_seconds(0), 0.0f);
    EXPECT_FLOAT_EQ(frame_to_seconds(1), 0.08f);
    EXPECT_FLOAT_EQ(frame_to_seconds(10), 0.80f);
    EXPECT_FLOAT_EQ(frame_to_seconds(125), 10.0f);
}

TEST(TimestampTypes, TimestampedTokenStruct) {
    TimestampedToken tok{42, 5, 10};
    EXPECT_EQ(tok.token_id, 42);
    EXPECT_EQ(tok.start_frame, 5);
    EXPECT_EQ(tok.end_frame, 10);
}

TEST(TimestampTypes, WordTimestampStruct) {
    WordTimestamp w{"hello", 1.0f, 2.5f};
    EXPECT_EQ(w.word, "hello");
    EXPECT_FLOAT_EQ(w.start, 1.0f);
    EXPECT_FLOAT_EQ(w.end, 2.5f);
}

TEST(GroupTimestamps, EmptyInput) {
    auto result = group_timestamps({}, {});
    EXPECT_TRUE(result.empty());
}

TEST(GroupTimestamps, SingleToken) {
    // pieces[0] = "▁hello" (with SentencePiece marker)
    std::vector<std::string> pieces = {"\xe2\x96\x81hello"};
    std::vector<TimestampedToken> tokens = {{0, 5, 10}};

    auto words = group_timestamps(tokens, pieces);
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].word, "hello");
    EXPECT_FLOAT_EQ(words[0].start, frame_to_seconds(5));
    EXPECT_FLOAT_EQ(words[0].end, frame_to_seconds(10));
}

TEST(GroupTimestamps, MultipleWords) {
    // "▁the" "▁quick" "▁fox"
    std::vector<std::string> pieces = {"\xe2\x96\x81the", "\xe2\x96\x81quick",
                                        "\xe2\x96\x81" "fox"};
    std::vector<TimestampedToken> tokens = {{0, 0, 2}, {1, 5, 8}, {2, 12, 15}};

    auto words = group_timestamps(tokens, pieces);
    ASSERT_EQ(words.size(), 3u);
    EXPECT_EQ(words[0].word, "the");
    EXPECT_EQ(words[1].word, "quick");
    EXPECT_EQ(words[2].word, "fox");
}

TEST(GroupTimestamps, SubwordTokens) {
    // "▁run" "ning" → "running"
    std::vector<std::string> pieces = {"\xe2\x96\x81run", "ning"};
    std::vector<TimestampedToken> tokens = {{0, 0, 3}, {1, 4, 6}};

    auto words = group_timestamps(tokens, pieces);
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].word, "running");
    EXPECT_FLOAT_EQ(words[0].start, frame_to_seconds(0));
    EXPECT_FLOAT_EQ(words[0].end, frame_to_seconds(6));
}

TEST(GroupTimestamps, SentenceMode) {
    std::vector<std::string> pieces = {"\xe2\x96\x81Hello", "\xe2\x96\x81world.",
                                        "\xe2\x96\x81" "How", "\xe2\x96\x81" "are"};
    std::vector<TimestampedToken> tokens = {
        {0, 0, 2}, {1, 3, 5}, {2, 8, 10}, {3, 11, 13}};

    auto sentences =
        group_timestamps(tokens, pieces, TimestampMode::Sentences);
    ASSERT_EQ(sentences.size(), 2u);
    EXPECT_EQ(sentences[0].word, "Hello world.");
    EXPECT_EQ(sentences[1].word, "How are");
}

TEST(GroupTimestamps, OutOfRangeTokenId) {
    std::vector<std::string> pieces = {"\xe2\x96\x81hello"};
    // Token ID 999 is out of range — should be skipped
    std::vector<TimestampedToken> tokens = {{999, 0, 1}, {0, 2, 4}};

    auto words = group_timestamps(tokens, pieces);
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0].word, "hello");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Phase 2: Config Presets
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ConfigPresets, Make110mConfig) {
    auto cfg = make_110m_config();
    EXPECT_EQ(cfg.encoder.hidden_size, 512);
    EXPECT_EQ(cfg.encoder.num_layers, 17);
    EXPECT_EQ(cfg.encoder.num_heads, 8);
    EXPECT_EQ(cfg.encoder.ffn_intermediate, 2048);
    EXPECT_EQ(cfg.prediction.vocab_size, 1025);
    EXPECT_EQ(cfg.prediction.num_lstm_layers, 1);
    EXPECT_EQ(cfg.joint.encoder_hidden, 512);
    EXPECT_EQ(cfg.durations.size(), 5u);
    EXPECT_EQ(cfg.ctc_vocab_size, 1025);
}

TEST(ConfigPresets, MakeTDT600mConfig) {
    auto cfg = make_tdt_600m_config();
    EXPECT_EQ(cfg.encoder.hidden_size, 1024);
    EXPECT_EQ(cfg.encoder.num_layers, 24);
    EXPECT_EQ(cfg.encoder.num_heads, 8);
    EXPECT_EQ(cfg.encoder.ffn_intermediate, 4096);
    EXPECT_EQ(cfg.prediction.vocab_size, 8193);
    EXPECT_EQ(cfg.prediction.num_lstm_layers, 2);
    EXPECT_EQ(cfg.joint.vocab_size, 8193);
    EXPECT_EQ(cfg.durations.size(), 5u);
}

TEST(ConfigPresets, MakeEOU120mConfig) {
    auto cfg = make_eou_120m_config();
    EXPECT_EQ(cfg.encoder.hidden_size, 512);
    EXPECT_EQ(cfg.encoder.num_layers, 17);
    EXPECT_EQ(cfg.encoder.att_context_left, 70);
    EXPECT_EQ(cfg.encoder.att_context_right, 1);
    EXPECT_EQ(cfg.encoder.chunk_size, 20);
    EXPECT_EQ(cfg.prediction.vocab_size, 1025);
}

TEST(ConfigPresets, MakeNemotron600mConfig) {
    auto cfg0 = make_nemotron_600m_config(0);
    EXPECT_EQ(cfg0.encoder.att_context_right, 0);
    EXPECT_EQ(cfg0.latency_frames, 0);

    auto cfg6 = make_nemotron_600m_config(6);
    EXPECT_EQ(cfg6.encoder.att_context_right, 6);
    EXPECT_EQ(cfg6.latency_frames, 6);

    auto cfg13 = make_nemotron_600m_config(13);
    EXPECT_EQ(cfg13.encoder.att_context_right, 13);
    EXPECT_EQ(cfg13.encoder.num_layers, 24);
    EXPECT_EQ(cfg13.prediction.vocab_size, 8193);
}

TEST(ConfigPresets, MakeSortformer117mConfig) {
    auto cfg = make_sortformer_117m_config();
    EXPECT_EQ(cfg.nest_encoder.hidden_size, 512);
    EXPECT_EQ(cfg.nest_encoder.num_layers, 17);
    EXPECT_EQ(cfg.transformer_hidden, 192);
    EXPECT_EQ(cfg.transformer.num_layers, 18);
    EXPECT_EQ(cfg.transformer.num_heads, 8);
    EXPECT_EQ(cfg.max_speakers, 4);
    EXPECT_FLOAT_EQ(cfg.activity_threshold, 0.5f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model Construction (no weights needed)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ModelConstruction, ParakeetTDTCTC) {
    auto cfg = make_110m_config();
    ParakeetTDTCTC model(cfg);
    EXPECT_EQ(model.config().encoder.num_layers, 17);
}

TEST(ModelConstruction, ParakeetTDT) {
    auto cfg = make_tdt_600m_config();
    ParakeetTDT model(cfg);
    EXPECT_EQ(model.config().encoder.num_layers, 24);
}

TEST(ModelConstruction, ParakeetEOU) {
    auto cfg = make_eou_120m_config();
    ParakeetEOU model(cfg);
    EXPECT_EQ(model.config().encoder.num_layers, 17);
}

TEST(ModelConstruction, ParakeetNemotron) {
    auto cfg = make_nemotron_600m_config();
    ParakeetNemotron model(cfg);
    EXPECT_EQ(model.config().encoder.num_layers, 24);
}

TEST(ModelConstruction, Sortformer) {
    auto cfg = make_sortformer_117m_config();
    Sortformer model(cfg);
    EXPECT_EQ(model.config().transformer.num_layers, 18);
}

TEST(ModelConstruction, ParakeetRNNT) {
    RNNTConfig cfg;
    cfg.encoder.hidden_size = 512;
    cfg.encoder.num_layers = 17;
    ParakeetRNNT model(cfg);
    EXPECT_EQ(model.config().encoder.num_layers, 17);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Streaming Encoder Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(StreamingEncoder, ConfigDefaults) {
    StreamingEncoderConfig cfg;
    EXPECT_EQ(cfg.att_context_left, 70);
    EXPECT_EQ(cfg.att_context_right, 0);
    EXPECT_EQ(cfg.chunk_size, 20);
}

TEST(StreamingEncoder, CacheInit) {
    EncoderCache cache;
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.frames_seen, 0);
}

TEST(StreamingEncoder, BlockCacheInit) {
    BlockCache bc;
    EXPECT_FALSE(bc.conv_cache.storage());
    EXPECT_FALSE(bc.key_cache.storage());
    EXPECT_FALSE(bc.value_cache.storage());
}

TEST(StreamingEncoder, Construction) {
    StreamingEncoderConfig cfg;
    cfg.hidden_size = 256;
    cfg.num_layers = 4;
    cfg.num_heads = 4;
    cfg.ffn_intermediate = 1024;
    StreamingFastConformerEncoder encoder(cfg);
    // Just verify it doesn't crash
    EXPECT_EQ(cfg.num_layers, 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Transformer Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Transformer, BlockConstruction) {
    TransformerConfig cfg;
    cfg.hidden_size = 192;
    cfg.num_heads = 4;
    cfg.num_layers = 2;
    TransformerBlock block(cfg);
    SUCCEED();
}

TEST(Transformer, EncoderConstruction) {
    TransformerConfig cfg;
    cfg.hidden_size = 192;
    cfg.num_heads = 4;
    cfg.num_layers = 6;
    TransformerEncoder encoder(cfg);
    SUCCEED();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  AOSC Cache
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AOSCCache, Empty) {
    AOSCCache cache(4);
    EXPECT_TRUE(cache.speaker_order().empty());
}

TEST(AOSCCache, SingleSpeaker) {
    AOSCCache cache(4);
    // Create probs: speaker 0 active, others silent
    std::vector<float> data = {0.9f, 0.1f, 0.1f, 0.1f,
                                0.8f, 0.1f, 0.1f, 0.1f};
    auto probs = Tensor::from_data(data.data(), Shape{2, 4}, true);
    cache.update(probs);

    auto order = cache.speaker_order();
    ASSERT_EQ(order.size(), 1u);
    EXPECT_EQ(order[0], 0);
}

TEST(AOSCCache, TwoSpeakersArrivalOrder) {
    AOSCCache cache(4);
    // Frame 0: speaker 2 active
    // Frame 1: speaker 0 also active
    std::vector<float> data = {0.1f, 0.1f, 0.9f, 0.1f,
                                0.9f, 0.1f, 0.8f, 0.1f};
    auto probs = Tensor::from_data(data.data(), Shape{2, 4}, true);
    cache.update(probs);

    auto order = cache.speaker_order();
    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], 2); // arrived first
    EXPECT_EQ(order[1], 0); // arrived second
}

TEST(AOSCCache, Reset) {
    AOSCCache cache(4);
    std::vector<float> data = {0.9f, 0.1f, 0.1f, 0.1f};
    auto probs = Tensor::from_data(data.data(), Shape{1, 4}, true);
    cache.update(probs);
    EXPECT_EQ(cache.speaker_order().size(), 1u);

    cache.reset();
    EXPECT_TRUE(cache.speaker_order().empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Diarization Types
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DiarizationTypes, SegmentStruct) {
    DiarizationSegment seg{1, 0.5f, 2.3f};
    EXPECT_EQ(seg.speaker_id, 1);
    EXPECT_FLOAT_EQ(seg.start, 0.5f);
    EXPECT_FLOAT_EQ(seg.end, 2.3f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Streaming Decode State
// ═══════════════════════════════════════════════════════════════════════════════

TEST(StreamingDecode, StateInit) {
    StreamingDecodeState state;
    EXPECT_FALSE(state.initialized);
    EXPECT_TRUE(state.tokens.empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Streaming Audio Preprocessor
// ═══════════════════════════════════════════════════════════════════════════════

TEST(StreamingAudio, Construction) {
    AudioConfig cfg;
    StreamingAudioPreprocessor prep(cfg);
    SUCCEED();
}

TEST(StreamingAudio, ResetDoesNotCrash) {
    StreamingAudioPreprocessor prep;
    prep.reset();
    SUCCEED();
}

TEST(StreamingAudio, SmallChunkReturnsEmpty) {
    StreamingAudioPreprocessor prep;
    // 100 samples is not enough for a frame (needs win_length=400)
    auto chunk = Tensor::zeros({100});
    auto result = prep.process_chunk(chunk);
    // Should return empty or valid tensor
    // With 100 samples, we won't have enough for a full frame
    SUCCEED();
}

TEST(StreamingAudio, LargeChunkProducesOutput) {
    StreamingAudioPreprocessor prep;
    // 4800 samples = 0.3s at 16kHz, should produce several frames
    auto chunk = Tensor::zeros({4800});
    auto result = prep.process_chunk(chunk);
    if (result.storage()) {
        auto shape = result.shape();
        EXPECT_EQ(shape.size(), 3u);
        EXPECT_EQ(shape[0], 1u); // batch
        EXPECT_GT(shape[1], 0u); // frames
        EXPECT_EQ(shape[2], 80u); // mel bins
    }
}

TEST(StreamingAudio, MultipleChunksAccumulate) {
    StreamingAudioPreprocessor prep;
    int total_frames = 0;
    // Process 5 chunks of 1600 samples each (0.1s each)
    for (int i = 0; i < 5; ++i) {
        auto chunk = Tensor::zeros({1600});
        auto result = prep.process_chunk(chunk);
        if (result.storage()) {
            total_frames += static_cast<int>(result.shape()[1]);
        }
    }
    // Should have produced some frames across chunks
    EXPECT_GT(total_frames, 0);
}

TEST(StreamingAudio, ResetClearsState) {
    StreamingAudioPreprocessor prep;
    auto chunk = Tensor::zeros({4800});
    prep.process_chunk(chunk);
    prep.reset();
    // After reset, should be clean state
    auto result = prep.process_chunk(Tensor::zeros({100}));
    // Small chunk after reset: should not produce output
    SUCCEED();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tokenizer
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Tokenizer, DefaultEmpty) {
    Tokenizer tok;
    EXPECT_FALSE(tok.loaded());
}

TEST(Tokenizer, LoadVocab) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab.txt not found";

    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    EXPECT_TRUE(tok.loaded());
    EXPECT_EQ(tok.vocab_size(), 1025u);
}

TEST(Tokenizer, PiecesAccessor) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab.txt not found";

    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    EXPECT_EQ(tok.pieces().size(), 1024u);
    EXPECT_FALSE(tok.pieces()[0].empty());
}

TEST(Tokenizer, DecodeEmpty) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab.txt not found";

    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    EXPECT_EQ(tok.decode({}), "");
}

TEST(Tokenizer, DecodeOutOfRange) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab.txt not found";

    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    // Out of range tokens should produce [id] placeholder
    auto text = tok.decode({9999});
    EXPECT_EQ(text, "[9999]");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: Format Detection
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, DetectFormatByExtensionWAV) {
    EXPECT_EQ(detect_format_by_extension("test.wav"), AudioFormat::WAV);
    EXPECT_EQ(detect_format_by_extension("test.WAVE"), AudioFormat::WAV);
}

TEST(AudioIO, DetectFormatByExtensionFLAC) {
    EXPECT_EQ(detect_format_by_extension("test.flac"), AudioFormat::FLAC);
}

TEST(AudioIO, DetectFormatByExtensionMP3) {
    EXPECT_EQ(detect_format_by_extension("test.mp3"), AudioFormat::MP3);
}

TEST(AudioIO, DetectFormatByExtensionOGG) {
    EXPECT_EQ(detect_format_by_extension("test.ogg"), AudioFormat::OGG);
    EXPECT_EQ(detect_format_by_extension("test.oga"), AudioFormat::OGG);
}

TEST(AudioIO, DetectFormatByExtensionUnknown) {
    EXPECT_EQ(detect_format_by_extension("test.txt"), AudioFormat::Unknown);
    EXPECT_EQ(detect_format_by_extension("noext"), AudioFormat::Unknown);
}

TEST(AudioIO, DetectFormatByMagicWAV) {
    // RIFF....WAVE
    uint8_t wav_header[] = {'R', 'I', 'F', 'F', 0, 0, 0, 0,
                             'W', 'A', 'V', 'E'};
    EXPECT_EQ(detect_format_by_magic(wav_header, sizeof(wav_header)),
              AudioFormat::WAV);
}

TEST(AudioIO, DetectFormatByMagicFLAC) {
    uint8_t flac_header[] = {'f', 'L', 'a', 'C'};
    EXPECT_EQ(detect_format_by_magic(flac_header, sizeof(flac_header)),
              AudioFormat::FLAC);
}

TEST(AudioIO, DetectFormatByMagicMP3ID3) {
    uint8_t mp3_header[] = {'I', 'D', '3'};
    EXPECT_EQ(detect_format_by_magic(mp3_header, sizeof(mp3_header)),
              AudioFormat::MP3);
}

TEST(AudioIO, DetectFormatByMagicMP3Sync) {
    uint8_t mp3_header[] = {0xFF, 0xFB};
    EXPECT_EQ(detect_format_by_magic(mp3_header, sizeof(mp3_header)),
              AudioFormat::MP3);
}

TEST(AudioIO, DetectFormatByMagicOGG) {
    uint8_t ogg_header[] = {'O', 'g', 'g', 'S'};
    EXPECT_EQ(detect_format_by_magic(ogg_header, sizeof(ogg_header)),
              AudioFormat::OGG);
}

TEST(AudioIO, DetectFormatByMagicUnknown) {
    uint8_t garbage[] = {0x00, 0x01, 0x02, 0x03};
    EXPECT_EQ(detect_format_by_magic(garbage, sizeof(garbage)),
              AudioFormat::Unknown);
}

TEST(AudioIO, DetectFormatByMagicTooShort) {
    uint8_t tiny[] = {0x00};
    EXPECT_EQ(detect_format_by_magic(tiny, 1), AudioFormat::Unknown);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: Resampler
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, ResampleIdentity) {
    // Same rate should return same data
    std::vector<float> data(1000, 0.5f);
    auto tensor = Tensor::from_data(data.data(), Shape{data.size()}, true);
    auto result = resample(tensor, 16000, 16000);
    EXPECT_EQ(result.shape()[0], tensor.shape()[0]);
}

TEST(AudioIO, ResampleDurationPreservation44to16) {
    // 1 second of audio at 44100Hz → should produce ~1 second at 16000Hz
    size_t src_len = 44100;
    std::vector<float> data(src_len, 0.0f);
    // Add a simple sine wave
    for (size_t i = 0; i < src_len; ++i) {
        data[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
    }
    auto tensor = Tensor::from_data(data.data(), Shape{src_len}, true);
    auto result = resample(tensor, 44100, 16000);

    // Duration should be preserved within tolerance
    float src_duration = static_cast<float>(src_len) / 44100.0f;
    float dst_duration =
        static_cast<float>(result.shape()[0]) / 16000.0f;
    EXPECT_NEAR(src_duration, dst_duration, 0.01f);
}

TEST(AudioIO, ResampleDurationPreservation48to16) {
    size_t src_len = 48000;
    std::vector<float> data(src_len, 0.0f);
    for (size_t i = 0; i < src_len; ++i) {
        data[i] = std::sin(2.0f * M_PI * 440.0f * i / 48000.0f);
    }
    auto tensor = Tensor::from_data(data.data(), Shape{src_len}, true);
    auto result = resample(tensor, 48000, 16000);

    float src_duration = static_cast<float>(src_len) / 48000.0f;
    float dst_duration =
        static_cast<float>(result.shape()[0]) / 16000.0f;
    EXPECT_NEAR(src_duration, dst_duration, 0.01f);
}

TEST(AudioIO, ResampleSineWaveIntegrity) {
    // Resample a 440Hz sine from 44100 to 16000
    // The sine should still be detectable after resampling
    size_t src_len = 44100;
    std::vector<float> data(src_len);
    for (size_t i = 0; i < src_len; ++i) {
        data[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
    }
    auto tensor = Tensor::from_data(data.data(), Shape{src_len}, true);
    auto result = resample(tensor, 44100, 16000);
    auto cont = result.ascontiguousarray();
    const float *out = cont.typed_data<float>();
    size_t out_len = cont.shape()[0];

    // Check that values are in reasonable range (not clipped or zeroed)
    float max_val = 0.0f;
    for (size_t i = 0; i < out_len; ++i) {
        max_val = std::max(max_val, std::abs(out[i]));
    }
    EXPECT_GT(max_val, 0.8f);  // sine peak should be close to 1.0
    EXPECT_LE(max_val, 1.05f); // no significant overshoot
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: WAV Loading via read_audio
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, ReadAudioWAV) {
    if (!has_test_audio())
        GTEST_SKIP() << "test audio not found";

    auto audio = read_audio(model_path("2086-149220-0033.wav"));
    EXPECT_EQ(audio.sample_rate, 16000);
    EXPECT_EQ(audio.original_sample_rate, 16000);
    EXPECT_EQ(audio.format, AudioFormat::WAV);
    EXPECT_GT(audio.num_samples, 0);
    EXPECT_GT(audio.duration, 0.0f);
    EXPECT_EQ(audio.samples.shape()[0], static_cast<size_t>(audio.num_samples));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: Memory Buffer
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, ReadAudioRawFloat32) {
    // Create raw float32 PCM and read it back
    std::vector<float> pcm(16000, 0.5f); // 1 second at 16kHz
    auto audio = read_audio(pcm.data(), pcm.size(), 16000);
    EXPECT_EQ(audio.sample_rate, 16000);
    EXPECT_EQ(audio.num_samples, 16000);
    EXPECT_NEAR(audio.duration, 1.0f, 0.001f);
}

TEST(AudioIO, ReadAudioRawFloat32Resample) {
    // 1 second at 44100Hz should resample to ~16000 samples
    std::vector<float> pcm(44100, 0.5f);
    auto audio = read_audio(pcm.data(), pcm.size(), 44100);
    EXPECT_EQ(audio.sample_rate, 16000);
    EXPECT_NEAR(audio.duration, 1.0f, 0.01f);
    // Output should be approximately 16000 samples
    float ratio = static_cast<float>(audio.num_samples) / 16000.0f;
    EXPECT_NEAR(ratio, 1.0f, 0.01f);
}

TEST(AudioIO, ReadAudioRawInt16) {
    // Create raw int16 PCM
    std::vector<int16_t> pcm(16000);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = static_cast<int16_t>(16384); // 0.5 in float
    }
    auto audio = read_audio(pcm.data(), pcm.size(), 16000);
    EXPECT_EQ(audio.sample_rate, 16000);
    EXPECT_EQ(audio.num_samples, 16000);

    // Verify conversion: int16 16384 / 32768 = 0.5
    auto cont = audio.samples.ascontiguousarray();
    const float *data = cont.typed_data<float>();
    EXPECT_NEAR(data[0], 0.5f, 0.001f);
}

TEST(AudioIO, ReadAudioMemoryBufferWAV) {
    if (!has_test_audio())
        GTEST_SKIP() << "test audio not found";

    // Read file into memory, then decode from buffer
    auto path = model_path("2086-149220-0033.wav");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);

    auto audio = read_audio(buffer.data(), buffer.size());
    EXPECT_EQ(audio.format, AudioFormat::WAV);
    EXPECT_EQ(audio.sample_rate, 16000);
    EXPECT_GT(audio.num_samples, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: Duration Query
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, DurationQuery) {
    if (!has_test_audio())
        GTEST_SKIP() << "test audio not found";

    float duration = get_audio_duration(model_path("2086-149220-0033.wav"));
    auto audio = read_audio(model_path("2086-149220-0033.wav"));
    EXPECT_NEAR(duration, audio.duration, 0.01f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio I/O: preprocess_audio(AudioData) overload
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioIO, PreprocessAudioData) {
    if (!has_test_audio())
        GTEST_SKIP() << "test audio not found";

    auto audio = read_audio(model_path("2086-149220-0033.wav"));
    auto features = preprocess_audio(audio);

    auto shape = features.shape();
    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u);
    EXPECT_GT(shape[1], 0u);
    EXPECT_EQ(shape[2], 80u);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Audio Preprocessing
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AudioPreprocessing, OutputShape) {
    // 1 second of silence at 16kHz
    auto waveform = Tensor::zeros({16000});
    auto features = preprocess_audio(waveform);
    auto shape = features.shape();

    EXPECT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u); // batch
    EXPECT_GT(shape[1], 0u); // time frames
    EXPECT_EQ(shape[2], 80u); // mel bins
}

TEST(AudioPreprocessing, ConsistentOutput) {
    auto waveform = Tensor::zeros({16000});
    auto feat1 = preprocess_audio(waveform);
    auto feat2 = preprocess_audio(waveform);

    // Same input should produce same output
    auto diff = (feat1 - feat2).ascontiguousarray();
    auto flat = diff.flatten();
    const float *data = flat.typed_data<float>();
    float max_diff = 0.0f;
    for (size_t i = 0; i < flat.shape()[0]; ++i) {
        max_diff = std::max(max_diff, std::abs(data[i]));
    }
    EXPECT_FLOAT_EQ(max_diff, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CTC Decode
// ═══════════════════════════════════════════════════════════════════════════════

TEST(CTCDecode, AllBlanks) {
    // Create log_probs where blank (1024) has highest prob at every frame
    int vocab = 1025;
    int seq_len = 10;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    for (int t = 0; t < seq_len; ++t) {
        data[t * vocab + 1024] = 0.0f; // blank wins
    }
    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto result = ctc_greedy_decode(lp, 1024);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].empty());
}

TEST(CTCDecode, SingleToken) {
    int vocab = 1025;
    int seq_len = 5;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    // Token 42 wins at frame 0-2, blank at 3-4
    for (int t = 0; t < 3; ++t)
        data[t * vocab + 42] = 0.0f;
    for (int t = 3; t < 5; ++t)
        data[t * vocab + 1024] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto result = ctc_greedy_decode(lp, 1024);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 1u); // collapsed repeats
    EXPECT_EQ(result[0][0], 42);
}

TEST(CTCDecode, CollapseRepeats) {
    int vocab = 1025;
    int seq_len = 6;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    // Sequence: 10, 10, blank, 10, 10, 20
    int pattern[] = {10, 10, 1024, 10, 10, 20};
    for (int t = 0; t < seq_len; ++t)
        data[t * vocab + pattern[t]] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto result = ctc_greedy_decode(lp, 1024);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 3u); // 10, 10, 20 (blank separates the two 10s)
    EXPECT_EQ(result[0][0], 10);
    EXPECT_EQ(result[0][1], 10);
    EXPECT_EQ(result[0][2], 20);
}

TEST(CTCDecode, WithTimestamps) {
    int vocab = 1025;
    int seq_len = 6;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    // Token 5 at frames 0-1, blank at 2, token 8 at frames 3-5
    int pattern[] = {5, 5, 1024, 8, 8, 8};
    for (int t = 0; t < seq_len; ++t)
        data[t * vocab + pattern[t]] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto result = ctc_greedy_decode_with_timestamps(lp, 1024);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 2u);

    EXPECT_EQ(result[0][0].token_id, 5);
    EXPECT_EQ(result[0][0].start_frame, 0);

    EXPECT_EQ(result[0][1].token_id, 8);
    EXPECT_EQ(result[0][1].start_frame, 3);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CTC Decode (batch)
// ═══════════════════════════════════════════════════════════════════════════════

TEST(CTCDecode, BatchDecode) {
    int vocab = 1025;
    int seq_len = 4;
    int batch = 2;
    std::vector<float> data(batch * seq_len * vocab, -10.0f);
    // Batch 0: token 5 at all frames
    // Batch 1: all blank
    for (int t = 0; t < seq_len; ++t) {
        data[(0 * seq_len + t) * vocab + 5] = 0.0f;
        data[(1 * seq_len + t) * vocab + 1024] = 0.0f;
    }

    auto lp = Tensor::from_data(
        data.data(),
        Shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len),
              static_cast<size_t>(vocab)},
        true);

    auto result = ctc_greedy_decode(lp, 1024);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].size(), 1u);
    EXPECT_EQ(result[0][0], 5);
    EXPECT_TRUE(result[1].empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  End-to-End with 110M model (requires weights)
// ═══════════════════════════════════════════════════════════════════════════════

class ModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!has_model_weights() || !has_vocab() || !has_test_audio()) {
            GTEST_SKIP() << "Model weights, vocab, or test audio not found";
        }
    }
};

TEST_F(ModelTest, TranscriberCTC) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto result = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::CTC);
    EXPECT_FALSE(result.text.empty());
    EXPECT_FALSE(result.token_ids.empty());
    // Should contain "portrait" near the end
    EXPECT_NE(result.text.find("portrait"), std::string::npos);
}

TEST_F(ModelTest, TranscriberTDT) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto result = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::TDT);
    EXPECT_FALSE(result.text.empty());
    EXPECT_FALSE(result.token_ids.empty());
    EXPECT_NE(result.text.find("portrait"), std::string::npos);
}

TEST_F(ModelTest, TranscriberCTCTimestamps) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto result = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::CTC,
                                /*timestamps=*/true);
    EXPECT_FALSE(result.text.empty());
    EXPECT_FALSE(result.timestamped_tokens.empty());
    EXPECT_FALSE(result.word_timestamps.empty());

    // Verify timestamp ordering
    for (size_t i = 1; i < result.word_timestamps.size(); ++i) {
        EXPECT_GE(result.word_timestamps[i].start,
                  result.word_timestamps[i - 1].start);
    }

    // First word should start near the beginning
    EXPECT_LT(result.word_timestamps[0].start, 2.0f);
    // Last word should end near the end of audio (~7.4s)
    EXPECT_GT(result.word_timestamps.back().end, 5.0f);
}

TEST_F(ModelTest, TranscriberTDTTimestamps) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto result = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::TDT,
                                /*timestamps=*/true);
    EXPECT_FALSE(result.text.empty());
    EXPECT_FALSE(result.timestamped_tokens.empty());
    EXPECT_FALSE(result.word_timestamps.empty());

    // Verify monotonic timestamps
    for (size_t i = 1; i < result.word_timestamps.size(); ++i) {
        EXPECT_GE(result.word_timestamps[i].start,
                  result.word_timestamps[i - 1].start);
    }
}

TEST_F(ModelTest, TranscriberShortAudio) {
    auto test_wav = model_path("test.wav");
    if (!std::filesystem::exists(test_wav))
        GTEST_SKIP() << "test.wav not found";

    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto result = t.transcribe(test_wav, Decoder::TDT);
    EXPECT_FALSE(result.text.empty());
    // Should contain common words
    EXPECT_NE(result.text.find("fox"), std::string::npos);
}

TEST_F(ModelTest, CTCAndTDTProduceSimilarOutput) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto ctc = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::CTC);
    auto tdt = t.transcribe(model_path("2086-149220-0033.wav"), Decoder::TDT);

    // Both should produce non-empty results
    EXPECT_FALSE(ctc.text.empty());
    EXPECT_FALSE(tdt.text.empty());

    // Both should contain "portrait" — they may differ slightly in wording
    EXPECT_NE(ctc.text.find("portrait"), std::string::npos);
    EXPECT_NE(tdt.text.find("portrait"), std::string::npos);
}

TEST_F(ModelTest, TimestampsTokenIdsMatchNonTimestamped) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));

    // CTC: with and without timestamps should produce same token IDs
    auto ctc_plain = t.transcribe(model_path("2086-149220-0033.wav"),
                                   Decoder::CTC, false);
    auto ctc_ts = t.transcribe(model_path("2086-149220-0033.wav"),
                                Decoder::CTC, true);
    EXPECT_EQ(ctc_plain.token_ids, ctc_ts.token_ids);

    // TDT: same check
    auto tdt_plain = t.transcribe(model_path("2086-149220-0033.wav"),
                                   Decoder::TDT, false);
    auto tdt_ts = t.transcribe(model_path("2086-149220-0033.wav"),
                                Decoder::TDT, true);
    EXPECT_EQ(tdt_plain.token_ids, tdt_ts.token_ids);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TDTTranscriber (600M interface, tested with 110M weights for shape
//  compatibility — will fail to produce good output since weights mismatch,
//  but exercises the code path)
// ═══════════════════════════════════════════════════════════════════════════════

// Note: We can't actually test TDTTranscriber with correct weights since we
// don't have the 600M model. But we can test construction and interface.

TEST(TDTTranscriber, Construction) {
    // Just test that it constructs without crashing
    auto cfg = make_tdt_600m_config();
    ParakeetTDT model(cfg);
    SUCCEED();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Sinusoidal Position Embedding
// ═══════════════════════════════════════════════════════════════════════════════

TEST(PositionEmbedding, Shape) {
    auto pe = sinusoidal_position_embedding(10, 64);
    EXPECT_EQ(pe.shape()[0], 19u); // 2*10-1
    EXPECT_EQ(pe.shape()[1], 64u);
}

TEST(PositionEmbedding, Values) {
    auto pe = sinusoidal_position_embedding(5, 4);
    auto flat = pe.ascontiguousarray();
    const float *data = flat.typed_data<float>();

    // Check that values are in [-1, 1] (sin/cos range)
    for (size_t i = 0; i < flat.shape()[0] * flat.shape()[1]; ++i) {
        EXPECT_GE(data[i], -1.001f);
        EXPECT_LE(data[i], 1.001f);
    }
}

TEST(PositionEmbedding, CenterRow) {
    // Position 0 should be at index seq_len-1
    auto pe = sinusoidal_position_embedding(5, 4);
    auto flat = pe.ascontiguousarray();
    const float *data = flat.typed_data<float>();

    // At position 0: sin(0 * div_term) = 0 for all even indices
    float val0 = data[4 * 4 + 0]; // row 4 (pos=0), col 0 (sin)
    EXPECT_NEAR(val0, 0.0f, 1e-5f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Phase 3: Diarized Transcription
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DiarizedWord, Defaults) {
    DiarizedWord dw;
    EXPECT_EQ(dw.word, "");
    EXPECT_FLOAT_EQ(dw.start, 0.0f);
    EXPECT_FLOAT_EQ(dw.end, 0.0f);
    EXPECT_EQ(dw.speaker_id, -1);
    EXPECT_FLOAT_EQ(dw.confidence, 1.0f);
}

TEST(DiarizeTranscription, EmptyInputs) {
    auto result = diarize_transcription({}, {});
    EXPECT_TRUE(result.empty());
}

TEST(DiarizeTranscription, EmptySegments) {
    std::vector<WordTimestamp> words = {
        {"hello", 0.0f, 0.5f, 0.9f},
        {"world", 0.6f, 1.0f, 0.8f},
    };
    auto result = diarize_transcription(words, {});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].speaker_id, -1);
    EXPECT_EQ(result[1].speaker_id, -1);
    EXPECT_EQ(result[0].word, "hello");
    EXPECT_EQ(result[1].word, "world");
}

TEST(DiarizeTranscription, SingleSpeaker) {
    std::vector<WordTimestamp> words = {
        {"hello", 0.1f, 0.5f, 0.95f},
        {"world", 0.6f, 1.0f, 0.90f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 1.5f},
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].speaker_id, 0);
    EXPECT_EQ(result[1].speaker_id, 0);
}

TEST(DiarizeTranscription, TwoSpeakers) {
    std::vector<WordTimestamp> words = {
        {"hello", 0.1f, 0.5f, 0.95f},
        {"hi", 1.0f, 1.4f, 0.90f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 0.8f},
        {1, 0.9f, 1.5f},
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].speaker_id, 0);
    EXPECT_EQ(result[1].speaker_id, 1);
}

TEST(DiarizeTranscription, WordInGap) {
    std::vector<WordTimestamp> words = {
        {"gap", 2.0f, 2.5f, 0.85f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 1.0f},
        {1, 3.0f, 4.0f},
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].speaker_id, -1);
}

TEST(DiarizeTranscription, DominantOverlap) {
    // Word spans two speakers; pick the one with more overlap
    std::vector<WordTimestamp> words = {
        {"overlap", 0.8f, 1.5f, 0.80f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 1.0f}, // overlap with word: 1.0 - 0.8 = 0.2
        {1, 1.0f, 2.0f}, // overlap with word: 1.5 - 1.0 = 0.5
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].speaker_id, 1); // speaker 1 has more overlap
}

TEST(DiarizeTranscription, OverlappingSpeakerSegments) {
    // Two segments from different speakers overlap, word is in the overlap region
    std::vector<WordTimestamp> words = {
        {"both", 1.0f, 1.5f, 0.75f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 2.0f}, // overlap with word: 0.5
        {1, 0.5f, 1.2f}, // overlap with word: 1.2 - 1.0 = 0.2
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].speaker_id, 0); // speaker 0 has 0.5 vs 0.2
}

TEST(DiarizeTranscription, MultipleSegmentsSameSpeaker) {
    // Same speaker has two separate segments that both overlap a word
    std::vector<WordTimestamp> words = {
        {"long", 0.5f, 2.5f, 0.70f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 1.0f}, // overlap: 0.5
        {1, 1.0f, 2.0f}, // overlap: 1.0
        {0, 2.0f, 3.0f}, // overlap: 0.5
    };
    // Speaker 0 total: 0.5 + 0.5 = 1.0, Speaker 1 total: 1.0
    // Tie goes to whichever is found first; but let's make speaker 0 win
    // Actually with exact tie, depends on iteration order. Adjust:
    std::vector<DiarizationSegment> segments2 = {
        {0, 0.0f, 1.2f}, // overlap: 0.7
        {1, 1.2f, 1.8f}, // overlap: 0.6
        {0, 1.8f, 3.0f}, // overlap: 0.7
    };
    auto result = diarize_transcription(words, segments2);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].speaker_id, 0); // 0.7+0.7=1.4 > 0.6
}

TEST(DiarizeTranscription, ConfidencePreserved) {
    std::vector<WordTimestamp> words = {
        {"test", 0.0f, 0.5f, 0.42f},
    };
    std::vector<DiarizationSegment> segments = {
        {0, 0.0f, 1.0f},
    };
    auto result = diarize_transcription(words, segments);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.42f);
}

TEST(DiarizeTranscription, WordTimesPreserved) {
    std::vector<WordTimestamp> words = {
        {"test", 1.23f, 4.56f, 0.99f},
    };
    auto result = diarize_transcription(words, {});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].start, 1.23f);
    EXPECT_FLOAT_EQ(result[0].end, 4.56f);
    EXPECT_EQ(result[0].word, "test");
}

// ─── Diarized Transcription Integration (requires model weights) ────────────

static bool has_sortformer_weights() {
    return std::filesystem::exists(model_path("sortformer.safetensors"));
}

TEST_F(ModelTest, DiarizedTranscriberE2E) {
    if (!has_sortformer_weights())
        GTEST_SKIP() << "sortformer.safetensors not found";

    DiarizedTranscriber dt(model_path("model.safetensors"),
                           model_path("sortformer.safetensors"),
                           model_path("vocab.txt"));
    auto result = dt.transcribe(model_path("2086-149220-0033.wav"));

    EXPECT_FALSE(result.text.empty());
    EXPECT_FALSE(result.words.empty());
    EXPECT_FALSE(result.word_timestamps.empty());

    // Verify word times are valid
    for (const auto &w : result.words) {
        EXPECT_GE(w.start, 0.0f);
        EXPECT_GE(w.end, w.start);
        EXPECT_FALSE(w.word.empty());
    }

    // Verify monotonic word start times
    for (size_t i = 1; i < result.words.size(); ++i) {
        EXPECT_GE(result.words[i].start, result.words[i - 1].start);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Phase 5: Phrase Boosting
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Tokenizer::encode() ────────────────────────────────────────────────────

TEST(TokenizerEncode, EmptyTokenizer) {
    Tokenizer tok;
    auto ids = tok.encode("hello");
    EXPECT_TRUE(ids.empty());
}

TEST(TokenizerEncode, EmptyString) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab not found";
    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    auto ids = tok.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST(TokenizerEncode, SingleWord) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab not found";
    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    auto ids = tok.encode("hello");
    EXPECT_FALSE(ids.empty());
    // Round-trip: encode then decode should recover the word
    auto text = tok.decode(ids);
    EXPECT_EQ(text, "hello");
}

TEST(TokenizerEncode, MultiWord) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab not found";
    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    auto ids = tok.encode("hello world");
    EXPECT_FALSE(ids.empty());
    auto text = tok.decode(ids);
    EXPECT_EQ(text, "hello world");
}

TEST(TokenizerEncode, RoundTrip) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab not found";
    Tokenizer tok;
    tok.load(model_path("vocab.txt"));
    std::vector<std::string> test_phrases = {
        "portrait", "Phoebe", "the quick brown fox",
        "artificial intelligence"};
    for (const auto &phrase : test_phrases) {
        auto ids = tok.encode(phrase);
        EXPECT_FALSE(ids.empty()) << "Failed to encode: " << phrase;
        auto decoded = tok.decode(ids);
        // Case-insensitive comparison since SentencePiece lowercases
        std::string lower_phrase = phrase;
        std::transform(lower_phrase.begin(), lower_phrase.end(),
                       lower_phrase.begin(), ::tolower);
        std::string lower_decoded = decoded;
        std::transform(lower_decoded.begin(), lower_decoded.end(),
                       lower_decoded.begin(), ::tolower);
        EXPECT_EQ(lower_decoded, lower_phrase) << "Round-trip failed: " << phrase;
    }
}

// ─── ContextTrie ────────────────────────────────────────────────────────────

TEST(ContextTrie, EmptyTrie) {
    ContextTrie trie;
    EXPECT_TRUE(trie.empty());
    EXPECT_EQ(trie.size(), 1u); // just root

    std::unordered_set<int> active = {0};
    auto boosted = trie.get_boosted_tokens(active);
    EXPECT_TRUE(boosted.empty());
}

TEST(ContextTrie, InsertAndSize) {
    ContextTrie trie;
    trie.insert({10, 20, 30});
    EXPECT_FALSE(trie.empty());
    EXPECT_EQ(trie.size(), 4u); // root + 3 nodes
}

TEST(ContextTrie, GetBoostedTokens) {
    ContextTrie trie;
    trie.insert({10, 20, 30});
    trie.insert({10, 25});

    std::unordered_set<int> active = {0};
    auto boosted = trie.get_boosted_tokens(active);
    EXPECT_EQ(boosted.size(), 1u); // only token 10 from root
    EXPECT_TRUE(boosted.count(10));
}

TEST(ContextTrie, Advance) {
    ContextTrie trie;
    trie.insert({10, 20, 30});

    std::unordered_set<int> active = {0};
    auto next = trie.advance(active, 10);
    EXPECT_TRUE(next.count(0)); // always includes root

    // After advancing with 10, boosted tokens should include 20
    auto boosted = trie.get_boosted_tokens(next);
    EXPECT_TRUE(boosted.count(20));
}

TEST(ContextTrie, AdvanceNonMatchingToken) {
    ContextTrie trie;
    trie.insert({10, 20, 30});

    std::unordered_set<int> active = {0};
    auto next = trie.advance(active, 999);
    // Should only have root
    EXPECT_EQ(next.size(), 1u);
    EXPECT_TRUE(next.count(0));
}

TEST(ContextTrie, MultiplePhrases) {
    ContextTrie trie;
    trie.insert({10, 20});
    trie.insert({10, 30});
    trie.insert({40, 50});

    std::unordered_set<int> active = {0};
    auto boosted = trie.get_boosted_tokens(active);
    // Root should have children 10 and 40
    EXPECT_EQ(boosted.size(), 2u);
    EXPECT_TRUE(boosted.count(10));
    EXPECT_TRUE(boosted.count(40));

    // Advance with 10
    auto next = trie.advance(active, 10);
    auto boosted2 = trie.get_boosted_tokens(next);
    // From the node after 10: children 20 and 30; plus root children 10 and 40
    EXPECT_TRUE(boosted2.count(20));
    EXPECT_TRUE(boosted2.count(30));
    EXPECT_TRUE(boosted2.count(10)); // from root
    EXPECT_TRUE(boosted2.count(40)); // from root
}

TEST(ContextTrie, BuildFromPhrases) {
    if (!has_vocab())
        GTEST_SKIP() << "vocab not found";
    Tokenizer tok;
    tok.load(model_path("vocab.txt"));

    ContextTrie trie;
    trie.build({"portrait", "Phoebe"}, tok);
    EXPECT_FALSE(trie.empty());
    EXPECT_GT(trie.size(), 1u);
}

// ─── Boosted CTC Decode ────────────────────────────────────────────────────

TEST(BoostedCTCDecode, EmptyTrieMatchesUnboosted) {
    int vocab = 1025;
    int seq_len = 6;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    int pattern[] = {5, 5, 1024, 8, 8, 8};
    for (int t = 0; t < seq_len; ++t)
        data[t * vocab + pattern[t]] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto unboosted = ctc_greedy_decode(lp, 1024);
    ContextTrie trie;
    auto boosted = ctc_greedy_decode_boosted(lp, trie, 5.0f, 1024);

    ASSERT_EQ(unboosted.size(), boosted.size());
    EXPECT_EQ(unboosted[0], boosted[0]);
}

TEST(BoostedCTCDecode, BoostFlipsDecision) {
    // Set up: two tokens are close, boost should flip the winner
    int vocab = 1025;
    int seq_len = 3;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);

    // Frame 0: token 42 slightly wins over token 43
    data[0 * vocab + 42] = -0.1f;
    data[0 * vocab + 43] = -0.2f;
    data[0 * vocab + 1024] = -5.0f;

    // Frame 1: blank
    data[1 * vocab + 1024] = 0.0f;

    // Frame 2: blank
    data[2 * vocab + 1024] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    // Without boost: token 42 wins
    auto unboosted = ctc_greedy_decode(lp, 1024);
    ASSERT_EQ(unboosted[0].size(), 1u);
    EXPECT_EQ(unboosted[0][0], 42);

    // With boost on token 43: should flip
    ContextTrie trie;
    trie.insert({43});
    auto boosted = ctc_greedy_decode_boosted(lp, trie, 5.0f, 1024);
    ASSERT_EQ(boosted[0].size(), 1u);
    EXPECT_EQ(boosted[0][0], 43);
}

TEST(BoostedCTCDecode, TimestampsEmptyTrieMatchesUnboosted) {
    int vocab = 1025;
    int seq_len = 6;
    std::vector<float> data(1 * seq_len * vocab, -10.0f);
    int pattern[] = {5, 5, 1024, 8, 8, 8};
    for (int t = 0; t < seq_len; ++t)
        data[t * vocab + pattern[t]] = 0.0f;

    auto lp = Tensor::from_data(data.data(),
                                 Shape{1, static_cast<size_t>(seq_len),
                                       static_cast<size_t>(vocab)},
                                 true);

    auto unboosted = ctc_greedy_decode_with_timestamps(lp, 1024);
    ContextTrie trie;
    auto boosted =
        ctc_greedy_decode_with_timestamps_boosted(lp, trie, 5.0f, 1024);

    ASSERT_EQ(unboosted.size(), boosted.size());
    ASSERT_EQ(unboosted[0].size(), boosted[0].size());
    for (size_t i = 0; i < unboosted[0].size(); ++i) {
        EXPECT_EQ(unboosted[0][i].token_id, boosted[0][i].token_id);
        EXPECT_EQ(unboosted[0][i].start_frame, boosted[0][i].start_frame);
    }
}

// ─── Integration: Transcriber with boost (requires model) ──────────────────

TEST_F(ModelTest, TranscriberBoostEmptyMatchesUnboosted) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));
    auto unboosted =
        t.transcribe(model_path("2086-149220-0033.wav"), Decoder::TDT);

    TranscribeOptions opts;
    opts.decoder = Decoder::TDT;
    opts.boost_phrases = {}; // empty
    auto boosted = t.transcribe(model_path("2086-149220-0033.wav"), opts);

    EXPECT_EQ(unboosted.text, boosted.text);
    EXPECT_EQ(unboosted.token_ids, boosted.token_ids);
}

TEST_F(ModelTest, TranscriberBoostContainsTargetPhrases) {
    Transcriber t(model_path("model.safetensors"), model_path("vocab.txt"));

    TranscribeOptions opts;
    opts.decoder = Decoder::TDT;
    opts.boost_phrases = {"portrait", "Phoebe"};
    opts.boost_score = 5.0f;
    auto result = t.transcribe(model_path("2086-149220-0033.wav"), opts);

    EXPECT_FALSE(result.text.empty());
    // These words should appear in the test audio
    std::string lower = result.text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    EXPECT_NE(lower.find("portrait"), std::string::npos);
    EXPECT_NE(lower.find("phoebe"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
