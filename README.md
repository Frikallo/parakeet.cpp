# parakeet.cpp

Fast speech recognition with NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet) models in pure C++.

Built on [axiom](https://github.com/frikallo/axiom) — a lightweight tensor library with automatic Metal GPU acceleration. No ONNX runtime, no Python runtime, no heavyweight dependencies. Just C++ and one tensor library that outruns PyTorch MPS.

**~27ms encoder inference on Apple Silicon GPU** for 10s audio (110M model) — 96x faster than CPU.

## Supported Models

| Model | Class | Size | Type | Description |
|-------|-------|------|------|-------------|
| `tdt-ctc-110m` | `ParakeetTDTCTC` | 110M | Offline | English, dual CTC/TDT decoder heads |
| `tdt-600m` | `ParakeetTDT` | 600M | Offline | Multilingual, TDT decoder |
| `eou-120m` | `ParakeetEOU` | 120M | Streaming | English, RNNT with end-of-utterance detection |
| `nemotron-600m` | `ParakeetNemotron` | 600M | Streaming | Multilingual, configurable latency (80ms–1120ms) |
| `sortformer` | `Sortformer` | 117M | Streaming | Speaker diarization (up to 4 speakers) |

All ASR models share the same audio pipeline: 16kHz mono WAV → 80-bin Mel spectrogram → FastConformer encoder.

## Quick Start

```cpp
#include <parakeet/parakeet.hpp>

parakeet::Transcriber t("model.safetensors", "vocab.txt");
t.to_gpu();  // optional — Metal acceleration

auto result = t.transcribe("audio.wav");
std::cout << result.text << std::endl;
```

Choose decoder at call site:
```cpp
auto result = t.transcribe("audio.wav", parakeet::Decoder::CTC);  // fast greedy
auto result = t.transcribe("audio.wav", parakeet::Decoder::TDT);  // better accuracy (default)
```

Word-level timestamps:
```cpp
auto result = t.transcribe("audio.wav", parakeet::Decoder::TDT, /*timestamps=*/true);
for (const auto &w : result.word_timestamps) {
    std::cout << "[" << w.start << "s - " << w.end << "s] " << w.word << std::endl;
}
```

## High-Level API

### Offline Transcription (TDT-CTC 110M)

```cpp
parakeet::Transcriber t("model.safetensors", "vocab.txt");
t.to_gpu();
auto result = t.transcribe("audio.wav");
```

### Offline Transcription (TDT 600M Multilingual)

```cpp
parakeet::TDTTranscriber t("model.safetensors", "vocab.txt",
                            parakeet::make_tdt_600m_config());
auto result = t.transcribe("audio.wav");
```

### Streaming Transcription (EOU 120M)

```cpp
parakeet::StreamingTranscriber t("model.safetensors", "vocab.txt",
                                  parakeet::make_eou_120m_config());

// Feed audio chunks (e.g., from microphone)
while (auto chunk = get_audio_chunk()) {
    auto text = t.transcribe_chunk(chunk);
    if (!text.empty()) std::cout << text << std::flush;
}
std::cout << t.get_text() << std::endl;
```

### Streaming Transcription (Nemotron 600M)

```cpp
// Latency modes: 0=80ms, 1=160ms, 6=560ms, 13=1120ms
auto cfg = parakeet::make_nemotron_600m_config(/*latency_frames=*/1);
parakeet::NemotronTranscriber t("model.safetensors", "vocab.txt", cfg);

while (auto chunk = get_audio_chunk()) {
    auto text = t.transcribe_chunk(chunk);
    if (!text.empty()) std::cout << text << std::flush;
}
```

### Speaker Diarization (Sortformer 117M)

Identify who spoke when — detects up to 4 speakers with per-frame activity probabilities:

```cpp
parakeet::Sortformer model(parakeet::make_sortformer_117m_config());
model.load_state_dict(axiom::io::safetensors::load("sortformer.safetensors"));

auto audio = parakeet::read_audio("meeting.wav");
auto features = parakeet::preprocess_audio(audio.samples, {.normalize = false});
auto segments = model.diarize(features);

for (const auto &seg : segments) {
    std::cout << "Speaker " << seg.speaker_id
              << ": [" << seg.start << "s - " << seg.end << "s]" << std::endl;
}
// Speaker 0: [0.56s - 2.96s]
// Speaker 0: [3.36s - 4.40s]
// Speaker 1: [4.80s - 6.24s]
```

Streaming diarization with arrival-order speaker tracking:

```cpp
parakeet::Sortformer model(parakeet::make_sortformer_117m_config());
model.load_state_dict(axiom::io::safetensors::load("sortformer.safetensors"));

parakeet::EncoderCache enc_cache;
parakeet::AOSCCache aosc_cache(4);  // max 4 speakers

while (auto chunk = get_audio_chunk()) {
    auto features = parakeet::preprocess_audio(chunk, {.normalize = false});
    auto segments = model.diarize_chunk(features, enc_cache, aosc_cache);
    for (const auto &seg : segments) {
        std::cout << "Speaker " << seg.speaker_id
                  << ": [" << seg.start << "s - " << seg.end << "s]" << std::endl;
    }
}
```

## Low-Level API

For full control over the pipeline:

**CTC** (English, punctuation & capitalization):
```cpp
auto cfg = parakeet::make_110m_config();
parakeet::ParakeetTDTCTC model(cfg);
model.load_state_dict(axiom::io::safetensors::load("model.safetensors"));

auto audio = parakeet::read_audio("audio.wav");
auto features = parakeet::preprocess_audio(audio.samples);
auto encoder_out = model.encoder()(features);

auto log_probs = model.ctc_decoder()(encoder_out);
auto tokens = parakeet::ctc_greedy_decode(log_probs);

parakeet::Tokenizer tokenizer;
tokenizer.load("vocab.txt");
std::cout << tokenizer.decode(tokens[0]) << std::endl;
```

**TDT** (Token-and-Duration Transducer):
```cpp
auto encoder_out = model.encoder()(features);
auto tokens = parakeet::tdt_greedy_decode(model, encoder_out, cfg.durations);
std::cout << tokenizer.decode(tokens[0]) << std::endl;
```

**Timestamps** (CTC or TDT):
```cpp
// CTC timestamps
auto ts = parakeet::ctc_greedy_decode_with_timestamps(log_probs);

// TDT timestamps
auto ts = parakeet::tdt_greedy_decode_with_timestamps(model, encoder_out, cfg.durations);

// Group into word-level timestamps
auto words = parakeet::group_timestamps(ts[0], tokenizer.pieces());
```

**GPU acceleration** (Metal):
```cpp
model.to(axiom::Device::GPU);
auto features_gpu = features.gpu();
auto encoder_out = model.encoder()(features_gpu);

// Decode on CPU
auto tokens = parakeet::ctc_greedy_decode(
    model.ctc_decoder()(encoder_out).cpu()
);
```

## CLI

```
Usage: parakeet <model.safetensors> <audio.wav> [options]

Model types:
  --model TYPE     Model type (default: tdt-ctc-110m)
                   Types: tdt-ctc-110m, tdt-600m, eou-120m,
                          nemotron-600m, sortformer

Decoder options:
  --ctc            Use CTC decoder (default: TDT)
  --tdt            Use TDT decoder

Other options:
  --vocab PATH     SentencePiece vocab file
  --gpu            Run on Metal GPU
  --timestamps     Show word-level timestamps
  --streaming      Use streaming mode (eou/nemotron models)
  --latency N      Right context frames for nemotron (0/1/6/13)
  --features PATH  Load pre-computed features from .npy file
```

Examples:

```bash
# Basic transcription (TDT decoder, default)
./build/parakeet model.safetensors audio.wav --vocab vocab.txt

# CTC decoder
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --ctc

# GPU acceleration
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --gpu

# Word-level timestamps
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --timestamps

# 600M multilingual TDT model
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model tdt-600m

# Streaming with EOU
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model eou-120m

# Nemotron streaming with configurable latency
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model nemotron-600m --latency 6

# Speaker diarization
./build/parakeet sortformer.safetensors meeting.wav --model sortformer
# Speaker 0: [0.56s - 2.96s]
# Speaker 0: [3.36s - 4.40s]
# Speaker 1: [4.80s - 6.24s]
```

## Setup

### Build

Requires C++20. Axiom is the only dependency (included as a submodule).

```bash
git clone --recursive https://github.com/frikallo/parakeet.cpp
cd parakeet.cpp
make build
```

### Test

```bash
make test
```

### Convert Weights

Download a NeMo checkpoint from NVIDIA and convert to safetensors:

```bash
# Download from HuggingFace (requires pip install huggingface_hub)
huggingface-cli download nvidia/parakeet-tdt_ctc-110m --include "*.nemo" --local-dir .

# Convert to safetensors
pip install safetensors torch
python scripts/convert_nemo.py parakeet-tdt_ctc-110m.nemo -o model.safetensors
```

The converter supports all model types via the `--model` flag:

```bash
# 110M TDT-CTC (default)
python scripts/convert_nemo.py checkpoint.nemo -o model.safetensors --model 110m-tdt-ctc

# 600M multilingual TDT
python scripts/convert_nemo.py checkpoint.nemo -o model.safetensors --model 600m-tdt

# 120M EOU streaming
python scripts/convert_nemo.py checkpoint.nemo -o model.safetensors --model eou-120m

# 600M Nemotron streaming
python scripts/convert_nemo.py checkpoint.nemo -o model.safetensors --model nemotron-600m

# 117M Sortformer diarization
python scripts/convert_nemo.py checkpoint.nemo -o model.safetensors --model sortformer
```

Also supports raw `.ckpt` files and inspection:
```bash
python scripts/convert_nemo.py model_weights.ckpt -o model.safetensors
python scripts/convert_nemo.py --dump model.nemo  # inspect checkpoint keys
```

### Download Vocab

Grab the SentencePiece vocab from the same HuggingFace repo. The file is inside the `.nemo` archive, or download directly:

```bash
# Extract from .nemo
tar xf parakeet-tdt_ctc-110m.nemo ./tokenizer.model
# or use the vocab.txt from the HF files page
```

## Architecture

### Offline Models

Built on a shared FastConformer encoder (Conv2d 8x subsampling → N Conformer blocks with relative positional attention):

| Model | Class | Decoder | Use case |
|-------|-------|---------|----------|
| CTC | `ParakeetCTC` | Greedy argmax | Fast, English-only |
| RNNT | `ParakeetRNNT` | Autoregressive LSTM | Streaming capable |
| TDT | `ParakeetTDT` | LSTM + duration prediction | Better accuracy than RNNT |
| TDT-CTC | `ParakeetTDTCTC` | Both TDT and CTC heads | Switch decoder at inference |

### Streaming Models

Built on a cache-aware streaming FastConformer encoder with causal convolutions and bounded-context attention:

| Model | Class | Decoder | Use case |
|-------|-------|---------|----------|
| EOU | `ParakeetEOU` | Streaming RNNT | End-of-utterance detection |
| Nemotron | `ParakeetNemotron` | Streaming TDT | Configurable latency streaming |

### Diarization

| Model | Class | Architecture | Use case |
|-------|-------|-------------|----------|
| Sortformer | `Sortformer` | NEST encoder → Transformer → sigmoid | Speaker diarization (up to 4 speakers) |

## Benchmarks

Measured on Apple M3 16GB with simulated audio input (`Tensor::randn`). Times are per-encoder-forward-pass (Sortformer: full forward pass).

**Encoder throughput — 10s audio:**

| Model | Params | CPU (ms) | GPU (ms) | GPU Speedup |
|-------|--------|----------|----------|-------------|
| 110m (TDT-CTC) | 110M | 2,581 | 27 | **96x** |
| tdt-600m | 600M | 10,779 | 520 | **21x** |
| rnnt-600m | 600M | 10,648 | 1,468 | **7x** |
| sortformer | 117M | 3,195 | 479 | **7x** |

**110m GPU scaling across audio lengths:**

| Audio | CPU (ms) | GPU (ms) | RTF | Throughput |
|-------|----------|----------|-----|------------|
| 1s | 262 | 24 | 0.024 | 41x |
| 5s | 1,222 | 26 | 0.005 | 190x |
| 10s | 2,581 | 27 | 0.003 | 370x |
| 30s | 10,061 | 32 | 0.001 | 935x |
| 60s | 26,559 | 72 | 0.001 | 833x |

GPU acceleration powered by axiom's Metal graph compiler which fuses the full encoder into optimized MPSGraph operations.

### Running benchmarks

```bash
# Full suite
make bench ARGS="--110m=models/model.safetensors --tdt-600m=models/tdt.safetensors"

# Single model
make bench-single ARGS="--110m=models/model.safetensors --benchmark_filter=110m"

# Markdown table output
./build/parakeet_bench --110m=models/model.safetensors --markdown

# Skip GPU benchmarks
./build/parakeet_bench --110m=models/model.safetensors --no-gpu
```

Available model flags: `--110m`, `--tdt-600m`, `--rnnt-600m`, `--sortformer`. All Google Benchmark flags (`--benchmark_filter`, `--benchmark_format=json`, `--benchmark_repetitions=N`) are passed through.

## Roadmap

### Tier 1 — High Impact

- [ ] **Confidence scores** — Per-word confidence (0.0–1.0) from token log-probs. Entropy-based or max-logprob aggregation.
- [ ] **Phrase boosting (context biasing)** — Token-level trie over a boost list. Bias log-probs during decode for domain-specific vocabulary (product names, jargon, proper nouns). Works with greedy decode.
- [ ] **Beam search decoding** — CTC prefix beam search and TDT/RNNT beam search with configurable width. 5–15% relative WER reduction over greedy.
- [ ] **N-gram LM shallow fusion** — Load ARPA language models, score partial hypotheses during beam search. Domain-adapted decoding.

### Audio & I/O

- [x] **Multi-format audio loading** — WAV (all formats), FLAC, MP3, OGG Vorbis via dr_libs + stb_vorbis. `read_audio(path)` auto-detects format.
- [x] **Automatic resampling** — Windowed sinc interpolation (Kaiser, 16-tap, ~80dB stopband). Arbitrary rate conversion with GCD simplification.
- [x] **Sample rate validation** — `preprocess_audio(AudioData)` validates sample rate matches config.
- [x] **Load from memory buffer** — `read_audio(bytes, len)`, `read_audio(float*, n, rate)`, `read_audio(int16_t*, n, rate)`.
- [x] **Extended WAV support** — All WAV formats via dr_wav (8/16/24/32-bit PCM, float, A-law, mu-law).
- [ ] **Audio duration query** — `get_audio_duration(path)` without fully decoding the file. Read header only.
- [ ] **Progress callbacks** — `transcribe(path, {.on_progress = callback})` for long files. Report preprocessing / encoder / decode stages.
- [ ] **Streaming from raw PCM** — Helper to feed `int16_t*` or `float*` microphone buffers directly into `StreamingTranscriber` without manual Tensor construction.

### Tier 2 — Production Readiness

- [ ] **Diarized transcription** — Fuse Sortformer speaker segments with ASR word timestamps. Output: "Speaker 1: Hello. Speaker 2: Hi there."
- [ ] **Long-form audio chunking** — Split audio >30s into overlapping windows, run encoder on each, merge transcriptions at overlap boundaries.
- [ ] **VAD (voice activity detection)** — Skip silent regions, reduce compute. Silero VAD integration or energy-based.
- [ ] **Batch inference** — Pad + length-mask multiple audio files, batch through encoder and decoder. GPU utilization improvement.
- [ ] **Neural LM rescoring** — N-best reranking with a Transformer LM after beam search.

### Tier 3 — Ecosystem

- [ ] **C API** — Flat C interface (`parakeet_transcribe(...)`) for FFI from Python, Swift, Go, Rust.
- [ ] **f16 inference** — Half-precision weights and compute. 2x memory reduction, faster on Apple Silicon.
- [ ] **Model quantization** — INT8/INT4 weight quantization for mobile deployment.
- [ ] **Hotword / wake word detection** — "Hey Parakeet" trigger phrase detection.
- [ ] **Speaker embedding extraction** — Speaker verification from Sortformer intermediate layers or TitaNet.

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- Offline models have ~4-5 minute audio length limits; split longer files or use streaming models
- Blank token ID is 1024 (110M) or 8192 (600M)
- GPU acceleration requires Apple Silicon with Metal support
- Timestamps use frame-level alignment: `frame * 0.08s` (8x subsampling × 160 hop / 16kHz)
- Sortformer diarization uses unnormalized features (`normalize = false`) — this differs from ASR models

## License

MIT
