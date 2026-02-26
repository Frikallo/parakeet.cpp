# parakeet.cpp

Fast speech recognition with NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet-702d03111484ef) models in pure C++.

Built on [axiom](https://github.com/noahkay13/axiom) — a lightweight tensor library with automatic Metal GPU acceleration. No ONNX runtime, no Python runtime, no heavyweight dependencies. Just C++ and one tensor library that outruns PyTorch MPS.

**~13ms encoder inference on Apple Silicon GPU** — 2.5x faster than PyTorch MPS (32ms).

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

Transcribe from raw samples:
```cpp
auto wav = parakeet::read_wav("audio.wav");
auto result = t.transcribe(wav.samples);
```

## Low-Level API

For full control over the pipeline:

**CTC** (English, punctuation & capitalization):
```cpp
#include <parakeet/parakeet.hpp>

auto cfg = parakeet::make_110m_config();
parakeet::ParakeetTDTCTC model(cfg);
model.load_state_dict(axiom::io::safetensors::load("model.safetensors"));

auto wav = parakeet::read_wav("audio.wav");
auto features = parakeet::preprocess_audio(wav.samples);
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

## Setup

### Build

Requires C++20. Axiom is the only dependency (included as a submodule).

```bash
git clone --recursive https://github.com/noahkay13/parakeet.cpp
cd parakeet.cpp
make build
```

### Convert weights

Download a NeMo checkpoint from NVIDIA and convert to safetensors:

```bash
# Download from HuggingFace (requires pip install huggingface_hub)
huggingface-cli download nvidia/parakeet-tdt_ctc-110m --include "*.nemo" --local-dir .

# Convert to safetensors
pip install safetensors torch
python scripts/convert_nemo.py parakeet-tdt_ctc-110m.nemo -o model.safetensors
```

The converter also supports raw `.ckpt` files:
```bash
python scripts/convert_nemo.py model_weights.ckpt -o model.safetensors

# Inspect checkpoint keys
python scripts/convert_nemo.py --dump model.nemo
```

### Download vocab

Grab the SentencePiece vocab from the same HuggingFace repo. The file is inside the `.nemo` archive, or download directly:

```bash
# Extract from .nemo
tar xf parakeet-tdt_ctc-110m.nemo ./tokenizer.model
# or use the vocab.txt from the HF files page
```

### Run

```bash
./build/parakeet model.safetensors audio.wav --vocab vocab.txt
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --ctc
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --gpu
```

## Architecture

Four model variants built on a shared FastConformer encoder:

| Model | Class | Decoder | Use case |
|-------|-------|---------|----------|
| CTC | `ParakeetCTC` | Greedy argmax | Fast, English-only |
| RNNT | `ParakeetRNNT` | Autoregressive LSTM | Streaming capable |
| TDT | `ParakeetTDT` | LSTM + duration prediction | Better accuracy than RNNT |
| TDT-CTC | `ParakeetTDTCTC` | Both TDT and CTC heads | Switch decoder at inference |

All models use the same audio pipeline: 16kHz mono WAV → Mel spectrogram (80 bins, 512-point FFT) → FastConformer encoder (Conv2d 8x subsampling → N Conformer blocks with relative positional attention).

## API Reference

Everything lives in the `parakeet` namespace:

```cpp
// High-level
Transcriber t(weights_path, vocab_path);                      // load model + vocab
t.to_gpu();                                                    // optional Metal acceleration
TranscribeResult result = t.transcribe("audio.wav");           // WAV file → text
TranscribeResult result = t.transcribe(samples, Decoder::CTC); // raw samples → text

// Audio I/O
WavData read_wav(const std::string &path);                    // 16-bit PCM or 32-bit float WAV
Tensor preprocess_audio(const Tensor &waveform,               // waveform → (1, frames, 80) Mel features
                        const AudioConfig &config = {});

// Tokenizer
Tokenizer tokenizer;
tokenizer.load("vocab.txt");                                  // SentencePiece BPE vocab
std::string text = tokenizer.decode(token_ids);               // token IDs → text

// Models — construct with config, load weights from safetensors
ParakeetCTC model(CTCConfig{});
ParakeetTDT model(TDTConfig{});
ParakeetTDTCTC model(TDTCTCConfig{});
ParakeetRNNT model(RNNTConfig{});

// Presets
TDTCTCConfig cfg = make_110m_config();                        // nvidia/parakeet-tdt_ctc-110m

// Decoding
ctc_greedy_decode(log_probs, blank_id=1024);                  // CTC: argmax → collapse → remove blank
tdt_greedy_decode(model, encoder_out, durations);             // TDT: per-frame with duration prediction
rnnt_greedy_decode(model, encoder_out);                       // RNNT: per-frame autoregressive
```

## Benchmarks

Measured on Apple M3 16GB, 110M parameter model, ~10s audio clip:

| Stage | CPU | GPU (Metal) |
|-------|-----|-------------|
| Preprocessing | ~25ms | ~25ms |
| Encoder | ~1848ms | **~13ms** |
| CTC decode | ~2ms | ~2ms |

**GPU encoder is ~140x faster than CPU and 2.5x faster than PyTorch MPS** — powered entirely by axiom's Metal graph compiler which fuses the full encoder into optimized MPSGraph operations.

## Notes

- Audio: 16kHz mono WAV (16-bit PCM or 32-bit float)
- CTC/TDT models have ~4-5 minute audio length limits; split longer files into chunks
- Blank token ID is 1024 (not 0)
- GPU acceleration requires Apple Silicon with Metal support

## License

MIT
