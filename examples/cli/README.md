# CLI

The `parakeet` CLI provides a command-line interface for all supported models and features.

## Build

The CLI is built automatically with the main project:

```bash
make build
# Binary: ./build/parakeet
```

Disable with `make build CLI=OFF`.

## Usage

```
Usage: parakeet <model.safetensors> <audio.wav> [audio2.wav ...] [options]

Model types:
  --model TYPE     Model type (default: tdt-ctc-110m)
                   Types: tdt-ctc-110m, tdt-600m, eou-120m,
                          nemotron-600m, sortformer, diarized

Decoder options:
  --ctc            Use CTC greedy decoder (default: TDT)
  --tdt            Use TDT decoder
  --ctc-beam       Use CTC beam search decoder
  --tdt-beam       Use TDT beam search decoder
  --beam-width N   Beam width for beam search (default: 8)
  --lm PATH        ARPA language model for beam search
  --lm-weight N    LM interpolation weight (default: 0.5)

Phrase boost:
  --boost PHRASE   Boost a phrase (repeatable)
  --boost-score N  Boost score (default: 5.0)

Other options:
  --vocab PATH     SentencePiece vocab file
  --sortformer-weights PATH  Sortformer weights (for diarized mode)
  --gpu            Run on Metal GPU
  --fp16           Use half-precision inference (less memory, requires --gpu)
  --timestamps     Show word-level timestamps
  --streaming      Use streaming mode (eou/nemotron models)
  --latency N      Right context frames for nemotron (0/1/6/13)
  --vad PATH       Enable Silero VAD with given weights file
  --vad-threshold F  VAD speech threshold (default: 0.5)
  --features PATH  Load pre-computed features from .npy file

Batch mode:
  Multiple audio files use batched encoder inference.
  Supported for tdt-ctc-110m and tdt-600m models.
```

## Examples

```bash
# Basic transcription (TDT decoder, default)
./build/parakeet model.safetensors audio.wav --vocab vocab.txt

# CTC decoder
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --ctc

# GPU acceleration
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --gpu

# GPU + FP16 (half memory usage)
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --gpu --fp16

# Word-level timestamps
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --timestamps

# CTC beam search
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --ctc-beam --beam-width 16

# TDT beam search
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --tdt-beam --beam-width 4

# CTC beam search with ARPA language model
./build/parakeet model.safetensors audio.wav --vocab vocab.txt \
  --ctc-beam --lm lm.arpa --lm-weight 0.5

# Phrase boosting for domain-specific terms
./build/parakeet model.safetensors audio.wav --vocab vocab.txt \
  --boost "Phoebe" --boost "portrait" --boost-score 5.0

# Batch inference (multiple files in one forward pass)
./build/parakeet model.safetensors audio1.wav audio2.wav audio3.wav --vocab vocab.txt --gpu

# 600M multilingual TDT model
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model tdt-600m

# Streaming with EOU
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model eou-120m

# Nemotron streaming with configurable latency
./build/parakeet model.safetensors audio.wav --vocab vocab.txt --model nemotron-600m --latency 6

# Speaker diarization
./build/parakeet sortformer.safetensors meeting.wav --model sortformer

# VAD preprocessing (strip silence before ASR)
./build/parakeet model.safetensors audio.wav --vocab vocab.txt \
  --vad silero_vad_v5.safetensors --timestamps

# Diarized transcription (ASR + Sortformer)
./build/parakeet model.safetensors meeting.wav --model diarized \
  --sortformer-weights sortformer.safetensors --vocab vocab.txt
```
