# Voice Activity Detection

Standalone Silero VAD usage and ASR with VAD preprocessing.

## Build & Run

```bash
make build
./build/examples/example-vad model.safetensors vocab.txt silero_vad_v5.safetensors audio.wav
```

## Features

- **Standalone VAD**: detect speech segments, get start/end sample indices
- **Speech collection**: strip silence from audio (`collect_speech()`)
- **ASR + VAD**: `enable_vad()` on any transcriber — silence auto-filtered, timestamps remapped to original timeline

## Converting VAD Weights

```bash
pip install safetensors torch
python scripts/convert_silero_vad.py -o silero_vad_v5.safetensors
```
