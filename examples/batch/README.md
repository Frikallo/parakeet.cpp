# Batch Transcription

Transcribe multiple audio files in a single batched encoder forward pass for better throughput.

## Build & Run

```bash
make build
./build/examples/example-batch model.safetensors vocab.txt audio1.wav audio2.wav audio3.wav
```

## How It Works

`transcribe_batch()` pads all inputs to the same length, runs a single batched encoder forward pass, then decodes each output independently. This is more efficient than sequential transcription, especially on GPU.

Available on `Transcriber`, `TDTTranscriber`, the C API (`parakeet_transcriber_transcribe_batch`), and the CLI (multiple positional audio args).
