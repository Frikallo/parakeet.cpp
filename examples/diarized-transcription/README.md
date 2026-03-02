# Diarized Transcription

Combines ASR word timestamps with Sortformer speaker diarization to produce speaker-attributed words.

## Build & Run

```bash
make build
./build/examples/example-diarized-transcription model.safetensors sortformer.safetensors vocab.txt meeting.wav
```

Requires both ASR (110M TDT-CTC) and Sortformer (117M) weights.

## How It Works

`DiarizedTranscriber` runs ASR and Sortformer in parallel on the same audio, then aligns word timestamps with speaker segments by maximum temporal overlap. Consecutive words from the same speaker are grouped automatically.

## Expected Output

```
Speaker 0 [0.08s]: Good morning, how can I help you today?
Speaker 1 [2.88s]: Hi, I'd like to check on my order status please.
Speaker 0 [5.76s]: Sure, can you give me your order number?
```
