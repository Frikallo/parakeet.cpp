# Nemotron Streaming

Nemotron 600M multilingual streaming transcription with configurable latency.

## Build & Run

```bash
make build

# Default latency (160ms)
./build/examples/example-nemotron nemotron_model.safetensors vocab.txt audio.wav

# Custom latency
./build/examples/example-nemotron nemotron_model.safetensors vocab.txt audio.wav 6
```

Requires a Nemotron-600M model (`--model nemotron-600m` weights).

## Latency Modes

| `latency` | Right Context | Latency |
|-----------|--------------|---------|
| 0 | 0 frames | 80ms |
| 1 | 1 frame | 160ms |
| 6 | 6 frames | 560ms |
| 13 | 13 frames | 1120ms |

Lower latency = faster response, higher latency = better accuracy.
