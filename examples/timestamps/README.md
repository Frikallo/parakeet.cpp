# Timestamps

Word-level and token-level timestamps with confidence scores.

## Build & Run

```bash
make build
./build/examples/example-timestamps model.safetensors vocab.txt audio.wav
```

## Features

- **Word timestamps**: start/end time and confidence per word
- **Token timestamps**: frame-level timing per BPE token
- Confidence scores in [0, 1] range (from token log-probabilities)

## Expected Output

```
Text: Well, I don't wish to see it anymore, ...

Word timestamps:
  [0.24s - 0.48s] (0.98) Well,
  [0.48s - 0.56s] (0.95) I
  [0.56s - 0.96s] (0.87) don't

Token timestamps (42 tokens):
  token=123 frames=[3-6] conf=0.982
  ...
```
