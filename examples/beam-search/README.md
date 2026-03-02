# Beam Search

CTC and TDT beam search decoding with optional ARPA language model rescoring.

## Build & Run

```bash
make build

# Without LM
./build/examples/example-beam-search model.safetensors vocab.txt audio.wav

# With ARPA language model
./build/examples/example-beam-search model.safetensors vocab.txt audio.wav lm.arpa
```

## Features

- **CTC beam search**: prefix beam search with configurable width
- **TDT beam search**: time-synchronous beam search
- **ARPA LM fusion**: n-gram language model scoring at word boundaries
- Beam search with timestamps

## Decoders Compared

| Decoder | Speed | Accuracy |
|---------|-------|----------|
| Greedy TDT | Fastest | Good |
| CTC beam (width=16) | Moderate | Better |
| TDT beam (width=4) | Moderate | Best |
| + ARPA LM | Slower | Best with domain LM |
