# Examples

Self-contained examples demonstrating parakeet.cpp features. Each example is a standalone program that compiles against the parakeet library.

## Building

All examples are built with the main project by default:

```bash
make build
# Binaries: ./build/examples/example-*
```

Disable with `-DPARAKEET_BUILD_EXAMPLES=OFF`.

## Getting Started

| Example | Description |
|---------|-------------|
| [basic](basic/) | Simplest transcription — 20 lines of code |
| [cli](cli/) | Full-featured CLI with all options |

## Feature-Specific

| Example | Description |
|---------|-------------|
| [timestamps](timestamps/) | Word-level and token-level timestamps with confidence |
| [beam-search](beam-search/) | CTC and TDT beam search with optional ARPA LM |
| [phrase-boost](phrase-boost/) | Context biasing for domain-specific vocabulary |
| [batch](batch/) | Batch transcription of multiple files |
| [vad](vad/) | Voice activity detection (standalone + ASR preprocessing) |
| [gpu](gpu/) | Metal GPU acceleration and FP16 with timing comparison |

## Streaming

| Example | Description |
|---------|-------------|
| [stream](stream/) | EOU streaming transcription with chunked audio |
| [nemotron](nemotron/) | Nemotron streaming with configurable latency modes |

## Advanced

| Example | Description |
|---------|-------------|
| [diarize](diarize/) | Sortformer speaker diarization |
| [diarized-transcription](diarized-transcription/) | ASR + diarization for speaker-attributed words |

## Language Bindings

| Example | Description |
|---------|-------------|
| [c-api](c-api/) | Pure C99 usage via the FFI interface |
