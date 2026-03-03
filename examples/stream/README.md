# Streaming Transcription (EOU)

Streaming transcription using the EOU (end-of-utterance) model with chunked audio input.

## Build & Run

```bash
make build
./build/examples/example-stream eou_model.safetensors vocab.txt audio.wav
```

Requires an EOU-120M model (`--model eou-120m` weights).

## How It Works

`StreamingTranscriber` maintains internal state (encoder cache, decoder state) across calls to `transcribe_chunk()`. Feed audio in any chunk size — the example uses 0.5s chunks to simulate a microphone stream.

```cpp
parakeet::StreamingTranscriber t("model.safetensors", "vocab.txt",
                                  parakeet::make_eou_120m_config());
while (has_audio) {
    auto text = t.transcribe_chunk(samples, n);
    if (!text.empty()) std::cout << text << std::flush;
}
```
