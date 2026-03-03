# GPU Acceleration

Metal GPU acceleration and FP16 inference with timing comparison.

## Build & Run

```bash
make build
./build/examples/example-gpu model.safetensors vocab.txt audio.wav
```

Requires macOS 13+ with Apple Silicon.

## Features

- **Metal GPU**: `to_gpu()` moves model to GPU, encoder runs via MPSGraph
- **FP16**: `to_half()` casts weights to half-precision (~2x memory reduction)
- Order matters: call `to_half()` before `to_gpu()`

## Expected Output

```
=== CPU (FP32) ===
  Text: Well, I don't wish to see it anymore, ...
  Time: 2581 ms

=== GPU (FP32) ===
  Text: Well, I don't wish to see it anymore, ...
  Time: 27 ms

=== GPU (FP16) ===
  Text: Well, I don't wish to see it anymore, ...
  Time: 25 ms
```
