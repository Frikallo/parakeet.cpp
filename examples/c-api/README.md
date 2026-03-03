# C API

Pure C99 usage of parakeet.cpp via the FFI interface. Useful for calling from Python, Swift, Go, Rust, or any language with C FFI support.

## Build & Run

```bash
make build
./build/examples/example-c-api model.safetensors vocab.txt audio.wav
```

## API Overview

```c
#include <parakeet/api/parakeet_c.h>

// Create → transcribe → read result → free
parakeet_transcriber_t t = parakeet_transcriber_create(weights, vocab, NULL);
parakeet_result_t r = parakeet_transcriber_transcribe_file(t, "audio.wav", NULL);
printf("%s\n", parakeet_result_text(r));
parakeet_result_free(r);
parakeet_transcriber_free(t);
```

All 5 transcriber types are wrapped, plus audio I/O, VAD, and result accessors. Error handling uses a thread-local error string via `parakeet_last_error()`.

See [`parakeet_c.h`](../../include/parakeet/api/parakeet_c.h) for the full API reference.
