# Basic Transcription

The simplest possible transcription with parakeet.cpp.

## Build & Run

```bash
make build
./build/examples/example-basic model.safetensors vocab.txt audio.wav
```

## Code

```cpp
#include <parakeet/parakeet.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    parakeet::Transcriber t(argv[1], argv[2]);
    auto result = t.transcribe(argv[3]);
    std::cout << result.text << std::endl;
}
```

## Expected Output

```
Well, I don't wish to see it anymore, observed Phoebe, turning away her eyes.
It is certainly very like the old portrait.
```
