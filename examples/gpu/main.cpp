// GPU acceleration and FP16 inference with timing
//
// Usage: example-gpu <model.safetensors> <vocab.txt> <audio.wav>

#include <parakeet/parakeet.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.safetensors> <vocab.txt> <audio.wav>\n";
        return 1;
    }

    // --- CPU baseline ---
    std::cout << "=== CPU (FP32) ===\n";
    {
        parakeet::Transcriber t(argv[1], argv[2]);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t.transcribe(argv[3]);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                      .count();
        std::cout << "  Text: " << result.text << "\n";
        std::cout << "  Time: " << ms << " ms\n\n";
    }

    // --- GPU FP32 ---
    std::cout << "=== GPU (FP32) ===\n";
    {
        parakeet::Transcriber t(argv[1], argv[2]);
        t.to_gpu();
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t.transcribe(argv[3]);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                      .count();
        std::cout << "  Text: " << result.text << "\n";
        std::cout << "  Time: " << ms << " ms\n\n";
    }

    // --- GPU FP16 ---
    std::cout << "=== GPU (FP16) ===\n";
    {
        parakeet::Transcriber t(argv[1], argv[2]);
        t.to_half(); // fp16 before gpu
        t.to_gpu();
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t.transcribe(argv[3]);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                      .count();
        std::cout << "  Text: " << result.text << "\n";
        std::cout << "  Time: " << ms << " ms\n";
    }

    return 0;
}
