#include "../include/parakeet.hpp"
#include <iostream>

namespace parakeet {

void init(const std::string &model_path) {
    std::cout << "Parakeet initialized" << std::endl;
}

void run(const Tensor &input) {
    std::cout << "Parakeet running" << std::endl;
}

void cleanup() {
    std::cout << "Parakeet cleaned up" << std::endl;
}

} // namespace parakeet

int main() {
    Tensor input = Tensor::randn({1, 3, 224, 224}, DType::Float32, Device::GPU);
    std::cout << "Input: " << input << std::endl;
    return 0;
}