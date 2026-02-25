#pragma once

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

using namespace axiom;

namespace parakeet {

nn::Module model_;

Tensor input_;
Tensor output_;

void init(const std::string &model_path);

void run(const Tensor &input);

void cleanup();

} // namespace parakeet