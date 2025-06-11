// sylvan/core/src/layers/linear.cc
//
// Implements the Linear (Dense or Fully Connected) layer's constructor,
// forward pass, and parameter management. It initializes weights with Kaiming
// Uniform (He) initialization and biases to zeros.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/layers/linear.h"
#include "sylvan/tensor/operators.h"
#include <cmath> 

namespace sylvan::core {

namespace {

sylvan::tensor::Tensor create_weights(int in_features, int out_features) {
  // Kaiming Uniform (He) initialization for ReLU activations.
  float std = std::sqrt(2.0f / static_cast<float>(in_features));
  sylvan::tensor::Tensor t({(int64_t)in_features, (int64_t)out_features},
                           sylvan::tensor::DType::Float32);
  sylvan::tensor::ops::uniform_(t, -std,
                                std); // Fill with random uniform values.
  return t;
}

sylvan::tensor::Tensor create_bias(int out_features) {
  sylvan::tensor::Tensor t({1, (int64_t)out_features},
                           sylvan::tensor::DType::Float32);
  sylvan::tensor::ops::fill_(t, 0.0f); // Fill with zeros.
  return t;
}

} // namespace

Linear::Linear(int in_features, int out_features)
    : W(create_weights(in_features, out_features)),
      b(create_bias(out_features)) {}

Variable *Linear::forward(GraphContext &ctx, Variable *x) {
  auto out = matmul(ctx, x, &W); // Matrix multiplication: input * weights
  return add(ctx, out, &b);      // Add bias
}

std::vector<Variable *> Linear::parameters() { return {&W, &b}; }

} // namespace sylvan::core
