// sylvan/core/src/layers/layernorm.cc
//
// Implements the LayerNorm module's logic, specifically its constructor,
// forward pass, and parameter collection. It delegates the core normalization
// operation to a free autograd function for proper graph tracing.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/layers/layernorm.h"
#include "sylvan/tensor/operators.h"

namespace sylvan::core {

Variable* layer_norm(GraphContext& ctx, Variable* x, Variable* gain, Variable* bias, float eps);

} // namespace sylvan::core

namespace sylvan::core {

LayerNorm::LayerNorm(int feature_dim, float eps)
    : gain(sylvan::tensor::ops::from_host(std::vector<float>(feature_dim, 1.0f), {1, (int64_t)feature_dim})),
      bias(sylvan::tensor::ops::from_host(std::vector<float>(feature_dim, 0.0f), {1, (int64_t)feature_dim})),
      eps(eps) {}

Variable* LayerNorm::forward(GraphContext& ctx, Variable* x) {
    return layer_norm(ctx, x, &gain, &bias, eps);
}

std::vector<Variable*> LayerNorm::parameters() { return {&gain, &bias}; }

} // namespace sylvan::core
