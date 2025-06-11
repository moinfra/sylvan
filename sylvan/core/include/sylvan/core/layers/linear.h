// sylvan/core/layers/linear.h
//
// Defines the Linear (Dense or Fully Connected) layer, which performs a
// linear transformation of the input data: y = x * W^T + b.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include "sylvan/core/autograd.h"

namespace sylvan::core {
/**
 * @brief Implements a Linear (Fully Connected) layer.
 *
 * This layer performs a linear transformation on its input: `output = input @ W.T + b`.
 * It includes learnable weight (`W`) and bias (`b`) parameters.
 */
struct Linear {
    Variable W; ///< Learnable weight matrix. Shape is `[out_features, in_features]`.
    Variable b; ///< Learnable bias vector. Shape is `[out_features]`.

    /**
     * @brief Constructs a Linear layer.
     *
     * Initializes the weight matrix `W` and bias vector `b`.
     *
     * @param in_features The number of input features (size of the last dimension of input tensor).
     * @param out_features The number of output features (size of the last dimension of output tensor).
     */
    Linear(int in_features, int out_features);

    /**
     * @brief Performs the forward pass of the Linear layer.
     *
     * Computes the linear transformation `x @ W.T + b`.
     *
     * @param ctx The graph context for tracing operations.
     * @param x A Variable containing the input tensor. Its last dimension must match `in_features`.
     * @return A Variable containing the output tensor. Its last dimension will be `out_features`.
     */
    Variable* forward(GraphContext& ctx, Variable* x);

    /**
     * @brief Returns a vector of learnable parameters for this layer.
     *
     * @return A `std::vector` containing pointers to the `W` and `b` Variables.
     */
    std::vector<Variable*> parameters();
};
} // namespace sylvan::core
