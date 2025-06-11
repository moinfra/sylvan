// sylvan/core/layers/layernorm.h
//
// Defines the Layer Normalization layer, which normalizes inputs across the
// feature dimension, independently for each sample in a batch.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include "sylvan/core/autograd.h"

namespace sylvan::core {
/**
 * @brief Implements the Layer Normalization operation.
 *
 * Layer Normalization normalizes the input across the feature dimension,
 * applying a learnable gain (gamma) and bias (beta). It helps stabilize
 * training in deep neural networks.
 */
struct LayerNorm {
    Variable gain; ///< Learnable scaling factor (gamma). Applied element-wise after normalization.
    Variable bias; ///< Learnable additive bias (beta). Applied element-wise after scaling.
    float eps;     ///< A small value added to the variance for numerical stability.

    /**
     * @brief Constructs a LayerNorm layer.
     *
     * @param feature_dim The dimension along which normalization is applied.
     *                    This is typically the last dimension of the input tensor.
     * @param eps A small constant added to the variance to prevent division by zero.
     */
    LayerNorm(int feature_dim, float eps = 1e-5f);

    /**
     * @brief Performs the forward pass of the Layer Normalization.
     *
     * Normalizes the input tensor `x` across its last dimension, then scales
     * by `gain` and shifts by `bias`.
     *
     * @param ctx The graph context for tracing operations.
     * @param x A Variable containing the input tensor to be normalized.
     * @return A Variable containing the normalized, scaled, and shifted output tensor.
     */
    Variable* forward(GraphContext& ctx, Variable* x);

    /**
     * @brief Returns a vector of learnable parameters for this layer.
     *
     * @return A `std::vector` containing pointers to the `gain` and `bias` Variables.
     */
    std::vector<Variable*> parameters();
};
} // namespace sylvan::core
