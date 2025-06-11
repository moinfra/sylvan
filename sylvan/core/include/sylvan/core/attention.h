#pragma once
#include "sylvan/core/autograd.h"
#include "sylvan/core/layers/linear.h"

namespace sylvan::core {

/**
 * @brief Computes the scaled dot-product attention.
 *
 * This is the fundamental attention mechanism: Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V.
 * It is used within MultiHeadAttention and handles optional masking.
 *
 * @param ctx The GraphContext for the current computation pass.
 * @param q The query tensor.
 * @param k The key tensor.
 * @param v The value tensor.
 * @param mask An optional mask tensor. If provided, elements in the attention
 *             scores corresponding to 0 in the mask will be set to a large negative
 *             value before softmax (e.g., -1e9 for causal masking or padding masking).
 *             If nullptr, no masking is applied.
 * @return A Variable representing the attention output.
 */
Variable* scaled_dot_product_attention(GraphContext& ctx, Variable* q, Variable* k, Variable* v, Variable* mask = nullptr);

/**
 * @brief Implements Multi-Head Attention (MHA) mechanism.
 *
 * MHA processes input by linearly projecting queries, keys, and values into
 * multiple "heads", performing scaled dot-product attention independently on
 * each head, and then concatenating and linearly transforming the results.
 * This allows the model to jointly attend to information from different
 * representation subspaces at different positions.
 */
struct MultiHeadAttention {
    int d_model;   ///< Dimensionality of the input and output features.
    int num_heads; ///< Number of attention heads.

    // Linear projection layers for query, key, value, and output.
    Linear wq, wk, wv, wo;

    /**
     * @brief Constructs a MultiHeadAttention module.
     * @param d_model Dimensionality of the input and output features.
     * @param num_heads Number of attention heads. `d_model` must be divisible by `num_heads`.
     */
    MultiHeadAttention(int d_model, int num_heads);

    /**
     * @brief Performs the forward pass of Multi-Head Attention.
     *
     * @param ctx The GraphContext for the current computation pass.
     * @param q The query tensor.
     * @param k The key tensor.
     * @param v The value tensor.
     * @param mask An optional mask tensor applied to attention scores.
     * @return A Variable representing the output of the multi-head attention.
     */
    Variable* forward(GraphContext& ctx, Variable* q, Variable* k, Variable* v, Variable* mask = nullptr);

    /**
     * @brief Returns a vector of all trainable parameters in the module.
     * @return A vector of Variable pointers representing the module's parameters.
     */
    std::vector<Variable*> parameters();
};

} // namespace sylvan::core
