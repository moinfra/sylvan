// sylvan/tensor/ops.h
//
// Declares the public API for tensor operations in the Sylvan framework.
// This includes creation, manipulation, mathematical, and neural network-specific
// operations. These functions typically operate on `sylvan::tensor::Tensor` objects
// and are implemented with CUDA for GPU acceleration.
//
// Author: Sylvan Team
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "sylvan/tensor/tensor.h"
#include <string>
#include <vector>


namespace sylvan::tensor::ops {
/**
 * @brief Provides the primary functional API for tensor manipulation.
 *
 * This namespace contains a comprehensive set of functions for creating,
 * transforming, and performing mathematical computations on Tensors.
 * Operations are categorized into in-place modifications, functions that
 * return new tensors, and specialized neural network layers.
 */

/**
 * @brief Creates a Float32 Tensor on the device from host data.
 * @param data A vector of floats on the host.
 * @param shape The desired shape of the new tensor.
 * @return A new Tensor on the device containing the copied data.
 */
Tensor from_host(const std::vector<float> &data, const Shape &shape);

/**
 * @brief Creates an Int32 Tensor on the device from host data.
 * @param data A vector of integers on the host.
 * @param shape The desired shape of the new tensor.
 * @return A new Tensor on the device containing the copied data.
 */
Tensor from_host(const std::vector<int> &data, const Shape &shape);

/**
 * @brief Clones tensor data from the device to a host vector.
 * @tparam T The data type of the elements (e.g., float, int).
 * @param t The tensor on the device to clone.
 * @return A std::vector on the host containing the tensor's data.
 */
template <typename T> std::vector<T> clone_to_host(const Tensor &t);

// -------- In-place Operations --------
// In-place operations modify the tensor directly and are suffixed with `_`.

/**
 * @brief [In-place] Fills a tensor with a specified constant value.
 * @param t The tensor to be filled. This tensor is modified directly.
 * @param value The scalar float value to fill the tensor with.
 */
void fill_(Tensor &t, float value);

/**
 * @brief [In-place] Adds another tensor element-wise. Modifies `t`.
 * @param t The tensor to be modified (t = t + other).
 * @param other The tensor to add. Must be broadcast-compatible with `t`.
 */
void add_(Tensor &t, const Tensor &other);

/**
 * @brief [In-place] Accumulates gradients from a slice into the full gradient tensor.
 *
 * This is the backward operation for `slice`. It adds the `sliced_grad`
 * into the `full_grad_tensor` at the specified offsets.
 *
 * @param full_grad_tensor The tensor holding the gradients for the original, unsliced tensor. It is modified in-place.
 * @param sliced_grad The gradient corresponding to the sliced tensor.
 * @param ranges The same ranges used in the forward slice operation, defining where to add the gradients.
 */
void slice_backward_(Tensor &full_grad_tensor, const Tensor &sliced_grad,
                     const std::vector<std::pair<int64_t, int64_t>> &ranges);

/**
 * @brief [In-place] Fills a tensor with random numbers from a uniform distribution.
 * @param t The tensor to be filled. This tensor is modified directly.
 * @param from The lower bound of the uniform distribution (inclusive).
 * @param to The upper bound of the uniform distribution (exclusive).
 */
void uniform_(Tensor &t, float from, float to);

// -------- Operations Returning a New Tensor --------

/**
 * @brief Performs element-wise addition, C = A + B. Supports broadcasting.
 * @param a The first input tensor.
 * @param b The second input tensor.
 * @return A new tensor containing the result of the addition.
 */
Tensor add(const Tensor &a, const Tensor &b);

/**
 * @brief Performs element-wise subtraction, C = A - B. Supports broadcasting.
 * @param a The tensor to subtract from.
 * @param b The tensor to subtract.
 * @return A new tensor containing the result of the subtraction.
 */
Tensor sub(const Tensor &a, const Tensor &b);

/**
 * @brief Performs element-wise multiplication, C = A * B. Supports broadcasting.
 * @param a The first input tensor.
 * @param b The second input tensor.
 * @return A new tensor containing the result of the multiplication.
 */
Tensor mul(const Tensor &a, const Tensor &b);

/**
 * @brief Performs matrix multiplication, C = A @ B.
 * @note Currently only supports 2D tensors.
 * @param a The first input matrix (2D Tensor).
 * @param b The second input matrix (2D Tensor).
 * @return A new tensor containing the matrix multiplication result.
 */
Tensor matmul(const Tensor &a, const Tensor &b);

/**
 * @brief Transposes a 2D tensor.
 * @param t The 2D tensor to transpose.
 * @return A new, transposed tensor.
 */
Tensor transpose(const Tensor &t);

/**
 * @brief Reduces a tensor by summing elements along a given axis.
 * @param input The tensor to reduce.
 * @param axis The axis along which to sum.
 * @param keep_dims If true, the reduced axis is kept with size 1.
 * @return A new tensor with the specified axis reduced.
 */
Tensor sum(const Tensor &input, int axis, bool keep_dims);

/**
 * @brief Computes the sum of all elements in a tensor.
 * @param t The input tensor.
 * @return A new scalar tensor (shape {1}) containing the sum.
 */
Tensor sum(const Tensor &t);

/**
 * @brief Computes the mean of all elements in a tensor.
 * @param t The input tensor.
 * @return A new scalar tensor (shape {1}) containing the mean.
 */
Tensor mean(const Tensor &t);

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function element-wise.
 * @param t The input tensor.
 * @return A new tensor with the ReLU function applied.
 */
Tensor relu(const Tensor &t);

/**
 * @brief Computes the gradient of the ReLU function.
 * @param in The original input to the forward ReLU function.
 * @param grad The upstream gradient from the next layer.
 * @return A new tensor containing the computed gradients.
 */
Tensor relu_backward(const Tensor &in, const Tensor &grad);

/**
 * @brief Applies the Softmax function along the last dimension of the tensor.
 * @param t The input tensor (logits).
 * @return A new tensor with probabilities.
 */
Tensor softmax(const Tensor &t);

/**
 * @brief Converts a tensor of integer IDs into one-hot encoded vectors.
 * @param ids A 1D tensor of class indices (Int32).
 * @param vocab_size The total number of classes (the depth of the one-hot vector).
 * @return A new 2D tensor of shape [num_ids, vocab_size].
 */
Tensor one_hot(const Tensor &ids, int vocab_size);

/**
 * @brief Performs Layer Normalization on the input tensor.
 * @param x The input tensor to normalize.
 * @param gain The learnable gain parameter (gamma).
 * @param bias The learnable bias parameter (beta).
 * @param eps A small value added to the variance for numerical stability.
 * @return A tuple containing: {output, mean, inv_std}. The mean and inv_std
 *         are returned for use in the backward pass.
 */
std::tuple<Tensor, Tensor, Tensor> layer_norm_forward(const Tensor &x,
                                                      const Tensor &gain,
                                                      const Tensor &bias,
                                                      float eps);

/**
 * @brief Computes the gradients for Layer Normalization.
 * @param d_out The upstream gradient from the subsequent layer.
 * @param in The original input to the forward function.
 * @param gain The gain parameter used in the forward pass.
 * @param mean The mean calculated during the forward pass.
 * @param inv_std The inverse standard deviation from the forward pass.
 * @return A tuple containing the gradients: {dx, d_gain, d_bias}.
 */
std::tuple<Tensor, Tensor, Tensor>
layer_norm_backward(const Tensor &d_out, const Tensor &in, const Tensor &gain,
                    const Tensor &mean, const Tensor &inv_std);

/**
 * @brief Performs an embedding lookup.
 * @param weights The embedding matrix, shape [vocab_size, d_model].
 * @param ids A tensor of token indices to look up, shape [batch_size, seq_len].
 * @return A new tensor with the corresponding embedding vectors, shape [batch_size, seq_len, d_model].
 */
Tensor embedding_forward(const Tensor &weights, const Tensor &ids);

/**
 * @brief Computes the gradient for the embedding weights (scatter-add).
 * @param grad_out The upstream gradient from the embedding output.
 * @param ids The original token indices used in the forward pass.
 * @param weights_shape The shape of the original embedding weight matrix.
 * @return A new tensor containing the gradients for the embedding weights.
 */
Tensor embedding_backward(const Tensor &grad_out, const Tensor &ids,
                          const Shape &weights_shape);

/**
 * @brief [In-place] Fills a 2D tensor to be a causal (lower-triangular) mask.
 * @param mask The tensor to be modified into a causal mask.
 */
void create_causal_mask_(Tensor &mask);

/**
 * @brief Reduces a matrix by summing its columns.
 * @tparam TILE_DIM The tile size for the underlying CUDA kernel.
 * @param in The input 2D tensor.
 * @return A new 1D tensor containing the sum of each column.
 */
template <int TILE_DIM = 32>
Tensor sum_columns(const Tensor &in);

/**
 * @brief Reshapes a tensor to a new shape without changing its data.
 * @param t The input tensor.
 * @param new_shape The target shape. Must have the same number of elements as the original.
 * @return A new tensor with the specified shape, sharing data with the original.
 */
Tensor reshape(const Tensor &t, const Shape &new_shape);

/**
 * @brief Permutes the dimensions of a tensor.
 * @param t The input tensor.
 * @param perm A vector specifying the new order of dimensions.
 * @return A new tensor with permuted dimensions.
 */
Tensor transpose(const Tensor &t, const std::vector<int> &perm);

/**
 * @brief Scales a tensor by a constant factor.
 * @param t The input tensor.
 * @param factor The scalar value to multiply each element by.
 * @return A new tensor containing the scaled values.
 */
Tensor scale(const Tensor &t, float factor);

/**
 * @brief Extracts a slice from a tensor along specified ranges.
 * @param t The input tensor.
 * @param ranges A vector of pairs, where each pair {offset, length} defines the slice for a dimension.
 * @return A new tensor representing the slice.
 */
Tensor slice(const Tensor &t,
             const std::vector<std::pair<int64_t, int64_t>> &ranges);

/**
 * @brief Reduces an input tensor by summing elements to match a target shape.
 *
 * This is the backward operation for broadcasting. It sums the `input` tensor
 * along the dimensions that were broadcasted to produce the `target_shape`.
 *
 * @param input The larger tensor (e.g., a gradient) to be reduced.
 * @param target_shape The smaller shape to reduce to.
 * @return A new tensor with the `target_shape`.
 */
Tensor sum_to(const Tensor &input, const Shape &target_shape);

/**
 * @brief Computes cross-entropy loss using a fused kernel for efficiency.
 *
 * This operation combines Softmax and Negative Log-Likelihood Loss, which is
 * more numerically stable and faster than performing the operations separately.
 *
 * @param logits The raw output from the model, shape [B, V].
 * @param targets The correct class indices, shape [B], dtype Int32.
 * @return A tuple containing:
 *         1. The final scalar loss Tensor.
 *         2. The softmax probabilities Tensor (for backward pass), shape [B, V].
 */
std::tuple<Tensor, Tensor> cross_entropy(const Tensor &logits,
                                         const Tensor &targets);

/**
 * @brief Creates a padding mask from a tensor of token IDs.
 *
 * The resulting mask is shaped for broadcasting in attention mechanisms.
 * A value of 1.0 indicates a valid token, and 0.0 indicates a padding token.
 *
 * @param ids A 2D tensor of token IDs, shape [B, L], dtype Int32.
 * @param pad_id The integer ID representing padding.
 * @return A float tensor of shape [B, 1, 1, L].
 */
Tensor create_padding_mask(const Tensor &ids, int pad_id);

/**
 * @brief Creates a causal (look-ahead) mask for decoder self-attention.
 *
 * The mask is shaped for broadcasting. A value of 1.0 allows attention,
 * while 0.0 prevents it.
 *
 * @param seq_len The length of the sequence.
 * @return A float tensor of shape [1, 1, L, L] with a lower-triangular pattern.
 */
Tensor create_causal_mask(int seq_len);

/**
 * @brief Performs a single Adam optimizer update step on raw device pointers.
 * @note This is a low-level helper function, typically called by an Optimizer class.
 *
 * @param param_data Pointer to the parameters to be updated.
 * @param grad_data Pointer to the gradients.
 * @param m_data Pointer to the first moment vector (momentum).
 * @param v_data Pointer to the second moment vector (RMSprop).
 * @param n Number of elements in the tensors.
 * @param lr Learning rate.
 * @param beta1 Exponential decay rate for the first moment.
 * @param beta2 Exponential decay rate for the second moment.
 * @param eps Term added for numerical stability.
 * @param t The current timestep.
 */
void adam_update(float *param_data, const float *grad_data,
                               float *m_data, float *v_data, size_t n, float lr,
                               float beta1, float beta2, float eps, int t);

} // namespace sylvan::tensor::ops
