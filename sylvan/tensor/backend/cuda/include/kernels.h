// sylvan/tensor/backend/cuda/include/kernels.h
//
// Declares the CUDA kernel launch functions used by the Sylvan tensor library.
// These functions are the low-level entry points for GPU computations, covering
// element-wise operations, reductions, neural network layers, and more.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint> // For int64_t

//===----------------------------------------------------------------------===//
// Element-wise & Initialization Kernels
//===----------------------------------------------------------------------===//

/**
 * @brief Launches an element-wise addition kernel (a += b).
 * @param a The tensor to be modified in-place.
 * @param b The tensor to add.
 * @param n The number of elements in the tensors.
 */
void launch_elementwise_add_kernel(float* a, const float* b, size_t n);

/**
 * @brief Launches a kernel to fill a tensor with a constant value.
 * @param data The tensor data to fill.
 * @param value The scalar value to fill with.
 * @param n The number of elements in the tensor.
 */
void launch_fill_kernel(float *data, float value, size_t n);

/**
 * @brief Launches a kernel to scale and shift a tensor in-place (data = data * scale + bias).
 * @param data The tensor data to modify.
 * @param scale The scaling factor.
 * @param bias The additive bias.
 * @param n The number of elements in the tensor.
 */
void launch_scale_kernel(float *data, float scale, float bias, size_t n);

//===----------------------------------------------------------------------===//
// Broadcasting Kernels
//===----------------------------------------------------------------------===//

/**
 * @brief Launches a broadcasted element-wise addition kernel (out = a + b).
 * @param out The output tensor.
 * @param out_numel The number of elements in the output tensor.
 * @param a The first input tensor.
 * @param b The second input tensor.
 * @param d_out_shape Device pointer to the shape of the output tensor.
 * @param ndim The number of dimensions.
 * @param d_a_strides Device pointer to the strides of tensor `a`.
 * @param d_b_strides Device pointer to the strides of tensor `b`.
 */
void launch_broadcast_add_kernel(float* out, size_t out_numel,
                                 const float* a, const float* b,
                                 const int64_t* d_out_shape, int ndim,
                                 const int64_t* d_a_strides, const int64_t* d_b_strides);

/**
 * @brief Launches a broadcasted in-place element-wise addition kernel (a += b).
 * @param a The tensor to be modified in-place.
 * @param b The tensor to add, which will be broadcast to `a`'s shape.
 * @param a_numel The number of elements in tensor `a`.
 * @param ndim The number of dimensions of tensor `a`.
 * @param d_a_shape Device pointer to the shape of `a`.
 * @param d_a_strides Device pointer to the strides of `a`.
 * @param d_b_strides Device pointer to the strides of `b` (adjusted for broadcasting).
 */
void launch_broadcast_add_inplace_kernel(
    float* a, const float* b, size_t a_numel, int ndim,
    const int64_t* d_a_shape,
    const int64_t* d_a_strides,
    const int64_t* d_b_strides
);

/**
 * @brief Launches a broadcasted element-wise multiplication kernel (out = a * b).
 * @param out The output tensor.
 * @param out_numel The number of elements in the output tensor.
 * @param a The first input tensor.
 * @param b The second input tensor.
 * @param d_out_shape Device pointer to the shape of the output tensor.
 * @param ndim The number of dimensions.
 * @param d_a_strides Device pointer to the strides of tensor `a`.
 * @param d_b_strides Device pointer to the strides of tensor `b`.
 */
void launch_broadcast_mul_kernel(float *out, size_t out_numel, const float *a,
                                 const float *b, const int64_t *d_out_shape,
                                 int ndim, const int64_t *d_a_strides,
                                 const int64_t *d_b_strides);

//===----------------------------------------------------------------------===//
// Reduction Kernels
//===----------------------------------------------------------------------===//

template <int TILE_DIM>
void launch_sum_columns_kernel_template(float *out, const float *in, int rows,
                                        int cols);

/**
 * @brief Launches a high-performance parallel reduction sum on a large array.
 * @param out Output buffer. Can be a buffer of partial sums or the final scalar result.
 * @param in Input buffer to be summed.
 * @param n Number of elements in the input buffer.
 */
void launch_global_sum_kernel(float* out, const float* in, size_t n);

/**
 * @brief Launches a kernel to sum a tensor along a specified axis.
 * @param out The output tensor data.
 * @param in The input tensor data.
 * @param outer_dim Product of dimensions before the reduction axis.
 * @param reduce_dim Size of the dimension to reduce.
 * @param inner_dim Product of dimensions after the reduction axis.
 */
void launch_sum_axis_kernel(
    void* out,
    const void* in,
    int outer_dim,
    int reduce_dim,
    int inner_dim
);

/**
 * @brief Launches a kernel to sum an input tensor to match an output shape (for broadcasting backward).
 * @param out The output tensor (gradient accumulator).
 * @param in The input tensor (incoming gradient).
 * @param out_numel Number of elements in the output tensor.
 * @param in_ndim Number of dimensions of the input tensor.
 * @param d_in_shape Device pointer to the shape of the input tensor.
 * @param d_in_strides Device pointer to the strides of the input tensor.
 * @param d_out_strides_for_in Device pointer to the output strides, adjusted for input indexing.
 * @param d_is_dim_reduced Device boolean array indicating which dimensions were broadcasted.
 */
void launch_sum_to_kernel(float* out, const float* in,
                          size_t out_numel,
                          int in_ndim,
                          const int64_t* d_in_shape,
                          const int64_t* d_in_strides,
                          const int64_t* d_out_strides_for_in,
                          const bool* d_is_dim_reduced);

//===----------------------------------------------------------------------===//
// Data Manipulation Kernels
//===----------------------------------------------------------------------===//

/**
 * @brief Launches a kernel to transpose a 2D matrix.
 * @param out The output tensor data.
 * @param in The input tensor data.
 * @param rows The number of rows in the input matrix.
 * @param cols The number of columns in the input matrix.
 */
void launch_transpose_kernel(float *out, const float *in, int rows, int cols);

/**
 * @brief Launches a kernel to permute the dimensions of an N-dimensional tensor.
 * @param out The output tensor.
 * @param in The input tensor.
 * @param d_in_strides Device pointer to the strides of the input tensor.
 * @param d_out_strides Device pointer to the strides of the output tensor.
 * @param d_inv_perm Device pointer to the inverse permutation map.
 * @param ndim The number of dimensions.
 * @param n The total number of elements.
 */
void launch_transpose_permute_kernel(float* out, const float* in,
                                     const int64_t* d_in_strides,
                                     const int64_t* d_out_strides,
                                     const int* d_inv_perm,
                                     int ndim, size_t n);

/**
 * @brief Launches a kernel to slice a sub-tensor from an N-dimensional tensor.
 * @param out The output tensor.
 * @param in The input tensor.
 * @param d_in_shape Device pointer to the shape of the input tensor.
 * @param d_out_shape Device pointer to the shape of the output tensor.
 * @param d_offsets Device pointer to the start offsets for each dimension.
 * @param n The number of elements in the output tensor.
 * @param ndim The number of dimensions.
 */
void launch_slice_forward_kernel(float* out, const float* in, const int64_t* d_in_shape, const int64_t* d_out_shape, const int64_t* d_offsets, size_t n, int ndim);

/**
 * @brief Launches a kernel for the backward pass of slice (scatter add).
 * @param grad_out The gradient tensor of the original (pre-slice) tensor.
 * @param grad_in The gradient of the sliced (output) tensor.
 * @param d_out_shape Device pointer to the shape of the sliced tensor.
 * @param d_in_shape Device pointer to the shape of the original tensor.
 * @param d_offsets Device pointer to the start offsets for each dimension.
 * @param n The number of elements in the sliced tensor.
 * @param ndim The number of dimensions.
 */
void launch_slice_backward_kernel(float* grad_out, const float* grad_in, const int64_t* d_out_shape, const int64_t* d_in_shape, const int64_t* d_offsets, size_t n, int ndim);

/**
 * @brief Launches a kernel to scatter-add an input tensor into an output tensor at given offsets.
 *
 * This operation is equivalent to `out[offsets[0]:offsets[0]+in_shape[0], ...] += in`.
 * It is commonly used as the backward pass for a slice operation.
 *
 * @param out The output tensor to be modified in-place.
 * @param in The input tensor whose values will be scattered and added.
 * @param out_shape Device pointer to the shape of the output tensor.
 * @param in_shape Device pointer to the shape of the input tensor.
 * @param offsets Device pointer to the starting coordinates in `out` for the scatter operation.
 * @param ndim The number of dimensions.
 * @param n_in The number of elements in the input tensor.
 */
void launch_scatter_add_kernel(float* out, const float* in,
                               const int64_t* out_shape,
                               const int64_t* in_shape,
                               const int64_t* offsets,
                               int ndim, size_t n_in);

//===----------------------------------------------------------------------===//
// Neural Network Layer Kernels
//===----------------------------------------------------------------------===//

/**
 * @brief Launches the forward pass kernel for ReLU activation.
 * @param out The output tensor.
 * @param in The input tensor.
 * @param n The number of elements.
 */
void launch_relu_forward_kernel(float* out, const float* in, size_t n);

/**
 * @brief Launches the backward pass kernel for ReLU activation.
 * @param out_grad The output gradient tensor (gradient w.r.t. input).
 * @param in The original input tensor from the forward pass.
 * @param grad_in The incoming gradient from the next layer.
 * @param n The number of elements.
 */
void launch_relu_backward_kernel(float* out_grad, const float* in, const float* grad_in, size_t n);

/**
 * @brief Launches the forward pass kernel for the Embedding layer (gather).
 * @param out The output tensor of embedded vectors.
 * @param weights The embedding weight matrix.
 * @param ids The input tensor of integer IDs.
 * @param batch_size The batch size.
 * @param seq_len The sequence length.
 * @param d_model The embedding dimension.
 */
void launch_gather_forward_kernel(float* out, const float* weights, const int* ids, size_t batch_size, size_t seq_len, size_t d_model);

/**
 * @brief Launches the backward pass kernel for the Embedding layer (scatter-add).
 * @param weights_grad The gradient tensor for the embedding weights.
 * @param out_grad The incoming gradient from the next layer.
 * @param ids The original input tensor of integer IDs.
 * @param batch_size The batch size.
 * @param seq_len The sequence length.
 * @param d_model The embedding dimension.
 */
void launch_scatter_add_backward_kernel(float* weights_grad, const float* out_grad, const int* ids, size_t batch_size, size_t seq_len, size_t d_model);

/**
 * @brief Launches a numerically stable Softmax kernel.
 * @param out The output tensor of probabilities.
 * @param in The input logits tensor.
 * @param rows The number of rows (batch size * seq_len).
 * @param cols The number of columns (feature dimension).
 */
void launch_softmax_kernel(float* out, const float* in, int rows, int cols);

/**
 * @brief Launches a fused kernel for the forward pass of Layer Normalization.
 * @param out The output tensor.
 * @param mean Output buffer for row-wise means (for backward pass).
 * @param inv_std Output buffer for row-wise inverse standard deviations (for backward pass).
 * @param in The input tensor.
 * @param gain The learnable gain (gamma) parameter.
 * @param bias The learnable bias (beta) parameter.
 * @param rows The number of rows (batch size).
 * @param cols The number of columns (feature dimension).
 * @param eps A small value for numerical stability.
 */
void launch_fused_layer_norm_forward_kernel(
    float* out, float* mean, float* inv_std, const float* in,
    const float* gain, const float* bias, int rows, int cols, float eps
);

/**
 * @brief Launches a fused kernel for the backward pass of Layer Normalization.
 * @param dx The output gradient w.r.t. the input `in`.
 * @param d_gain The output gradient w.r.t. the `gain` parameter.
 * @param d_bias The output gradient w.r.t. the `bias` parameter.
 * @param temp_d_gain Temporary buffer for parallel reduction of d_gain.
 * @param temp_d_bias Temporary buffer for parallel reduction of d_bias.
 * @param d_out The incoming gradient from the next layer.
 * @param in The original input tensor from the forward pass.
 * @param gain The original gain parameter.
 * @param mean The mean computed during the forward pass.
 * @param inv_std The inverse standard deviation computed during the forward pass.
 * @param rows The number of rows.
 * @param cols The number of columns.
 */
void launch_fused_layer_norm_backward_kernel(
    float* dx, float* d_gain, float* d_bias,
    float* temp_d_gain, float* temp_d_bias,
    const float* d_out, const float* in, const float* gain,
    const float* mean, const float* inv_std,
    int rows, int cols
);

/**
 * @brief Launches a fused kernel for the forward pass of Softmax + Cross-Entropy loss.
 * @param loss_per_row Output buffer of size `rows` to store the loss for each sample.
 * @param softmax_out Output buffer for softmax probabilities (needed for backward pass).
 * @param logits Input logits tensor data.
 * @param targets Input target class IDs (Int32).
 * @param rows Number of rows (batch size).
 * @param cols Number of columns (vocabulary size).
 */
void launch_fused_softmax_cross_entropy_forward_kernel(
    float* loss_per_row,
    float* softmax_out,
    const float* logits,
    const int* targets,
    int rows,
    int cols
);

/**
 * @brief Launches a fused kernel for the Adam optimization algorithm update step.
 * @param param_data Parameter tensor to be updated in-place.
 * @param grad_data Gradient of the parameter.
 * @param m_data First moment vector (updated in-place).
 * @param v_data Second moment vector (updated in-place).
 * @param n Number of elements in the tensors.
 * @param lr Learning rate.
 * @param beta1 Exponential decay rate for the first moment estimates.
 * @param beta2 Exponential decay rate for the second-moment estimates.
 * @param eps Term added to the denominator to improve numerical stability.
 * @param t Current timestep for bias correction.
 */
void launch_adam_update_kernel(
    float* param_data, const float* grad_data,
    float* m_data, float* v_data,
    size_t n, float lr, float beta1, float beta2, float eps, int t);

//===----------------------------------------------------------------------===//
// Utility & Masking Kernels
//===----------------------------------------------------------------------===//

/**
 * @brief Launches a kernel to create a one-hot encoding tensor.
 * @param out The output one-hot tensor.
 * @param ids The input tensor of integer IDs.
 * @param rows The number of samples.
 * @param vocab_size The depth of the one-hot encoding (number of classes).
 */
void launch_one_hot_kernel(float* out, const int* ids, int rows, int vocab_size);

/**
 * @brief Launches a kernel to create a causal (look-ahead) mask.
 * @param mask The output mask tensor to be filled.
 * @param seq_len The sequence length of the square mask.
 */
void launch_create_causal_mask_kernel(float* mask, int seq_len);

/**
 * @brief Launches a kernel to create a padding mask from token IDs.
 * @param mask Output mask tensor.
 * @param ids Input tensor of token IDs (Int32).
 * @param n Total number of elements.
 * @param pad_id The token ID to be masked.
 */
void launch_create_padding_mask_kernel(float* mask, const int* ids, size_t n, int pad_id);
