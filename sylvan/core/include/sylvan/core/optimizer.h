// sylvan/core/optimizer.h
//
// Defines various optimization algorithms (e.g., SGD, Adam) used to update
// the parameters of a neural network based on computed gradients.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "sylvan/core/autograd.h"
#include <cmath>
#include <sylvan/tensor/operators.h>
#include <vector>

/**
 * @brief Calculates the L2 norm (Euclidean norm) of a given tensor.
 *
 * The L2 norm is calculated as the square root of the sum of the squares of
 * all elements in the tensor. It is used for operations like gradient clipping.
 *
 * @param t The input tensor.
 * @return The L2 norm of the tensor. Returns 0.0f if the tensor has no
 * elements.
 */
inline float calculate_norm(const sylvan::tensor::Tensor &t) {
  if (t.numel() == 0)
    return 0.0f;
  auto t_squared = sylvan::tensor::ops::mul(t, t);
  auto sum_of_squares = sylvan::tensor::ops::sum(t_squared);
  auto host_sum = sylvan::tensor::ops::clone_to_host<float>(sum_of_squares);
  return sqrt(host_sum[0]);
}

namespace sylvan::core {

/**
 * @brief Implements the Stochastic Gradient Descent (SGD) optimizer.
 *
 * SGD updates parameters by moving them in the direction opposite to their
 * gradients, scaled by a learning rate. It optionally supports global gradient
 * clipping by norm to prevent exploding gradients.
 */
class SGD {
public:
  /**
   * @brief Constructs an SGD optimizer.
   *
   * @param params A vector of `Variable` pointers representing the model's
   * parameters to be optimized.
   * @param lr The learning rate, a positive scalar controlling the step size of
   * parameter updates.
   * @param clip_grad_norm The maximum allowed L2 norm for the combined
   * gradients. If positive, gradients will be scaled down if their total L2
   * norm exceeds this value. If -1.0f or non-positive, no clipping is applied.
   */
  SGD(std::vector<Variable *> params, float lr = 0.01,
      float clip_grad_norm = -1.0f);

  /**
   * @brief Performs a single optimization step.
   *
   * This method iterates through all registered parameters, applies optional
   * gradient clipping (if `clip_grad_norm` is positive), and then updates
   * each parameter's data using its gradient and the learning rate.
   */
  void step();

  /**
   * @brief Sets the gradients of all optimized parameters to zero.
   *
   * This is typically called at the beginning of each training iteration
   * to clear gradients accumulated from the previous backward pass.
   */
  void zero_grad();

private:
  std::vector<Variable *>
      params_; ///< Vector of pointers to the parameters to be optimized.
  float lr_;   ///< The learning rate for parameter updates.
  float clip_grad_norm_; ///< Threshold for global gradient norm clipping.
};

/**
 * @brief Implements the Adam (Adaptive Moment Estimation) optimizer.
 *
 * Adam is an adaptive learning rate optimization algorithm that computes
 * individual adaptive learning rates for different parameters from estimates
 * of first and second moments of the gradients.
 */
class Adam {
public:
  /**
   * @brief Constructs an Adam optimizer.
   *
   * @param params A vector of `Variable` pointers representing the model's
   * parameters to be optimized.
   * @param lr The learning rate.
   * @param beta1 The exponential decay rate for the first moment estimates
   * (momentum).
   * @param beta2 The exponential decay rate for the second moment estimates
   * (RMSprop).
   * @param eps A small constant for numerical stability to prevent division by
   * zero.
   */
  Adam(std::vector<Variable *> params, float lr = 0.001f, float beta1 = 0.9f,
       float beta2 = 0.999f, float eps = 1e-8f);

  /**
   * @brief Performs a single optimization step.
   *
   * This method updates each parameter using the Adam algorithm, which
   * involves updating biased first (momentum) and second (RMSprop) moment
   * estimates, correcting for their bias, and then applying updates.
   */
  void step();

  /**
   * @brief Sets the gradients of all optimized parameters to zero.
   *
   * This is typically called at the beginning of each training iteration
   * to clear gradients accumulated from the previous backward pass.
   */
  void zero_grad();

private:
  std::vector<Variable *>
      params_;  ///< Vector of pointers to the parameters to be optimized.
  float lr_;    ///< Learning rate.
  float beta1_; ///< Decay rate for first moment estimates.
  float beta2_; ///< Decay rate for second moment estimates.
  float eps_;   ///< Small constant for numerical stability.
  int t_;       ///< Timestep counter, increments with each `step()` call.

  // Using `tensor::Tensor` for moments to allow GPU-based accumulation.
  std::vector<tensor::Tensor>
      m_; ///< First moment vector estimates for each parameter.
  std::vector<tensor::Tensor>
      v_; ///< Second moment vector estimates for each parameter.
};

} // namespace sylvan::core
