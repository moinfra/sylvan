// sylvan/core/src/optimizer.cc
//
// Implements the Stochastic Gradient Descent (SGD) and Adam optimization
// algorithms. These optimizers are used to update model parameters based on
// computed gradients.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/optimizer.h"
#include "sylvan/tensor/operators.h"
#include <cmath>

namespace sylvan::core {

SGD::SGD(std::vector<Variable *> params, float lr, float clip_grad_norm)
    : params_(params), lr_(lr), clip_grad_norm_(clip_grad_norm) {}

void SGD::step() {
  // -------- Optional: Apply Gradient Clipping --------
  if (clip_grad_norm_ > 0) {
    float total_norm = 0.0f;
    // Calculate the squared L2 norm of all gradients combined.
    for (auto *param : params_) {
      if (param->grad.numel() > 0) {
        total_norm += std::pow(calculate_norm(param->grad), 2);
      }
    }
    total_norm = std::sqrt(total_norm);

    // If total norm exceeds the threshold, scale down all gradients.
    if (total_norm > clip_grad_norm_) {
      float clip_scale = clip_grad_norm_ / total_norm;
      // Create a scalar tensor for efficient scaling on device.
      auto scale_tensor = tensor::ops::from_host(std::vector{clip_scale}, {1});
      for (auto *param : params_) {
        if (param->grad.numel() > 0) {
          // Replace the original gradient with its scaled version.
          auto scaled_grad = tensor::ops::mul(param->grad, scale_tensor);
          param->grad = std::move(scaled_grad);
        }
      }
    }
  }

  // -------- Perform Parameter Update --------
  // Create a scalar tensor for -learning_rate for efficient in-place updates.
  auto lr_tensor = tensor::ops::from_host(std::vector{-lr_}, {1});
  for (auto *param : params_) {
    // Only update parameters that have accumulated gradients.
    if (param->grad.numel() > 0) {
      auto update =
          tensor::ops::mul(param->grad, lr_tensor); // Calculate -lr * grad
      tensor::ops::add_(param->data,
                        update); // In-place update: param.data -= lr * grad
    }
  }
}

void SGD::zero_grad() {
  for (auto *param : params_) {
    // Fill the gradient tensor with zeros.
    // This implicitly handles cases where grad might not be allocated yet for
    // non-requiring-grad parameters.
    if (param->grad.numel() > 0) { // Only zero out if allocated
      tensor::ops::fill_(param->grad, 0.0f);
    }
  }
}

Adam::Adam(std::vector<Variable *> params, float lr, float beta1, float beta2,
           float eps)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {
  // Initialize first and second moment vectors for each parameter to zeros.
  for (const auto *param : params_) {
    if (param->data.numel() > 0) {
      // m (first moment) has the same shape and dtype as the parameter data.
      m_.emplace_back(param->data.shape(), param->data.dtype());
      tensor::ops::fill_(m_.back(), 0.0f);

      // v (second moment) also has the same shape and dtype.
      v_.emplace_back(param->data.shape(), param->data.dtype());
      tensor::ops::fill_(v_.back(), 0.0f);
    } else {
      // Handle empty tensors, though parameters should generally not be empty.
      m_.emplace_back();
      v_.emplace_back();
    }
  }
}

void Adam::step() {
  t_++;

  for (size_t i = 0; i < params_.size(); ++i) {
    Variable *param = params_[i];

    // Skip update if parameter has no gradient or is empty.
    if (param->grad.numel() == 0) {
      continue;
    }

    // Call the fused CUDA kernel to perform the Adam update for this parameter.
    // This combines moment updates, bias correction, and parameter update for
    // efficiency.
    tensor::ops::adam_update(
        param->data.mutable_data<float>(), // Parameter data (in-place updated)
        param->grad.data<float>(),         // Gradient data
        m_[i].mutable_data<float>(),       // First moment (in-place updated)
        v_[i].mutable_data<float>(),       // Second moment (in-place updated)
        param->data.numel(),               // Number of elements in parameter
        lr_, beta1_, beta2_, eps_, t_);    // Adam hyperparameters and timestep
  }
}

void Adam::zero_grad() {
  for (auto *param : params_) {
    if (param->grad.numel() > 0) { // Only zero out if allocated
      tensor::ops::fill_(param->grad, 0.0f);
    }
  }
}

} // namespace sylvan::core
