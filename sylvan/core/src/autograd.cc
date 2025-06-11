// sylvan/core/autograd.cc
//
// Implements the automatic differentiation (autograd) system for Sylvan.
// This includes the Variable class, backward pass logic, and a collection
// of autograd-enabled operations. Each operation records its inputs and
// a `BackwardStep` (lambda function) to compute gradients during `backward()`.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/autograd.h"
#include "sylvan/core/graph.h"
#include "sylvan/tensor/operators.h"
#include <cassert>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace sylvan::core {

namespace {

/**
 * @brief Recursively builds a topological sort of the computational graph.
 *
 * This function performs a depth-first search (DFS) starting from a given node
 * to collect all reachable preceding nodes in reverse topological order (i.e.,
 * a node appears before its dependencies). This order is crucial for the
 * backward pass to ensure gradients are computed for dependencies first.
 *
 * @param node The current Variable node to process.
 * @param visited An unordered set to keep track of visited nodes to prevent
 * infinite loops.
 * @param tape A vector to store the topologically sorted nodes (in reverse
 * order after DFS).
 */
void build_topological_sort(Variable *node,
                            std::unordered_set<Variable *> &visited,
                            std::vector<Variable *> &tape) {
  if (!node || visited.count(node) > 0) {
    return;
  }
  visited.insert(node);
  for (Variable *prev_node : node->prev_) {
    build_topological_sort(prev_node, visited, tape);
  }
  tape.push_back(node);
}

} // namespace

//===----------------------------------------------------------------------===//
// Variable Class Implementation
//===----------------------------------------------------------------------===//

Variable::Variable(tensor::Tensor &&data, bool requires_grad,
                   std::vector<Variable *> prev,
                   std::shared_ptr<BackwardStep> grad_fn, std::string op_name)
    : data(std::move(data)), prev_(prev), grad_fn(grad_fn), op_name(op_name),
      requires_grad(requires_grad) {

  // Validate data type if gradients are required.
  if (this->requires_grad) {
    if (this->data.dtype() != tensor::DType::Float32) {
      throw std::runtime_error("Variable with requires_grad=true must have "
                               "Float32 data for differentiation. Op: " +
                               op_name);
    }
  }
}

void Variable::backward() {
  if (!this->requires_grad) {
    throw std::runtime_error(
        "backward() can only be called on a Variable that requires gradients.");
  }
  // Initialize the gradient of the root node (this Variable) to ones.
  if (grad.numel() == 0) {
    grad = tensor::Tensor(this->data.shape(), tensor::DType::Float32);
    tensor::ops::fill_(this->grad, 1.0f);
  }

  // Build the topological order of the graph from the root.
  std::vector<Variable *> tape;
  std::unordered_set<Variable *> visited;
  build_topological_sort(this, visited, tape);

  // Iterate through the graph in reverse topological order to compute
  // gradients.
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    Variable *v = *it;
    // Initialize gradients for intermediate nodes if they require grad and are
    // not yet initialized.
    if (v->grad.numel() == 0 && v->requires_grad) {
      v->grad = tensor::Tensor(v->data.shape(), tensor::DType::Float32);
      tensor::ops::fill_(v->grad, 0.0f);
    }
    // Execute the backward step for the current node if defined.
    if (v->grad_fn) {
      v->grad_fn->pass(v);
    }
  }
}

/**
 * @brief Accumulates an incoming gradient into a target Variable's gradient
 * tensor.
 *
 * This function handles lazy initialization of the target gradient tensor
 * (if it doesn't exist yet) and performs shape-aware accumulation, including
 * summation for broadcasting cases.
 *
 * @param target_var A pointer to the Variable whose gradient needs to be
 * accumulated.
 * @param incoming_grad The gradient tensor to be added to `target_var->grad`.
 * @throws std::runtime_error if gradient accumulation is attempted on
 * non-Float32 tensors.
 * @throws std::runtime_error if shape mismatch occurs after `sum_to` for
 * broadcasting.
 */
void accumulate_grad(Variable *target_var,
                     const tensor::Tensor &incoming_grad) {
  // 1. Lazy initialization of the gradient tensor.
  // If the target gradient tensor is not yet allocated, create it with zeros.
  if (target_var->grad.numel() == 0) {
    // Safety check: only create gradient for variables that require it.
    if (!target_var->requires_grad)
      return;
    target_var->grad =
        tensor::Tensor(target_var->data.shape(), tensor::DType::Float32);
    assert(target_var->grad.shape() == target_var->data.shape() &&
           "Newly created grad shape mismatch with data shape.");
    tensor::ops::fill_(target_var->grad, 0.0f);
  }

  // 2. Accumulate gradient.
  tensor::Tensor &target_grad =
      target_var->grad; // Now target_grad is guaranteed to exist.

  if (target_grad.dtype() != tensor::DType::Float32 ||
      incoming_grad.dtype() != tensor::DType::Float32) {
    throw std::runtime_error("Gradient accumulation only supports Float32.");
  }
  assert(target_grad.shape() == target_var->data.shape() &&
         "accumulate_grad entry check: target_grad shape mismatch with data "
         "shape.");

  // If shapes match, perform direct element-wise addition.
  if (target_grad.shape() == incoming_grad.shape()) {
    tensor::ops::add_(target_grad, incoming_grad);
    return;
  }

  // If shapes differ, assume broadcasting occurred in the forward pass and
  // sum the incoming gradient to match the target's shape.
  tensor::Tensor grad_summed =
      tensor::ops::sum_to(incoming_grad, target_grad.shape());
  assert(grad_summed.shape() == target_grad.shape() &&
         "sum_to result shape does not match target_grad shape.");

  tensor::ops::add_(target_grad, grad_summed);
}

//===----------------------------------------------------------------------===//
// Autograd Wrappers for Ops (Refactored with the Core Rule)
//===----------------------------------------------------------------------===//

Variable *add(GraphContext &ctx, Variable *a, Variable *b) {
  bool output_requires_grad = a->requires_grad || b->requires_grad;
  tensor::Tensor out_data = tensor::ops::add(a->data, b->data);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [a, b](Variable *out_var) {
      if (a->requires_grad) {
        accumulate_grad(a, out_var->grad);
      }
      if (b->requires_grad) {
        accumulate_grad(b, out_var->grad);
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{a, b}, backward_step,
                             "add");
}

Variable *matmul(GraphContext &ctx, Variable *a, Variable *b) {
  bool output_requires_grad = a->requires_grad || b->requires_grad;
  tensor::Tensor out_data = tensor::ops::matmul(a->data, b->data);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [a, b](Variable *out_var) {
      if (a->requires_grad) {
        // Transpose b for dL/dA = dL/dC @ B.T
        std::vector<int> b_perm(b->data.ndim(), 0);
        std::iota(b_perm.begin(), b_perm.end(), 0);
        std::swap(b_perm[b->data.ndim() - 2], b_perm[b->data.ndim() - 1]);
        tensor::Tensor b_T = tensor::ops::transpose(b->data, b_perm);
        accumulate_grad(a, tensor::ops::matmul(out_var->grad, b_T));
      }
      if (b->requires_grad) {
        // Transpose a for dL/dB = A.T @ dL/dC
        std::vector<int> a_perm(a->data.ndim(), 0);
        std::iota(a_perm.begin(), a_perm.end(), 0);
        std::swap(a_perm[a->data.ndim() - 2], a_perm[a->data.ndim() - 1]);
        tensor::Tensor a_T = tensor::ops::transpose(a->data, a_perm);
        accumulate_grad(b, tensor::ops::matmul(a_T, out_var->grad));
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{a, b}, backward_step,
                             "matmul");
}

Variable *cross_entropy_loss(GraphContext &ctx, Variable *logits,
                             Variable *targets) {
  if (targets->data.dtype() != tensor::DType::Int32) {
    throw std::runtime_error("cross_entropy_loss requires Int32 targets.");
  }

  bool output_requires_grad = logits->requires_grad;

  if (targets->requires_grad) {
    std::cerr
        << "Warning: Targets for cross_entropy_loss have requires_grad=true. "
           "Gradients will not be computed for them."
        << std::endl;
  }

  // Forward pass: compute loss value and softmax probabilities
  auto [loss_val, softmax_probs] =
      tensor::ops::cross_entropy(logits->data, targets->data);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    // Capture softmax_probs by clone to avoid dangling reference if `out_data`
    // is moved.
    auto non_copyable_lambda =
        [logits, targets,
         sm_probs = softmax_probs.clone()](Variable * /*loss_var*/) {
          if (logits->requires_grad) {
            int vocab_size = logits->data.dim(logits->data.ndim() - 1);

            // Compute one-hot encoding of targets
            tensor::Tensor one_hot_targets =
                tensor::ops::one_hot(targets->data, vocab_size);

            // Calculate initial gradient: (softmax_probs - one_hot_targets)
            tensor::Tensor grad = tensor::ops::sub(sm_probs, one_hot_targets);
            // Scale gradient by batch size (average over batch)
            float scale = 1.0f / static_cast<float>(logits->data.dim(0));
            tensor::Tensor grad_scaled = tensor::ops::scale(grad, scale);
            accumulate_grad(logits, grad_scaled);
          }
          // Note: No gradient is ever passed to targets as they are typically
          // constant.
        };
    // Wrap the non-copyable lambda in a shared_ptr for std::function
    // compatibility.
    auto shared_lambda = std::make_shared<decltype(non_copyable_lambda)>(
        std::move(non_copyable_lambda));
    backward_step->pass = [shared_lambda](Variable *out_var) {
      (*shared_lambda)(out_var);
    };
  }

  return ctx.create_variable(std::move(loss_val), output_requires_grad,
                             std::vector<Variable *>{logits, targets},
                             backward_step, "cross_entropy_loss");
}

Variable *layer_norm(GraphContext &ctx, Variable *x, Variable *gain,
                     Variable *bias, float eps) {
  bool output_requires_grad =
      x->requires_grad || gain->requires_grad || bias->requires_grad;

  // Forward pass for layer normalization
  auto [out_data, mean, inv_std] =
      tensor::ops::layer_norm_forward(x->data, gain->data, bias->data, eps);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    // Capture necessary intermediate values (mean, inv_std) for backward pass.
    auto non_copyable_lambda = [x, gain, bias, mean_cap = std::move(mean),
                                inv_std_cap =
                                    std::move(inv_std)](Variable *out_var) {
      // Compute gradients for x, gain, and bias
      auto [dx, d_gain, d_bias] =
          tensor::ops::layer_norm_backward(out_var->grad, x->data, gain->data, mean_cap, inv_std_cap);
      if (x->requires_grad)
        accumulate_grad(x, dx);
      if (gain->requires_grad)
        accumulate_grad(gain, d_gain);
      if (bias->requires_grad)
        accumulate_grad(bias, d_bias);
    };
    // Wrap the non-copyable lambda in a shared_ptr.
    auto shared_lambda = std::make_shared<decltype(non_copyable_lambda)>(
        std::move(non_copyable_lambda));
    backward_step->pass = [shared_lambda](Variable *out_var) {
      (*shared_lambda)(out_var);
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{x, gain, bias},
                             backward_step, "layer_norm");
}

Variable *reshape(GraphContext &ctx, Variable *v,
                  const tensor::Shape &new_shape) {
  bool output_requires_grad = v->requires_grad;
  tensor::Tensor out_data = tensor::ops::reshape(v->data, new_shape);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [v](Variable *out_var) {
      // Reshape the incoming gradient back to the original shape of 'v'.
      if (v->requires_grad) {
        tensor::Tensor grad_reshaped =
            tensor::ops::reshape(out_var->grad, v->data.shape());
        accumulate_grad(v, grad_reshaped);
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{v}, backward_step,
                             "reshape");
}

Variable *scale(GraphContext &ctx, Variable *v, float factor) {
  bool output_requires_grad = v->requires_grad;
  tensor::Tensor out_data = tensor::ops::scale(v->data, factor);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [v, factor](Variable *out_var) {
      // Gradient for scaling is simply the incoming gradient scaled by the same
      // factor.
      if (v->requires_grad) {
        accumulate_grad(v, tensor::ops::scale(out_var->grad, factor));
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{v}, backward_step,
                             "scale");
}

Variable *slice(GraphContext &ctx, Variable *v,
                const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  bool output_requires_grad = v->requires_grad;
  tensor::Tensor out_data = tensor::ops::slice(v->data, ranges);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [v, ranges](Variable *out_var) {
      // `slice_backward_` accumulates the incoming gradient into the
      // appropriate slice of `v->grad`, which is expected to be
      // zero-initialized.
      if (v->requires_grad) {
        tensor::ops::slice_backward_(v->grad, out_var->grad, ranges);
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{v}, backward_step,
                             "slice");
}

Variable *transpose(GraphContext &ctx, Variable *v,
                    const std::vector<int> &perm) {
  bool output_requires_grad = v->requires_grad;
  tensor::Tensor out_data = tensor::ops::transpose(v->data, perm);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [v, perm](Variable *out_var) {
      // To get the gradient for the original tensor, transpose the incoming
      // gradient using the reverse permutation.
      if (v->requires_grad) {
        std::vector<int> reverse_perm(perm.size());
        for (size_t i = 0; i < perm.size(); ++i)
          reverse_perm[perm[i]] = i; // Calculate the inverse permutation
        accumulate_grad(v, tensor::ops::transpose(out_var->grad, reverse_perm));
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{v}, backward_step,
                             "transpose");
}

Variable *relu(GraphContext &ctx, Variable *x) {
  bool output_requires_grad = x->requires_grad;
  tensor::Tensor out_data = tensor::ops::relu(x->data);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    backward_step->pass = [x](Variable *out_var) {
      // ReLU backward: gradient is 1 where input > 0, else 0.
      if (x->requires_grad) {
        tensor::Tensor x_grad =
            tensor::ops::relu_backward(x->data, out_var->grad);
        accumulate_grad(x, x_grad);
      }
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{x}, backward_step, "relu");
}

Variable *softmax(GraphContext &ctx, Variable *x, int axis) {
  bool output_requires_grad = x->requires_grad;
  // Forward pass: compute softmax output.
  // Note: `softmax` might be renamed to `softmax_forward` or similar for
  // clarity when `softmax_backward` exists.
  tensor::Tensor out_data = tensor::ops::softmax(x->data);

  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    // Capture the output data (softmax probabilities) for the backward pass.
    // This is crucial for computing the Jacobian-vector product of softmax.
    auto non_copyable_lambda =
        [x, s = out_data.clone(), // Clone 's' to capture by value.
         axis](Variable *out_var) {
          if (x->requires_grad) {
            const tensor::Tensor &upstream_grad = out_var->grad;
            // The Jacobian-vector product for softmax.
            // dL/dx = (dL/dy * y) - (y * sum(dL/dy * y, axis=last)).
            // This assumes softmax was applied along the specified 'axis'.
            tensor::Tensor s_mult_grad = tensor::ops::mul(s, upstream_grad);
            tensor::Tensor sum_s_grad =
                tensor::ops::sum(s_mult_grad, axis, /*keep_dims=*/true);
            tensor::Tensor sub_term = tensor::ops::mul(s, sum_s_grad);
            tensor::Tensor x_grad = tensor::ops::sub(s_mult_grad, sub_term);
            accumulate_grad(x, x_grad);
          }
        };
    // Use a shared_ptr to manage the lifetime of the lambda, allowing it to be
    // stored in std::function, which requires copyability.
    auto shared_lambda = std::make_shared<decltype(non_copyable_lambda)>(
        std::move(non_copyable_lambda));
    backward_step->pass = [shared_lambda](Variable *out_var) {
      (*shared_lambda)(out_var);
    };
  }

  return ctx.create_variable(std::move(out_data), output_requires_grad,
                             std::vector<Variable *>{x}, backward_step,
                             "softmax");
}

} // namespace sylvan::core
