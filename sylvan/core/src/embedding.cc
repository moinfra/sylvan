// sylvan/core/src/layers/embedding.cc
//
// Implements the Embedding layer, mapping integer IDs to dense vector representations.
// It includes initialization of embedding weights and the forward/backward
// pass logic for embedding lookups.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/layers/embedding.h"
#include "sylvan/core/graph.h"
#include "sylvan/tensor/operators.h"
#include <iostream> 

namespace sylvan::core {

using namespace sylvan::tensor;

// Embedding::Embedding(int num_embeddings, int embedding_dim)
Embedding::Embedding(int num_embeddings, int embedding_dim)
    : weights(tensor::ops::from_host(
          std::vector<float>{}, // Data will be filled by uniform_
          {(int64_t)num_embeddings, (int64_t)embedding_dim})) {
  // Initialize weights uniformly.
  tensor::ops::uniform_(weights.data, -0.1f, 0.1f);
}

// Variable *Embedding::forward(GraphContext &ctx, Variable *ids)
Variable *Embedding::forward(GraphContext &ctx, Variable *ids) {
  if (ids->requires_grad) {
    // Input ID tensors typically do not require gradients.
    std::cerr << "Warning: Input to Embedding layer has requires_grad=true."
              << std::endl;
  }

  // 1. Forward computation: perform embedding lookup.
  tensor::Tensor out_data = tensor::ops::embedding_forward(weights.data, ids->data);

  // 2. Determine if the output Variable requires gradients.
  // The output requires gradients if the embedding weights themselves require gradients.
  bool output_requires_grad = weights.requires_grad;

  // 3. Define the backward step for gradient computation.
  std::shared_ptr<BackwardStep> backward_step = nullptr;
  if (output_requires_grad) {
    backward_step = std::make_shared<BackwardStep>();
    // The lambda captures `this` (for weights) and `ids` (for their values).
    backward_step->pass = [this, ids](Variable* out_var){
        // Gradients flow to embedding weights but not to the input IDs.
        if (this->weights.requires_grad) {
            tensor::Tensor w_grad = tensor::ops::embedding_backward(
                out_var->grad,
                ids->data,
                this->weights.data.shape() // Original shape of weights for aggregation
            );
            // Accumulate gradient onto the weights.
            accumulate_grad(&this->weights, w_grad);
        }
    };
  }

  // 4. Create the output Variable, linking it to its dependencies.
  // Both `weights` and `ids` are dependencies for the graph, even if `ids` does not require grad.
  return ctx.create_variable(std::move(out_data),
                             output_requires_grad,
                             std::vector{&weights, ids},
                             backward_step,
                             "embedding");
}

// std::vector<Variable *> Embedding::parameters()
std::vector<Variable *> Embedding::parameters() { return {&weights}; }

} // namespace sylvan::core
