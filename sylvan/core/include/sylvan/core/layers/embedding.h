// Defines the Embedding layer, which maps integer IDs to dense vector representations.
// This layer is commonly used as the first step in neural networks processing
// discrete tokens (e.g., words in NLP).
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include "sylvan/core/autograd.h"

namespace sylvan::core {
/**
 * @brief Represents an Embedding layer that maps integer IDs to dense vectors.
 *
 * This layer maintains a learnable weight matrix where each row corresponds
 * to a unique embedding vector for a given integer ID. It's typically used
 * for token embeddings in sequence models.
 */
struct Embedding {
    Variable weights; ///< The learnable embedding matrix. Each row is an embedding vector.

    /**
     * @brief Constructs an Embedding layer.
     *
     * @param num_embeddings The size of the vocabulary (number of unique IDs).
     * @param embedding_dim The dimension of the output embedding vector for each ID.
     */
    Embedding(int num_embeddings, int embedding_dim);

    /**
     * @brief Performs the forward pass of the Embedding layer.
     *
     * Given a tensor of integer IDs, this method retrieves their corresponding
     * embedding vectors from the `weights` matrix.
     *
     * @param ctx The graph context for tracing operations.
     * @param ids A Variable containing a tensor of integer IDs (DType::Int32).
     * @return A Variable containing the embedded vectors (DType::Float32),
     *         with shape `[..., embedding_dim]`.
     */
    Variable* forward(GraphContext& ctx, Variable* ids);

    /**
     * @brief Returns a vector of learnable parameters for this layer.
     *
     * @return A `std::vector` containing pointers to the `weights` Variable.
     */
    std::vector<Variable*> parameters();
};
}
