#pragma once
#include "sylvan/tensor/tensor.h"

namespace sylvan::core {

/**
 * @brief Generates fixed sinusoidal positional encodings.
 *
 * This function creates a tensor of positional encodings following the
 * formula described in the "Attention Is All You Need" paper.
 * For each position `pos` and each dimension `i` within the embedding:
 * `PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))`
 * `PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))`
 *
 * The output tensor has shape `[1, max_len, d_model]`, allowing it to be
 * easily broadcasted across the batch dimension when added to token embeddings.
 *
 * @param max_len The maximum sequence length for which to generate encodings.
 * @param d_model The embedding dimension of the model.
 * @return A `sylvan::tensor::Tensor` containing the positional encodings.
 */
tensor::Tensor create_positional_encoding(int max_len, int d_model);

} // namespace sylvan::core
