// sylvan/core/src/position.cc
//
// Implements the utility function for generating fixed sinusoidal positional
// encodings, a crucial component for Transformer models to provide sequence
// order information.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/position.h"
#include "sylvan/tensor/operators.h"
#include <cmath>  // For std::sin, std::cos, std::pow
#include <vector> // For std::vector

namespace sylvan::core {

tensor::Tensor create_positional_encoding(int max_len, int d_model) {
  std::vector<float> data(max_len * d_model);
  for (int pos = 0; pos < max_len; ++pos) {
    for (int i = 0; i < d_model / 2; ++i) {
      double div_term_exp = (double)(2 * i) / d_model;
      double div_term = std::pow(10000.0, div_term_exp);
      double angle = (double)pos / div_term;
      data[pos * d_model + 2 * i] = static_cast<float>(std::sin(angle));
      // Handle odd d_model by ensuring the index is within bounds.
      if (2 * i + 1 < d_model) {
        data[pos * d_model + 2 * i + 1] = static_cast<float>(std::cos(angle));
      }
    }
  }
  // Reshape to [1, max_len, d_model] to allow broadcasting over the batch
  // dimension.
  return tensor::ops::from_host(
      data, {1, static_cast<int64_t>(max_len), static_cast<int64_t>(d_model)});
}
} // namespace sylvan::core
