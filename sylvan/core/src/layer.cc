#include "sylvan/core/layer.h"
#include "sylvan/tensor/operators.h"
#include <cmath>

namespace sylvan::core {

namespace {

sylvan::tensor::Tensor create_weights(int in_features, int out_features) {
  float std = sqrt(2.0f / in_features);
  sylvan::tensor::Tensor t({(int64_t)in_features, (int64_t)out_features});
  // Use uniform distribution that approximates standard normal * std
  tensor::ops::uniform_(t, -std, std);
  return t;
}

sylvan::tensor::Tensor create_bias(int out_features) {
  sylvan::tensor::Tensor t({1, (int64_t)out_features});
  tensor::ops::fill_(t, 0.0f);
  return t;
}

} // namespace

Linear::Linear(int in_features, int out_features)
    : W(create_weights(in_features, out_features)),
      b(create_bias(out_features)) {}

Variable Linear::forward(Variable &x) {
  auto out = matmul(x, W);
  return add(out, b);
}

std::vector<Variable *> Linear::parameters() { return {&W, &b}; }

} // namespace sylvan::core
