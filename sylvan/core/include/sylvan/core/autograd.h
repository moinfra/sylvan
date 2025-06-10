#pragma once

#include "sylvan/tensor/tensor.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sylvan::core {

class Variable;
class GraphContext;

struct BackwardStep {
  std::function<void(Variable *)> pass;
};

class Variable {
public:
  tensor::Tensor data;
  tensor::Tensor grad;
  std::vector<Variable *> prev_;
  std::shared_ptr<BackwardStep> grad_fn;
  std::string op_name;

  explicit Variable(tensor::Tensor &&data, std::vector<Variable *> prev = {},
                    std::shared_ptr<BackwardStep> grad_fn = nullptr,
                    std::string op_name = "");

  void backward();
};

Variable *add(GraphContext &ctx, Variable *a, Variable *b);
Variable *matmul(GraphContext &ctx, Variable *a, Variable *b);
Variable *mse_loss(GraphContext &ctx, Variable *pred, Variable *target);

} // namespace sylvan::core
