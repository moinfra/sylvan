#pragma once

#include "sylvan/tensor/tensor.h"
#include "sylvan/tensor/operators.h"
#include <functional>
#include <memory>
#include <vector>

namespace sylvan::core {

using sylvan::tensor::Tensor;

class Variable;

struct BackwardStep {
    std::function<void(Variable&)> pass;
};

class Variable {
public:
    Tensor data;
    Tensor grad;
    std::vector<Variable*> prev_;
    std::shared_ptr<BackwardStep> grad_fn;
    std::string op_name;

    explicit Variable(Tensor&& data, std::vector<Variable*> prev = {}, std::shared_ptr<BackwardStep> grad_fn = nullptr, std::string op_name = "");

    void backward();
};


Variable add(Variable& a, Variable& b);
Variable matmul(Variable& a, Variable& b);

Variable mse_loss(Variable& pred, Variable& target);

} // namespace sylvan::core
