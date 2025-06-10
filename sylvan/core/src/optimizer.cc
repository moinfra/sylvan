#include "sylvan/core/optimizer.h"
#include "sylvan/tensor/operators.h"

namespace sylvan::core {

SGD::SGD(std::vector<Variable*> params, float lr) : params_(params), lr_(lr) {}

void SGD::step() {
    // Create scalar tensor for learning rate once
    auto lr_tensor = tensor::ops::from_host({-lr_}, {1});

    for (auto* param : params_) {
        // param->data -= lr * param->grad
        auto update = tensor::ops::mul(param->grad, lr_tensor);
        tensor::ops::add_(param->data, update);
    }
}

void SGD::zero_grad() {
    for (auto* param : params_) {
        tensor::ops::fill_(param->grad, 0.0f);
    }
}

} // namespace sylvan::core
