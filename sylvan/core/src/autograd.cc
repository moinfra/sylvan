#include "sylvan/core/autograd.h"
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <vector>

namespace sylvan::core {
using namespace sylvan::tensor;

namespace {

void build_topological_sort(Variable *node,
                            std::unordered_set<Variable *> &visited,
                            std::vector<Variable *> &tape) {
  if (visited.count(node) > 0) {
    return;
  }
  visited.insert(node);
  for (Variable *prev_node : node->prev_) {
    build_topological_sort(prev_node, visited, tape);
  }
  tape.push_back(node);
}

} // namespace

Variable::Variable(Tensor &&data, std::vector<Variable *> prev,
                   std::shared_ptr<BackwardStep> grad_fn, std::string op_name)
    : data(std::move(data)), grad({this->data.shape()}), prev_(prev),
      grad_fn(grad_fn), op_name(op_name) {
  ops::fill_(this->grad, 0.0f);
}

void Variable::backward() {
  std::vector<Variable *> tape;
  std::unordered_set<Variable *> visited;
  build_topological_sort(this, visited, tape);

  ops::fill_(this->grad, 1.0f);

  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    Variable *v = *it;
    if (v->grad_fn) {
      v->grad_fn->pass(*v);
    }
  }
}

Variable add(Variable &a, Variable &b) {

  Tensor out_data = ops::add(a.data, b.data);

  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [&a, &b](Variable &out_var) {
    // Handle broadcasting: if an input was smaller, its gradient must be
    // summed.
    if (a.data.numel() < out_var.grad.numel()) {
      ops::add_(a.grad, ops::sum(out_var.grad));
    } else {
      ops::add_(a.grad, out_var.grad);
    }
    if (b.data.numel() < out_var.grad.numel()) {
      ops::add_(b.grad, ops::sum(out_var.grad));
    } else {
      ops::add_(b.grad, out_var.grad);
    }
  };

  return Variable(std::move(out_data), {&a, &b}, backward_step, "add");
}

Variable matmul(Variable &a, Variable &b) {
  Tensor out_data = ops::matmul(a.data, b.data);

  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [&a, &b](Variable &out_var) {
    // dL/da = dL/d_out * b^T
    Tensor b_t = ops::transpose(b.data);
    ops::add_(a.grad, ops::matmul(out_var.grad, b_t));

    // dL/db = a^T * dL/d_out
    Tensor a_t = ops::transpose(a.data);
    ops::add_(b.grad, ops::matmul(a_t, out_var.grad));
  };

  return Variable(std::move(out_data), {&a, &b}, backward_step, "matmul");
}

Variable mse_loss(Variable &pred, Variable &target) {
  // value calculation: loss = mean((pred - target)^2)
  Tensor diff = ops::sub(pred.data, target.data);
  Tensor diff_sq = ops::mul(diff, diff);
  Tensor loss_val = ops::mean(diff_sq);

  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [&pred, &target](Variable &out_var) {
    // d_loss/d_pred = 2/N * (pred - target)
    float n_inv = 2.0f / static_cast<float>(pred.data.dim(0));
    Tensor grad_val = ops::mul(ops::sub(pred.data, target.data),
                               ops::from_host({n_inv}, {1}));
    ops::add_(pred.grad, grad_val);
  };

  return Variable(std::move(loss_val), {&pred}, backward_step, "mse_loss");
}

} // namespace sylvan::core
