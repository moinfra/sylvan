#include "sylvan/core/autograd.h"
#include "sylvan/core/graph.h"
#include "sylvan/tensor/operators.h"
#include <functional>
#include <unordered_set>
#include <vector>

using namespace sylvan::tensor;

namespace sylvan::core {

namespace {
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
      v->grad_fn->pass(v);
    }
  }
}

Variable *add(GraphContext &ctx, Variable *a, Variable *b) {
  Tensor out_data = ops::add(a->data, b->data);
  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [a, b](Variable *out_var) {
    if (a->data.numel() < out_var->grad.numel()) {
      ops::add_(a->grad, ops::sum(out_var->grad));
    } else {
      ops::add_(a->grad, out_var->grad);
    }
    if (b->data.numel() < out_var->grad.numel()) {
      ops::add_(b->grad, ops::sum(out_var->grad));
    } else {
      ops::add_(b->grad, out_var->grad);
    }
  };
  return ctx.create_variable(std::move(out_data), std::vector<Variable *>{a, b},
                             backward_step, "add");
}

Variable *matmul(GraphContext &ctx, Variable *a, Variable *b) {
  Tensor out_data = ops::matmul(a->data, b->data);
  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [a, b](Variable *out_var) {
    Tensor b_t = ops::transpose(b->data);
    ops::add_(a->grad, ops::matmul(out_var->grad, b_t));
    Tensor a_t = ops::transpose(a->data);
    ops::add_(b->grad, ops::matmul(a_t, out_var->grad));
  };
  return ctx.create_variable(std::move(out_data), std::vector<Variable *>{a, b},
                             backward_step, "matmul");
}

Variable *mse_loss(GraphContext &ctx, Variable *pred, Variable *target) {
  Tensor diff = ops::sub(pred->data, target->data);
  Tensor diff_sq = ops::mul(diff, diff);
  Tensor loss_val = ops::mean(diff_sq);
  auto backward_step = std::make_shared<BackwardStep>();
  backward_step->pass = [pred, target](Variable * /*out_var*/) {
    float n_inv = 2.0f / static_cast<float>(pred->data.dim(0));
    Tensor grad_val = ops::mul(ops::sub(pred->data, target->data),
                               ops::from_host({n_inv}, {1}));
    ops::add_(pred->grad, grad_val);
  };
  return ctx.create_variable(std::move(loss_val), std::vector<Variable *>{pred},
                             backward_step, "mse_loss");
}

} // namespace sylvan::core
