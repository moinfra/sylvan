#include "sylvan/core/autograd.h"
#include "sylvan/core/graph.h"
#include "sylvan/core/layer.h"
#include "sylvan/core/optimizer.h"
#include "sylvan/tensor/operators.h"
#include <iostream>
#include <vector>

using namespace sylvan;

int main() {
  std::cout << "🚀 Starting Sylvan Linear Regression Example!" << std::endl;

  const int num_samples = 100;
  const int in_features = 1;
  const int out_features = 1;

  std::vector<float> x_data(num_samples * in_features);
  std::vector<float> y_data(num_samples * out_features);
  for (int i = 0; i < num_samples; ++i) {
    x_data[i] = static_cast<float>(i) / 10.0f;
    y_data[i] = 2.0f * x_data[i] + 0.5f;
  }

  // Base tensors that hold the dataset, owned by main.
  auto x_tensor_base =
      tensor::ops::from_host(x_data, {num_samples, in_features});
  auto y_tensor_base =
      tensor::ops::from_host(y_data, {num_samples, out_features});

  // Model and optimizer are created once and own the persistent parameters.
  core::Linear model(in_features, out_features);
  core::SGD optimizer(model.parameters(), 0.001f);

  int epochs = 200;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // A new graph context is created for each training iteration.
    // Its Arena will be automatically freed when 'graph' goes out of scope.
    core::GraphContext graph;

    // Create leaf variables for this iteration's graph from base data.
    auto *X = graph.create_variable(x_tensor_base.clone());
    auto *Y_true = graph.create_variable(y_tensor_base.clone());

    // Standard training steps
    optimizer.zero_grad();

    auto *Y_pred = model.forward(graph, X);
    auto *loss = core::mse_loss(graph, Y_pred, Y_true);

    loss->backward();
    optimizer.step();

    if (epoch % 20 == 0) {
      auto loss_val = tensor::ops::clone_to_host(loss->data);
      std::cout << "Epoch " << epoch << ", Loss: " << loss_val[0] << std::endl;
    }
  }

  std::cout << "✅ Training finished." << std::endl;
  tensor::print_tensor(model.W.data, "Final W (target ~2.0)");
  tensor::print_tensor(model.b.data, "Final b (target ~0.5)");

  return 0;
}
