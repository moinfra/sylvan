// sylvan/examples/linear_regression.cc
//
// Demonstrates a simple linear regression model using the Sylvan framework.
// This example showcases the core components: Tensor operations,
// Autograd graph construction, Layers (Linear), and Optimizers (SGD).
// It trains a model to fit a linear relationship y = 2x + 0.5.
//
// Author: Zijing Zhang
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/autograd.h"
#include "sylvan/core/graph.h"
#include "sylvan/core/layers/linear.h"
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

  auto x_tensor_base =
      tensor::ops::from_host(x_data, {num_samples, in_features});
  auto y_tensor_base =
      tensor::ops::from_host(y_data, {num_samples, out_features});

  core::Linear model(in_features, out_features);
  core::SGD optimizer(model.parameters(), 0.001f);

  int epochs = 200;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    core::GraphContext graph;

    auto *X = graph.create_variable(x_tensor_base.clone());
    auto *Y_true = graph.create_variable(y_tensor_base.clone());

    optimizer.zero_grad();

    auto *Y_pred = model.forward(graph, X);
    auto *loss = core::mse_loss(graph, Y_pred, Y_true);

    loss->backward();
    optimizer.step();

    if (epoch % 20 == 0) {
      auto loss_val = tensor::ops::clone_to_host<float>(loss->data);
      std::cout << "Epoch " << epoch << ", Loss: " << loss_val[0] << std::endl;
    }
  }

  std::cout << "✅ Training finished." << std::endl;
  tensor::print_tensor(model.W.data, "Final W (target ~2.0)");
  tensor::print_tensor(model.b.data, "Final b (target ~0.5)");

  return 0;
}
