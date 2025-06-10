#include "sylvan/core/autograd.h"
#include "sylvan/core/layer.h"
#include "sylvan/core/optimizer.h"
#include "sylvan/tensor/operators.h"
#include <iostream>
#include <vector>

using namespace sylvan;

int main() {
    std::cout << "🚀 Starting Sylvan Linear Regression Example!" << std::endl;

    // 1. 生成数据 y = 2 * x + 0.5 + noise
    const int num_samples = 100;
    const int in_features = 1;
    const int out_features = 1;

    std::vector<float> x_data(num_samples * in_features);
    std::vector<float> y_data(num_samples * out_features);
    for (int i = 0; i < num_samples; ++i) {
        x_data[i] = static_cast<float>(i) / 10.0f;
        y_data[i] = 2.0f * x_data[i] + 0.5f;
    }

    // 上传到GPU
    auto x_tensor = tensor::ops::from_host(x_data, {num_samples, in_features});
    auto y_tensor = tensor::ops::from_host(y_data, {num_samples, out_features});
    
    core::Variable X(std::move(x_tensor));
    core::Variable Y_true(std::move(y_tensor));

    // 2. 定义模型和优化器
    core::Linear model(in_features, out_features);
    core::SGD optimizer(model.parameters(), 0.001f);
    
    // 3. 训练循环
    int epochs = 200;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // a. 清零梯度
        optimizer.zero_grad();

        // b. 前向传播
        auto Y_pred = model.forward(X);

        // c. 计算损失
        auto loss = core::mse_loss(Y_pred, Y_true);

        // d. 反向传播计算梯度
        loss.backward();

        // e. 更新权重
        optimizer.step();

        if (epoch % 20 == 0) {
            auto loss_val = tensor::ops::clone_to_host(loss.data);
            std::cout << "Epoch " << epoch << ", Loss: " << loss_val[0] << std::endl;
        }
    }

    std::cout << "✅ Training finished." << std::endl;
    tensor::print_tensor(model.W.data, "Final W (target ~2.0)");
    tensor::print_tensor(model.b.data, "Final b (target ~0.5)");

    return 0;
}
