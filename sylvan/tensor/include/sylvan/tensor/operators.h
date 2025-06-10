#pragma once

#include "sylvan/tensor/tensor.h"
#include <vector>
#include <string>

namespace sylvan::tensor::ops {

/**
 * @brief 从主机 (CPU) 的 std::vector 创建一个张量，并将数据上传到 GPU。
 * @param data 包含要上传数据的 std::vector。
 * @param shape 张量的目标形状。
 * @return 一个新的、位于 GPU 上的 Tensor 对象。
 */
Tensor from_host(const std::vector<float>& data, const Shape& shape);

/**
 * @brief 将 GPU 上的张量数据克隆到主机的 std::vector。
 * @param t 要克隆的源张量。
 * @return 包含张量数据的 std::vector。
 */
std::vector<float> clone_to_host(const Tensor& t);

// --- 原地修改 (In-place) 操作 ---
// 按照 PyTorch 的惯例，原地操作以 `_` 结尾。

/**
 * @brief [In-place] 使用指定的常量值填充张量。
 * @param t 要填充的张量。
 * @param value 用于填充的浮点数值。
 */
void fill_(Tensor& t, float value);

void add_(Tensor& t, const Tensor& other);

/**
 * @brief [In-place] 使用 [from, to) 范围内的均匀分布随机数填充张量。
 * @param t 要填充的张量。
 * @param from 均匀分布的下界（包含）。
 * @param to 均匀分布的上界（不包含）。
 */
void uniform_(Tensor& t, float from, float to);

// --- 返回新张量的操作 ---

/**
 * @brief 逐元素加法，C = A + B。支持基础的广播。
 * @param a 第一个输入张量。
 * @param b 第二个输入张量。
 * @return 一个新的、包含计算结果的张量。
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief 逐元素减法，C = A - B。
 * @param a 第一个输入张量。
 * @param b 第二个输入张量。
 * @return 一个新的、包含计算结果的张量。
 */
Tensor sub(const Tensor& a, const Tensor& b);

/**
 * @brief 逐元素乘法，C = A * B。
 * @param a 第一个输入张量。
 * @param b 第二个输入张量。
 * @return 一个新的、包含计算结果的张量。
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief 矩阵乘法，C = A @ B。目前仅支持 2D 张量。
 * @param a 第一个输入张量 (2D)。
 * @param b 第二个输入张量 (2D)。
 * @return 一个新的、包含矩阵乘法结果的张量。
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief 矩阵转置。目前仅支持 2D 张量。
 * @param t 要转置的张量。
 * @return 一个新的、转置后的张量。
 */
Tensor transpose(const Tensor& t);

/**
 * @brief 计算张量所有元素的和。
 * @param t 输入张量。
 * @return 一个形状为 {1} 的标量张量，包含求和结果。
 */
Tensor sum(const Tensor& t);

/**
 * @brief 计算张量所有元素的均值。
 * @param t 输入张量。
 * @return 一个形状为 {1} 的标量张量，包含均值结果。
 */
Tensor mean(const Tensor& t);

} // namespace sylvan::tensor::ops
