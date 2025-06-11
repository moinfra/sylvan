#pragma once

// 这个文件不应该被公开的头文件包含。
// 它只被 sylvan/tensor/backend/cuda/src/operators.cc 包含。

#include "sylvan/tensor/tensor.h"
#include "sylvan/tensor/backend/cuda/include/macros.h"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <stdexcept>

// 将模板实现放在这个 -inl.h 文件中
namespace sylvan::tensor::ops {

template <typename T>
Tensor from_host(const std::vector<T> &data, const Shape &shape) {
  DType dtype;
  if (std::is_same_v<T, float>) {
    dtype = DType::Float32;
  } else if (std::is_same_v<T, int>) {
    dtype = DType::Int32;
  } else {
    throw std::runtime_error("Unsupported type for from_host.");
  }

  Tensor t(shape, dtype);
  if (!data.empty()) {
    size_t expected_numel = std::accumulate(shape.begin(), shape.end(), 1LL,
                                            std::multiplies<int64_t>());
    if (expected_numel != data.size()) {
      throw std::runtime_error("Shape and data size mismatch in from_host.");
    }
    CUDA_CHECK(cudaMemcpy(t.mutable_data<T>(), data.data(),
                          data.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  return t;
}

template<typename T>
std::vector<T> clone_to_host(const Tensor &t) {
    DType expected_dtype;
    if (std::is_same_v<T, float>) {
        expected_dtype = DType::Float32;
    } else if (std::is_same_v<T, int>) {
        expected_dtype = DType::Int32;
    } else {
        throw std::runtime_error("Unsupported type for clone_to_host.");
    }

    if (t.dtype() != expected_dtype) {
        throw std::runtime_error("Requested host type does not match tensor's DType.");
    }

    std::vector<T> host_data(t.numel());
    if (t.numel() > 0) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), t.data<void>(), 
                              t.numel() * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host_data;
}

} // namespace sylvan::tensor::ops
