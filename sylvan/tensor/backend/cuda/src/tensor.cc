#include "sylvan/tensor/tensor.h"
#include "macros.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace sylvan::tensor {

void CudaDeleter::operator()(void *ptr) const {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

Tensor::Tensor(const Shape &shape, Device device)
    : shape_(shape), device_(device) {
  if (device != Device::CUDA) {
    throw std::runtime_error("Sylvan currently only supports CUDA devices.");
  }
  numel_ = 1;
  for (int64_t dim : shape_) {
    numel_ *= dim;
  }

  void *ptr = nullptr;
  if (numel_ > 0) {
    CUDA_CHECK(cudaMalloc(&ptr, numel_ * sizeof(float)));
  }
  data_ = std::shared_ptr<void>(ptr, CudaDeleter());
}

Tensor::Tensor(Tensor &&other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
      numel_(other.numel_), device_(other.device_) {
  other.numel_ = 0; // 防止旧对象析构时误操作
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    device_ = other.device_;
    other.numel_ = 0;
  }
  return *this;
}

Tensor Tensor::clone() const {
  Tensor new_t(shape_, device_);
  if (numel_ > 0) {
    CUDA_CHECK(cudaMemcpy(new_t.mutable_data(), this->data(),
                          numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  return new_t;
}

Tensor::~Tensor() = default;

void print_tensor(const Tensor &t, const std::string &name) {
  if (!name.empty()) {
    std::cout << "Tensor: " << name << std::endl;
  }
  std::cout << "  Shape: [";
  for (size_t i = 0; i < t.ndim(); ++i) {
    std::cout << t.dim(i) << (i == t.ndim() - 1 ? "" : ", ");
  }
  std::cout << "], Device: CUDA" << std::endl;

  std::vector<float> host_data(t.numel());
  if (t.numel() > 0) {
    CUDA_CHECK(cudaMemcpy(host_data.data(), t.data(), t.numel() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  std::cout << "  Data:" << std::endl;

  size_t print_limit = std::min((size_t)100, t.numel());
  for (size_t i = 0; i < print_limit; ++i) {
    std::cout << host_data[i] << " ";
    if (t.ndim() > 1 && (i + 1) % t.dim(t.ndim() - 1) == 0) {
      std::cout << std::endl;
    }
  }
  if (t.numel() > print_limit) {
    std::cout << "... (" << t.numel() - print_limit << " more elements)"
              << std::endl;
  }
  std::cout << "--------------------" << std::endl;
}

} // namespace sylvan::tensor
