#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sylvan::tensor {

using Shape = std::vector<int64_t>;

//
enum class Device { CUDA };

struct CudaDeleter {
  void operator()(void *ptr) const;
};

class Tensor {
public:
  Tensor(const Shape &shape, Device device = Device::CUDA);

  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  Tensor clone() const;

  ~Tensor();

  const Shape &shape() const { return shape_; }
  Device device() const { return device_; }
  size_t numel() const { return numel_; }
  int64_t dim(size_t i) const { return shape_.at(i); }
  size_t ndim() const { return shape_.size(); }

  const float *data() const { return static_cast<const float *>(data_.get()); }

  float *mutable_data() { return static_cast<float *>(data_.get()); }

private:
  std::shared_ptr<void> data_;
  Shape shape_;
  size_t numel_;
  Device device_;
};

void print_tensor(const Tensor &t, const std::string &name = "");

} // namespace sylvan::tensor
