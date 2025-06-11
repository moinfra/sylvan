// sylvan/tensor/backend/cuda/src/tensor.cc
//
// Implements the core Tensor class methods, including constructors, memory
// management, and utility functions. This file handles the lifecycle of a
// tensor's metadata and its underlying GPU memory buffer.
//
// Author: Sylvan Team
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/tensor/tensor.h"
#include <algorithm> // For std::min
#include <cassert>
#include <iostream>
#include <numeric> // For std::accumulate
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "macros.h"

namespace sylvan::tensor {

//===----------------------------------------------------------------------===//
// CudaDeleter
//===----------------------------------------------------------------------===//

/**
 * @brief A custom deleter for `std::shared_ptr` to manage CUDA device memory.
 *
 * This functor is used by `std::shared_ptr` to correctly release GPU memory
 * using `cudaFree` when the last reference to a Tensor's data buffer is
 * destroyed. This is the core of our RAII-based memory management.
 */
void CudaDeleter::operator()(void *ptr) const {
  if (ptr) {
    // Note: This check runs during program shutdown. If the CUDA context is
    // already destroyed, this call might fail. For robust shutdown, error
    // handling here might be relaxed, but for general use, it's crucial.
    CUDA_CHECK(cudaFree(ptr));
  }
}

//===----------------------------------------------------------------------===//
// Tensor Constructors & Lifecycle
//===----------------------------------------------------------------------===//

/**
 * @brief The primary public constructor. Allocates new memory on the device.
 *
 * This constructor is used to create a new, owning Tensor. It calculates the
 * total number of elements, allocates the required memory on the CUDA device,
 * and wraps the device pointer in a `std::shared_ptr` with the `CudaDeleter`.
 */
Tensor::Tensor(const Shape &shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), numel_(0) {
  if (device != Device::CUDA) {
    throw std::runtime_error("Sylvan currently only supports CUDA devices.");
  }

  // Calculate the total number of elements from the shape.
  if (!shape.empty()) {
    numel_ = 1;
    for (int64_t dim : shape_) {
      assert(dim >= 0 && "Tensor dimensions cannot be negative.");
      numel_ *= dim;
    }
  }

  // Sanity checks for element count.
  assert((!shape.empty() || numel_ == 0) &&
         "numel should be 0 for empty shape.");
  bool has_zero_dim = false;
  for (int64_t dim : shape_) {
    if (dim == 0) {
      has_zero_dim = true;
      break;
    }
  }
  assert((!has_zero_dim || numel_ == 0) &&
         "numel should be 0 if any dimension is 0.");

  // Allocate memory only if the tensor is non-empty.
  void *ptr = nullptr;
  if (numel_ > 0) {
    size_t byte_size = numel_ * dtype_to_size(dtype_);
    assert(byte_size > 0 &&
           "Allocating 0 bytes is not expected for numel > 0.");
    CUDA_CHECK(cudaMalloc(&ptr, byte_size));
  }
  // Create the smart pointer that manages the allocated memory.
  data_ = std::shared_ptr<void>(ptr, CudaDeleter());
}

/**
 * @brief Private constructor for creating zero-copy views.
 *
 * This constructor does NOT allocate new memory. Instead, it creates a new
 * Tensor "header" (metadata) that shares the existing memory buffer of another
 * Tensor. This is the core mechanism for zero-copy operations like `reshape`.
 * It is private to ensure that views are only created through controlled,
 * safe member functions.
 */
Tensor::Tensor(const Shape &shape, DType dtype, Device device,
               std::shared_ptr<void> data)
    : data_(std::move(data)), // Share ownership by moving the shared_ptr
      shape_(shape), dtype_(dtype), device_(device) {
  // Recalculate the number of elements for this new view.
  numel_ = std::accumulate(shape_.begin(), shape_.end(), 1LL,
                           std::multiplies<int64_t>());
}

/**
 * @brief Move constructor. Efficiently transfers ownership of tensor resources.
 */
Tensor::Tensor(Tensor &&other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
      dtype_(other.dtype_), numel_(other.numel_), device_(other.device_) {
  // Leave the moved-from object in a valid but empty state.
  other.data_ = nullptr;
  other.numel_ = 0;
}

/**
 * @brief Move assignment operator.
 */
Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    numel_ = other.numel_;
    device_ = other.device_;
    other.data_ = nullptr;
    other.numel_ = 0;
  }
  return *this;
}

/**
 * @brief Destructor. The `std::shared_ptr` automatically handles memory
 * release.
 */
Tensor::~Tensor() = default;

//===----------------------------------------------------------------------===//
// Tensor Member Functions
//===----------------------------------------------------------------------===//

/**
 * @brief Creates a deep copy of the tensor.
 *
 * This function allocates new device memory and performs a full
 * device-to-device copy of the tensor's data.
 * @return A new Tensor with its own distinct memory buffer.
 */
Tensor Tensor::clone() const {
  Tensor new_t(shape_, dtype_, device_);
  if (numel_ > 0) {
    size_t byte_size = numel_ * dtype_to_size(dtype_);
    CUDA_CHECK(cudaMemcpy(new_t.mutable_data<void>(), this->data<void>(),
                          byte_size, cudaMemcpyDeviceToDevice));
  }
  return new_t;
}

/**
 * @brief Creates a zero-copy view of the tensor with a new shape.
 *
 * This is a highly efficient operation that does not copy any data. It creates
 * a new Tensor object with a different shape that points to the *same*
 * underlying GPU memory.
 * @param new_shape The desired new shape. Must have the same total number of
 *                  elements as the original tensor.
 * @return A new Tensor view.
 */
Tensor Tensor::reshape(const Shape &new_shape) const {
  // -------- 1. Validate that the total number of elements remains the same --------
  int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL,
                                      std::multiplies<int64_t>());
  if (this->numel() != new_numel) {
    throw std::runtime_error("reshape: Cannot reshape tensor, number of "
                             "elements must be preserved.");
  }

  // -------- 2. Create the view using the private constructor --------
  // This call is valid because it's a member function calling its own private
  // constructor. It passes its own shared_ptr (`this->data_`) to the new
  // Tensor, effectively sharing ownership of the memory buffer.
  return Tensor(new_shape, this->dtype_, this->device_, this->data_);
}

//===----------------------------------------------------------------------===//
// Utility and Debugging Functions
//===----------------------------------------------------------------------===//

/**
 * @brief Internal helper to print tensor data for a specific type.
 *
 * This templated function copies a small portion of the tensor's data from
 * device to host and prints it to the console for debugging.
 */
template <typename T> void print_data(const Tensor &t) {
  std::vector<T> host_data(t.numel());
  if (t.numel() > 0) {
    CUDA_CHECK(cudaMemcpy(host_data.data(), t.data<T>(), t.numel() * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  std::cout << "  Data:" << std::endl << "  ";

  // Limit the number of printed elements to avoid flooding the console.
  size_t print_limit = std::min((size_t)100, t.numel());
  for (size_t i = 0; i < print_limit; ++i) {
    std::cout << host_data[i] << " ";
    // Add newlines to respect the tensor's last dimension for better
    // readability.
    if (t.ndim() > 1 && (i + 1) % t.dim(t.ndim() - 1) == 0) {
      std::cout << std::endl << "  ";
    }
  }

  if (t.numel() > print_limit) {
    std::cout << "... (" << t.numel() - print_limit << " more elements)"
              << std::endl;
  }
  if (print_limit > 0) {
    std::cout << std::endl;
  }
}

/**
 * @brief Prints a summary of a tensor's properties and a preview of its data.
 *
 * This is the main public function for debugging. It prints the tensor's name,
 * shape, data type, and device, then calls a specialized helper to print a
 * preview of the data itself.
 */
void print_tensor(const Tensor &t, const std::string &name) {
  if (!name.empty()) {
    std::cout << "Tensor: " << name << std::endl;
  }

  std::cout << "  Shape: [";
  for (size_t i = 0; i < t.ndim(); ++i) {
    std::cout << t.dim(i) << (i == t.ndim() - 1 ? "" : ", ");
  }
  std::cout << "]";

  std::cout << ", DType: ";
  switch (t.dtype()) {
  case DType::Float32:
    std::cout << "Float32";
    break;
  case DType::Int32:
    std::cout << "Int32";
    break;
  default:
    std::cout << "Unknown";
    break;
  }

  std::cout << ", Device: CUDA" << std::endl;

  // Runtime dispatch to the correct templated print helper based on DType.
  switch (t.dtype()) {
  case DType::Float32:
    print_data<float>(t);
    break;
  case DType::Int32:
    print_data<int>(t);
    break;
  default:
    std::cout << "  (Data printing for this DType is not supported)"
              << std::endl;
    break;
  }

  std::cout << "-----------------------------------" << std::endl;
}

} // namespace sylvan::tensor
