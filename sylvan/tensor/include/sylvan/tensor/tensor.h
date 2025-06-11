// sylvan/tensor/tensor.h
//
// Defines the core Tensor class and related enums/utilities for Sylvan.
// This file provides the fundamental building blocks for numerical data
// representation and manipulation within the Sylvan framework, supporting
// different data types and devices (primarily CUDA).
//
// Author: Zijing Zhang
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>
#include <numeric>   // For std::accumulate
#include <stdexcept> // For std::runtime_error
#include <vector>

/**
 * @brief Main namespace for the Sylvan deep learning framework.
 */
namespace sylvan {

/**
 * @brief Contains core tensor definitions and operations.
 *
 * This namespace provides the fundamental data structure `Tensor`
 * along with enumerations for data types (`DType`) and devices (`Device`),
 * and utility functions for tensor creation, manipulation, and debugging.
 */
namespace tensor {

/**
 * @brief Type alias for representing tensor shapes as a vector of 64-bit integers.
 */
using Shape = std::vector<int64_t>;

/**
 * @brief Enumerates the supported computation devices.
 */
enum class Device {
  CUDA, ///< NVIDIA GPU device.
  // CPU,  ///< Central Processing Unit (future support).
};

/**
 * @brief Enumerates the supported data types for tensors.
 */
enum class DType {
  Float32, ///< 32-bit floating-point number.
  Int32,   ///< 32-bit signed integer.
  // Future types
  // Float16, ///< 16-bit floating-point number.
  // Bool,    ///< Boolean type.
};

/**
 * @brief Helper function to get the size in bytes for a given DType.
 *
 * @param dtype The DType to query.
 * @return The size in bytes of a single element of the specified DType.
 * @throws std::runtime_error if the DType is unsupported.
 */
inline size_t dtype_to_size(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return sizeof(float);
  case DType::Int32:
    return sizeof(int32_t);
  // case DType::Float16: return sizeof(half);
  default:
    throw std::runtime_error("Unsupported DType");
  }
}

/**
 * @brief Custom deleter for `std::shared_ptr` to manage CUDA device memory.
 *
 * This struct provides an `operator()` that calls `cudaFree` to correctly
 * deallocate memory allocated on a CUDA device when the `std::shared_ptr`
 * goes out of scope or is reset.
 */
struct CudaDeleter {
  void operator()(void *ptr) const;
};

/**
 * @brief Represents a multi-dimensional array (tensor) for numerical computation.
 *
 * The `Tensor` class manages data on a specified device (currently CUDA)
 * with a given shape and data type. It uses `std::shared_ptr` with a custom
 * deleter (`CudaDeleter`) for automatic memory management on the GPU.
 *
 * This class is designed to be movable but not copyable, promoting explicit
 * cloning when data duplication is needed.
 */
class Tensor {
public:
  /**
   * @brief Constructs a new Tensor with specified shape, data type, and device.
   *
   * Allocates memory on the specified device and initializes it (contents are undefined).
   *
   * @param shape The shape of the tensor (e.g., `{2, 3}` for a 2x3 matrix).
   * @param dtype The data type of the tensor elements (e.g., `DType::Float32`).
   * @param device The device where the tensor data will reside (default `Device::CUDA`).
   */
  Tensor(const Shape &shape, DType dtype, Device device = Device::CUDA);

  /**
   * @brief Default constructor for an empty Tensor.
   *
   * Creates a tensor with null data pointer, Float32 dtype, and 0 elements.
   * Useful for declaring tensors that will be assigned later via move assignment.
   */
  Tensor()
      : data_(nullptr), dtype_(DType::Float32), numel_(0),
        device_(Device::CUDA) {}

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of the underlying data from `other` to this Tensor.
   * After the move, `other` will be in a valid but unspecified state (e.g., empty).
   *
   * @param other The Tensor to move from.
   */
  Tensor(Tensor &&other) noexcept;

  /**
   * @brief Move assignment operator.
   *
   * Transfers ownership of the underlying data from `other` to this Tensor,
   * releasing any resources currently held by this Tensor.
   *
   * @param other The Tensor to move from.
   * @return A reference to this Tensor.
   */
  Tensor &operator=(Tensor &&other) noexcept;

  // Delete copy constructor and copy assignment operator to prevent accidental copies.
  // Tensors are typically explicitly cloned when duplication is desired.
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  /**
   * @brief Creates a deep copy of this Tensor.
   *
   * Allocates new memory on the same device and copies all data from this Tensor.
   *
   * @return A new Tensor instance containing a copy of the data.
   */
  Tensor clone() const;

  /**
   * @brief Destructor.
   *
   * Releases the memory held by the `data_` `std::shared_ptr` using `CudaDeleter`.
   */
  ~Tensor();

  /**
   * @brief Returns the shape of the tensor.
   * @return A constant reference to the `Shape` vector.
   */
  const Shape &shape() const { return shape_; }

  /**
   * @brief Returns a view of the tensor with new shape.
   * @param new_shape The new shape of the tensor.
   * @return A new Tensor instance with the same data but new shape.
   */
  Tensor reshape(const Shape &new_shape) const;

  /**
   * @brief Returns the device on which the tensor data resides.
   * @return The `Device` enum value.
   */
  Device device() const { return device_; }

  /**
   * @brief Returns the data type of the tensor elements.
   * @return The `DType` enum value.
   */
  DType dtype() const { return dtype_; }

  /**
   * @brief Returns the total number of elements in the tensor.
   * @return The number of elements.
   */
  size_t numel() const { return numel_; }

  /**
   * @brief Returns the size of a specific dimension.
   * @param i The index of the dimension (0-based).
   * @return The size of the `i`-th dimension.
   * @throws std::out_of_range if `i` is out of bounds.
   */
  int64_t dim(size_t i) const { return shape_.at(i); }

  /**
   * @brief Returns the number of dimensions (rank) of the tensor.
   * @return The number of dimensions.
   */
  size_t ndim() const { return shape_.size(); }

  /**
   * @brief Returns a constant pointer to the underlying data, cast to the specified type.
   * @tparam T The desired data type (e.g., `float`, `int32_t`).
   * @return A constant pointer to the tensor's data.
   */
  template <typename T> const T *data() const {
    return static_cast<const T *>(data_.get());
  }

  /**
   * @brief Returns a mutable pointer to the underlying data, cast to the specified type.
   * @tparam T The desired data type (e.g., `float`, `int32_t`).
   * @return A mutable pointer to the tensor's data.
   */
  template <typename T> T *mutable_data() {
    return static_cast<T *>(data_.get());
  }

  /**
   * @brief Specialization: Returns a mutable float pointer to the underlying data.
   *
   * This overload is provided for convenience when the DType is known to be Float32.
   * @return A mutable float pointer to the tensor's data.
   */
  float *mutable_data() { return static_cast<float *>(data_.get()); }

private:
  Tensor(const Shape &shape, DType dtype, Device device,
          std::shared_ptr<void> data);

  std::shared_ptr<void> data_; ///< Pointer to the raw tensor data on the device. Uses `void*` for type genericity.
  Shape shape_;                ///< The dimensions of the tensor.
  DType dtype_;                ///< The data type of the tensor elements.
  size_t numel_;               ///< Total number of elements in the tensor (product of all dimensions).
  Device device_;              ///< The device where the tensor data is stored.
};

/**
 * @brief Prints the contents of a tensor to the console.
 *
 * This utility function copies the tensor data to host memory and then prints it.
 * Useful for debugging and inspection.
 *
 * @param t The tensor to print.
 * @param name An optional name to prefix the output for better identification.
 */
void print_tensor(const Tensor &t, const std::string &name = "");

} // namespace tensor
} // namespace sylvan
