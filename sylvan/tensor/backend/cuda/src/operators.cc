// sylvan/tensor/operators.cc
//
// Implements the public API for tensor operations declared in operators.h.
// This file contains the logic for managing CUDA library handles, transferring
// data, and launching the CUDA kernels that perform the actual computations.
//
// Author: Sylvan Team
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/tensor/operators.h"

#include <cassert>
#include <stdexcept>

#include <cublas_v2.h>
#include <curand.h>

#include "kernels.h"
#include "macros.h"
#include "operators-inl.h"

namespace sylvan::tensor::ops {

//===----------------------------------------------------------------------===//
// CUDA Handle Management & RAII Wrappers
//===----------------------------------------------------------------------===//

/**
 * @brief Manages global CUDA library handles (cuBLAS, cuRAND) using RAII.
 *
 * This struct ensures that cuBLAS and cuRAND handles are created exactly once
 * and are properly destroyed upon program termination. It is accessed via the

 * `get_handles()` singleton accessor.
 */
struct CudaHandles {
  cublasHandle_t cublas_handle;
  curandGenerator_t curand_gen;

  CudaHandles() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    // Use a fixed seed for reproducibility in tests and examples.
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL));
  }

  ~CudaHandles() {
    // These calls are guaranteed to be safe even if creation failed,
    // as the handles would be nullptr.
    CURAND_CHECK(curandDestroyGenerator(curand_gen));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
  }
};

/**
 * @brief An RAII wrapper for managing a device memory buffer created from a
 * host vector.
 *
 * This class handles the allocation of device memory (`cudaMalloc`), copying
 * data from a host `std::vector`, and ensuring the memory is freed (`cudaFree`)
 * upon destruction. It is move-only to enforce unique ownership of the device
 * resource. It also correctly handles the `std::vector<bool>` specialization.
 *
 * @tparam T The data type of the elements.
 */
template <typename T> struct DeviceVector {
  T *d_ptr = nullptr;
  size_t count = 0;

  /**
   * @brief Constructs a DeviceVector by allocating and copying data from a host
   * vector.
   * @param h_vec The host `std::vector` containing the data to be copied to the
   * device.
   */
  DeviceVector(const std::vector<T> &h_vec) : count(h_vec.size()) {
    if (count > 0) {
      // `std::vector<bool>` is a packed bitset and doesn't have a contiguous
      // `.data()` method returning `bool*`. We must manually convert it to a
      // byte-per-value representation before copying.
      if constexpr (std::is_same_v<T, bool>) {
        std::vector<char> byte_vec(count);
        for (size_t i = 0; i < count; ++i) {
          byte_vec[i] = static_cast<char>(h_vec[i]);
        }
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(bool)));
        CUDA_CHECK(cudaMemcpy(d_ptr, byte_vec.data(), count * sizeof(char),
                              cudaMemcpyHostToDevice));
      } else {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_vec.data(), count * sizeof(T),
                              cudaMemcpyHostToDevice));
      }
    }
  }

  ~DeviceVector() {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
  }

  // Enforce unique ownership: delete copy constructor and assignment.
  DeviceVector(const DeviceVector &) = delete;
  DeviceVector &operator=(const DeviceVector &) = delete;

  // Enable move semantics for efficient transfers of ownership.
  DeviceVector(DeviceVector &&other) noexcept
      : d_ptr(other.d_ptr), count(other.count) {
    other.d_ptr = nullptr;
    other.count = 0;
  }
  DeviceVector &operator=(DeviceVector &&other) noexcept {
    if (this != &other) {
      if (d_ptr)
        cudaFree(d_ptr);
      d_ptr = other.d_ptr;
      count = other.count;
      other.d_ptr = nullptr;
      other.count = 0;
    }
    return *this;
  }

  /**
   * @brief Gets a raw pointer to the device memory.
   * @return A const pointer to the data on the device.
   */
  const T *get() const { return d_ptr; }
};

namespace {
// Anonymous namespace for file-internal helper functions.

/**
 * @brief Calculates the strides for a tensor with a given shape.
 * Assumes a standard row-major memory layout.
 * @param shape The shape of the tensor.
 * @return A vector of strides corresponding to each dimension.
 */
std::vector<int64_t> calculate_strides(const Shape &shape) {
  std::vector<int64_t> strides(shape.size());
  if (shape.empty()) {
    return strides;
  }

  strides.back() = 1;
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/**
 * @brief Validates if a vector represents a valid dimension permutation.
 * @param perm The permutation vector to check.
 * @param ndim The number of dimensions the permutation should apply to.
 * @return True if `perm` is a valid permutation of `[0, 1, ..., ndim-1]`.
 */
bool is_valid_permutation(const std::vector<int> &perm, int ndim) {
  if (perm.size() != ndim)
    return false;
  std::vector<bool> seen(ndim, false);
  for (int axis : perm) {
    if (axis < 0 || axis >= ndim || seen[axis]) {
      return false;
    }
    seen[axis] = true;
  }
  return true;
}

/**
 * @brief Flattens a shape for Layer Normalization.
 * LayerNorm normalizes over the last dimension. This function calculates the
 * product of all preceding dimensions (`rows`) and the size of the last
 * dimension (`cols`).
 * @param shape The input tensor shape.
 * @return A pair {rows, cols}.
 */
std::pair<int64_t, int64_t> get_norm_dims(const Shape &shape) {
  if (shape.size() < 1) {
    throw std::runtime_error("LayerNorm input must have at least 1 dimension.");
  }
  int64_t cols = shape.back();
  int64_t rows = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    rows *= shape[i];
  }
  return {rows, cols};
}

/**
 * @brief Calculates the number of elements up to the last dimension.
 * Useful for operations like Softmax that operate on the last dimension.
 * @param shape The input tensor shape.
 * @return The product of all dimensions except the last one.
 */
int64_t numel_to_last_dim(const Shape &shape) {
  if (shape.empty())
    return 0;
  int64_t numel = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    numel *= shape[i];
  }
  return numel;
}

/**
 * @brief Converts a Shape object to a human-readable string.
 */
std::string to_string(const Shape &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i < shape.size() - 1) {
      s += ", ";
    }
  }
  s += "]";
  return s;
}

/**
 * @brief Converts a DType enum to a human-readable string.
 */
std::string to_string(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return "Float32";
  case DType::Int32:
    return "Int32";
  default:
    return "Unknown DType";
  }
}

} // namespace

/**
 * @brief Provides thread-safe, singleton access to the CUDA library handles.
 * The static local variable ensures initialization happens only once.
 * @return A reference to the global CudaHandles object.
 */
static CudaHandles &get_handles() {
  static CudaHandles handles;
  return handles;
}

//===----------------------------------------------------------------------===//
// Host <-> Device Data Transfer
//===----------------------------------------------------------------------===//

Tensor from_host(const std::vector<float> &data, const Shape &shape) {
  return from_host<float>(data, shape);
}
Tensor from_host(const std::vector<int> &data, const Shape &shape) {
  return from_host<int>(data, shape);
}

template std::vector<float> clone_to_host<float>(const Tensor &t);
template std::vector<int> clone_to_host<int>(const Tensor &t);

//===----------------------------------------------------------------------===//
// In-place Ops
//===----------------------------------------------------------------------===//

void fill_(Tensor &t, float value) {
  if (t.numel() == 0)
    return;
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("fill_: operation only supports Float32 tensors, "
                             "but got tensor with dtype " +
                             to_string(t.dtype()));
  }
  launch_fill_kernel(t.mutable_data<float>(), value, t.numel());
}

void add_(Tensor &a, const Tensor &b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("add_: In-place add op currently only supports "
                             "Float32 tensors. Got a.dtype()=" +
                             to_string(a.dtype()) +
                             " and b.dtype()=" + to_string(b.dtype()));
  }
  if (b.numel() == 0)
    return;

  // -------- Fast path for identical shapes --------
  if (a.shape() == b.shape()) {
    assert(a.numel() == b.numel() && "Shapes are equal but numel are not.");
    assert(a.numel() > 0 && "add_ called with empty tensors in fast path.");
    launch_elementwise_add_kernel(a.mutable_data<float>(), b.data<float>(),
                                  a.numel());
    return;
  }

  // -------- Broadcasting Path --------
  // This section prepares data for a kernel that can handle broadcasting.
  int ndim_a = a.ndim();
  int ndim_b = b.ndim();

  if (ndim_b > ndim_a) {
    throw std::runtime_error("add_: Cannot broadcast a larger tensor to a "
                             "smaller one in-place. a.ndim=" +
                             std::to_string(ndim_a) +
                             ", b.ndim=" + std::to_string(ndim_b));
  }

  // -------- 1. Validate broadcast compatibility --------
  for (int i = 0; i < ndim_b; ++i) {
    int64_t dim_b_i = b.dim(i);
    int64_t dim_a_i = a.dim(ndim_a - ndim_b + i);
    if (dim_b_i != dim_a_i && dim_b_i != 1) {
      throw std::runtime_error(
          "add_: Tensors are not broadcast-compatible for in-place add. "
          "a.shape()=" +
          to_string(a.shape()) + ", b.shape()=" + to_string(b.shape()));
    }
  }

  // -------- 2. Calculate broadcast-aware strides for 'b' --------
  // The trick is that dimensions of size 1 have a stride of 0.
  auto a_strides = calculate_strides(a.shape());
  auto b_strides_full = calculate_strides(b.shape());
  std::vector<int64_t> b_strides_broadcast(ndim_a, 0);
  for (int i = 0; i < ndim_b; ++i) {
    if (b.dim(i) > 1) {
      b_strides_broadcast[ndim_a - ndim_b + i] = b_strides_full[i];
    }
  }

  // -------- 3. Transfer metadata to device and launch kernel --------
  DeviceVector<int64_t> d_a_shape(a.shape());
  DeviceVector<int64_t> d_a_strides(a_strides);
  DeviceVector<int64_t> d_b_strides(b_strides_broadcast);

  launch_broadcast_add_inplace_kernel(a.mutable_data<float>(), b.data<float>(),
                                      a.numel(), ndim_a, d_a_shape.get(),
                                      d_a_strides.get(), d_b_strides.get());
}

void slice_backward_(Tensor &full_grad_tensor, const Tensor &sliced_grad,
                     const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  if (full_grad_tensor.dtype() != DType::Float32 ||
      sliced_grad.dtype() != DType::Float32) {
    throw std::runtime_error(
        "slice_backward_: operation only supports Float32 tensors.");
  }
  if (sliced_grad.numel() == 0)
    return;

  std::vector<int64_t> offsets(ranges.size());
  for (size_t i = 0; i < ranges.size(); ++i) {
    offsets[i] = ranges[i].first;
  }

  DeviceVector<int64_t> d_full_shape(full_grad_tensor.shape());
  DeviceVector<int64_t> d_slice_shape(sliced_grad.shape());
  DeviceVector<int64_t> d_offsets(offsets);

  launch_scatter_add_kernel(full_grad_tensor.mutable_data<float>(),
                            sliced_grad.data<float>(), d_full_shape.get(),
                            d_slice_shape.get(), d_offsets.get(),
                            full_grad_tensor.ndim(), sliced_grad.numel());
}

void uniform_(Tensor &t, float from, float to) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("uniform_: operation only supports Float32 "
                             "tensors, but got tensor with dtype " +
                             to_string(t.dtype()));
  }
  if (t.numel() == 0)
    return;

  // Generate uniform random numbers in [0, 1).
  CURAND_CHECK(curandGenerateUniform(get_handles().curand_gen,
                                     t.mutable_data<float>(), t.numel()));
  // Scale and shift to the desired range [from, to).
  launch_scale_kernel(t.mutable_data<float>(), to - from, from, t.numel());
}

//===----------------------------------------------------------------------===//
// Out-of-place Ops
//===----------------------------------------------------------------------===//

Tensor add(const Tensor &a, const Tensor &b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("add: operation currently only supports Float32 "
                             "tensors.");
  }

  // -------- 1. Determine output shape via broadcasting rules --------
  int ndim_a = a.ndim();
  int ndim_b = b.ndim();
  int ndim_out = std::max(ndim_a, ndim_b);
  Shape out_shape(ndim_out);

  for (int i = 0; i < ndim_out; ++i) {
    int64_t dim_a =
        (i < ndim_out - ndim_a) ? 1 : a.dim(i - (ndim_out - ndim_a));
    int64_t dim_b =
        (i < ndim_out - ndim_b) ? 1 : b.dim(i - (ndim_out - ndim_b));
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::runtime_error(
          "add: Tensors are not broadcast-compatible. a.shape=" +
          to_string(a.shape()) + ", b.shape=" + to_string(b.shape()));
    }
    out_shape[i] = std::max(dim_a, dim_b);
  }

  Tensor c(out_shape, DType::Float32);
  if (c.numel() == 0)
    return c;

  // -------- 2. Calculate broadcast-aware strides for both inputs --------
  auto a_strides_full = calculate_strides(a.shape());
  auto b_strides_full = calculate_strides(b.shape());
  std::vector<int64_t> a_strides(ndim_out, 0);
  std::vector<int64_t> b_strides(ndim_out, 0);

  for (int i = 0; i < ndim_a; ++i) {
    if (a.dim(i) > 1 || ndim_a == 1)
      a_strides[ndim_out - ndim_a + i] = a_strides_full[i];
  }
  for (int i = 0; i < ndim_b; ++i) {
    if (b.dim(i) > 1 || ndim_b == 1)
      b_strides[ndim_out - ndim_b + i] = b_strides_full[i];
  }

  // -------- 3. Transfer metadata to device and launch kernel --------
  DeviceVector<int64_t> d_out_shape(out_shape);
  DeviceVector<int64_t> d_a_strides(a_strides);
  DeviceVector<int64_t> d_b_strides(b_strides);

  launch_broadcast_add_kernel(
      c.mutable_data<float>(), c.numel(), a.data<float>(), b.data<float>(),
      d_out_shape.get(), ndim_out, d_a_strides.get(), d_b_strides.get());

  return c;
}

Tensor sub(const Tensor &a, const Tensor &b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("sub: operation currently only supports Float32 "
                             "tensors.");
  }
  // Implemented as a + (-1 * b)
  Tensor neg_b = mul(b, from_host<float>({-1.0f}, {1}));
  return add(a, neg_b);
}

Tensor mul(const Tensor &a, const Tensor &b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("mul: operation currently only supports Float32 "
                             "tensors.");
  }

  // -------- Broadcasting logic is identical to add() --------
  int ndim_a = a.ndim();
  int ndim_b = b.ndim();
  int ndim_out = std::max(ndim_a, ndim_b);
  Shape out_shape(ndim_out);

  for (int i = 0; i < ndim_out; ++i) {
    int64_t dim_a =
        (i < ndim_out - ndim_a) ? 1 : a.dim(i - (ndim_out - ndim_a));
    int64_t dim_b =
        (i < ndim_out - ndim_b) ? 1 : b.dim(i - (ndim_out - ndim_b));
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::runtime_error(
          "mul: Tensors are not broadcast-compatible. a.shape=" +
          to_string(a.shape()) + ", b.shape=" + to_string(b.shape()));
    }
    out_shape[i] = std::max(dim_a, dim_b);
  }

  Tensor c(out_shape, DType::Float32);
  if (c.numel() == 0)
    return c;

  auto a_strides_full = calculate_strides(a.shape());
  auto b_strides_full = calculate_strides(b.shape());
  std::vector<int64_t> a_strides(ndim_out, 0);
  std::vector<int64_t> b_strides(ndim_out, 0);

  for (int i = 0; i < ndim_a; ++i) {
    if (a.dim(i) > 1)
      a_strides[ndim_out - ndim_a + i] = a_strides_full[i];
  }
  for (int i = 0; i < ndim_b; ++i) {
    if (b.dim(i) > 1)
      b_strides[ndim_out - ndim_b + i] = b_strides_full[i];
  }

  DeviceVector<int64_t> d_out_shape(out_shape);
  DeviceVector<int64_t> d_a_strides(a_strides);
  DeviceVector<int64_t> d_b_strides(b_strides);

  launch_broadcast_mul_kernel(
      c.mutable_data<float>(), c.numel(), a.data<float>(), b.data<float>(),
      d_out_shape.get(), ndim_out, d_a_strides.get(), d_b_strides.get());

  return c;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("matmul: operation currently only supports "
                             "Float32 tensors.");
  }

  const auto &a_shape = a.shape();
  const auto &b_shape = b.shape();
  if (a.ndim() < 2 || b.ndim() < 2) {
    throw std::runtime_error(
        "matmul: Tensors must have at least 2 dimensions.");
  }
  if (a_shape.back() != b_shape[b.ndim() - 2]) {
    throw std::runtime_error("matmul: Inner dimensions must match for matrix "
                             "multiplication. a.shape=" +
                             to_string(a.shape()) +
                             ", b.shape=" + to_string(b.shape()));
  }

  // -------- 1. Handle batch dimension broadcasting --------
  int batch_dims_a = a.ndim() - 2;
  int batch_dims_b = b.ndim() - 2;
  int max_batch_dims = std::max(batch_dims_a, batch_dims_b);
  std::vector<int64_t> batch_shape(max_batch_dims);
  int batch_count = 1;

  for (int i = 0; i < max_batch_dims; ++i) {
    int64_t dim_a = (i < max_batch_dims - batch_dims_a)
                        ? 1
                        : a_shape[i - (max_batch_dims - batch_dims_a)];
    int64_t dim_b = (i < max_batch_dims - batch_dims_b)
                        ? 1
                        : b_shape[i - (max_batch_dims - batch_dims_b)];
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::runtime_error(
          "matmul: Batch dimensions are not broadcastable.");
    }
    batch_shape[i] = std::max(dim_a, dim_b);
    batch_count *= batch_shape[i];
  }

  // -------- 2. Determine matrix dimensions and output shape --------
  int64_t M = a_shape[a.ndim() - 2];
  int64_t K = a_shape[a.ndim() - 1];
  int64_t N = b_shape.back();
  Shape out_shape(max_batch_dims + 2);
  std::copy(batch_shape.begin(), batch_shape.end(), out_shape.begin());
  out_shape[max_batch_dims] = M;
  out_shape[max_batch_dims + 1] = N;
  Tensor c(out_shape, DType::Float32);

  // -------- 3. Launch appropriate cuBLAS kernel --------
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (batch_count == 1 && a.ndim() <= 2 && b.ndim() <= 2) {
    // -------- Standard non-batched matrix multiplication --------
    CUBLAS_CHECK(cublasSgemm(get_handles().cublas_handle, CUBLAS_OP_N,
                             CUBLAS_OP_N, N, M, K, &alpha, b.data<float>(), N,
                             a.data<float>(), K, &beta, c.mutable_data<float>(),
                             N));
  } else {
    // -------- Batched matrix multiplication with broadcasting support --------
    int64_t a_matrix_size = M * K;
    int64_t b_matrix_size = K * N;
    int64_t c_matrix_size = M * N;
    // If a tensor's batch size is 1, its stride is 0, enabling broadcasting.
    int64_t a_batch_stride = (a.numel() == a_matrix_size) ? 0 : a_matrix_size;
    int64_t b_batch_stride = (b.numel() == b_matrix_size) ? 0 : b_matrix_size;

    // cublasSgemmStridedBatched is more efficient than pointer arrays
    // if strides are regular.
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        get_handles().cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        b.data<float>(), N, b_batch_stride, a.data<float>(), K, a_batch_stride,
        &beta, c.mutable_data<float>(), N, c_matrix_size, batch_count));
  }
  return c;
}

Tensor sum(const Tensor &input, int axis, bool keep_dims) {
  if (input.dtype() != DType::Float32) {
    throw std::runtime_error("sum(axis): operation only supports Float32.");
  }
  if (input.numel() == 0) {
    if (keep_dims)
      return Tensor(input.shape(), input.dtype());
    return Tensor({0}, input.dtype());
  }

  int ndim = input.ndim();
  int pos_axis = (axis < 0) ? (ndim + axis) : axis;
  if (pos_axis < 0 || pos_axis >= ndim) {
    throw std::runtime_error(
        "sum(axis): Axis " + std::to_string(axis) +
        " is out of bounds for tensor with ndim=" + std::to_string(ndim));
  }

  // -------- 1. Decompose dimensions for the kernel --------
  // The kernel views the tensor as [outer_dim, reduce_dim, inner_dim].
  int outer_dim = 1;
  for (int i = 0; i < pos_axis; ++i)
    outer_dim *= input.dim(i);
  int reduce_dim = input.dim(pos_axis);
  int inner_dim = 1;
  for (int i = pos_axis + 1; i < ndim; ++i)
    inner_dim *= input.dim(i);

  // -------- 2. Calculate output shape --------
  Shape out_shape_squeezed;
  for (int i = 0; i < ndim; ++i) {
    if (i != pos_axis)
      out_shape_squeezed.push_back(input.dim(i));
  }
  if (out_shape_squeezed.empty())
    out_shape_squeezed.push_back(1);

  Tensor out;
  if (keep_dims) {
    Shape out_shape_kept = input.shape();
    out_shape_kept[pos_axis] = 1;
    out = Tensor(out_shape_kept, input.dtype());
  } else {
    out = Tensor(out_shape_squeezed, input.dtype());
  }

  // -------- 3. Launch kernel --------
  launch_sum_axis_kernel(out.mutable_data<void>(), input.data<void>(),
                         outer_dim, reduce_dim, inner_dim);
  return out;
}

Tensor sum(const Tensor &t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("sum: Global sum only supports Float32.");
  }
  Tensor result({1}, DType::Float32);
  if (t.numel() == 0) {
    fill_(result, 0.0f);
    return result;
  }
  launch_global_sum_kernel(result.mutable_data<float>(), t.data<float>(),
                           t.numel());
  return result;
}

Tensor mean(const Tensor &t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("mean: operation only supports Float32.");
  }
  if (t.numel() == 0)
    return from_host<float>({NAN}, {1}); // Return NaN for mean of empty tensor.

  Tensor sum_t = sum(t);
  float inv_n = 1.0f / static_cast<float>(t.numel());
  Tensor scalar_n = from_host<float>({inv_n}, {1});
  return mul(sum_t, scalar_n);
}

Tensor relu(const Tensor &t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("relu: operation only supports Float32.");
  }
  Tensor out(t.shape(), t.dtype());
  if (t.numel() > 0) {
    launch_relu_forward_kernel(out.mutable_data<float>(), t.data<float>(),
                               t.numel());
  }
  return out;
}

Tensor relu_backward(const Tensor &in, const Tensor &grad) {
  if (in.dtype() != DType::Float32 || grad.dtype() != DType::Float32) {
    throw std::runtime_error("relu_backward: operation only supports Float32.");
  }
  Tensor out_grad(in.shape(), in.dtype());
  if (in.numel() > 0) {
    launch_relu_backward_kernel(out_grad.mutable_data<float>(),
                                in.data<float>(), grad.data<float>(),
                                in.numel());
  }
  return out_grad;
}

Tensor softmax(const Tensor &t) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("softmax: operation only supports Float32.");
  }
  if (t.ndim() == 0)
    return from_host<float>({1.0f}, {}); // Softmax of a scalar is 1.

  const auto &shape = t.shape();
  int64_t rows = numel_to_last_dim(shape);
  int64_t cols = shape.back();

  Tensor out(t.shape(), t.dtype());
  if (t.numel() > 0) {
    launch_softmax_kernel(out.mutable_data<float>(), t.data<float>(), rows,
                          cols);
  }
  return out;
}

Tensor one_hot(const Tensor &ids, int vocab_size) {
  if (ids.dtype() != DType::Int32) {
    throw std::runtime_error("one_hot: Expected an Int32 tensor for ids.");
  }
  if (ids.ndim() > 2) {
    throw std::runtime_error("one_hot: only supports 1D or 2D id tensors.");
  }

  Shape out_shape = ids.shape();
  out_shape.push_back(vocab_size);
  Tensor out(out_shape, DType::Float32);
  fill_(out, 0.0f);

  if (ids.numel() > 0) {
    launch_one_hot_kernel(out.mutable_data<float>(), ids.data<int>(),
                          ids.numel(), vocab_size);
  }
  return out;
}

Tensor reshape(const Tensor &t, const Shape &new_shape) {
  return t.reshape(new_shape);
}

Tensor scale(const Tensor &t, float factor) {
  if (t.dtype() != DType::Float32) {
    throw std::runtime_error("scale: operation only supports Float32.");
  }
  auto scalar = from_host<float>({factor}, {1});
  return mul(t, scalar);
}

Tensor transpose(const Tensor &t, const std::vector<int> &perm) {
  if (!is_valid_permutation(perm, t.ndim())) {
    throw std::runtime_error("transpose: Invalid permutation provided.");
  }

  // -------- 1. Calculate output shape and strides --------
  Shape out_shape(t.ndim());
  for (size_t i = 0; i < perm.size(); ++i) {
    out_shape[i] = t.dim(perm[i]);
  }
  if (t.numel() == 0) {
    return Tensor(out_shape, t.dtype());
  }

  auto in_strides = calculate_strides(t.shape());
  auto out_strides = calculate_strides(out_shape);

  // -------- 2. Calculate inverse permutation for the kernel --------
  // The key to an efficient kernel is to iterate over the output and use an
  // inverse permutation to find the corresponding input element. This ensures
  // coalesced writes to global memory.
  std::vector<int> inv_perm(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }

  // -------- 3. Prepare data and launch kernel --------
  Tensor out(out_shape, t.dtype());
  DeviceVector<int64_t> d_in_strides(in_strides);
  DeviceVector<int64_t> d_out_strides(out_strides);
  DeviceVector<int> d_inv_perm(inv_perm);

  if (t.dtype() == DType::Float32) {
    launch_transpose_permute_kernel(out.mutable_data<float>(), t.data<float>(),
                                    d_in_strides.get(), d_out_strides.get(),
                                    d_inv_perm.get(), t.ndim(), t.numel());
  } else {
    throw std::runtime_error("transpose: DType not supported yet.");
  }
  return out;
}

Tensor transpose(const Tensor &t) {
  if (t.ndim() != 2) {
    throw std::runtime_error(
        "transpose: Simple transpose is only for 2D tensors.");
  }
  return transpose(t, {1, 0});
}

Tensor slice(const Tensor &t,
             const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  if (t.ndim() != ranges.size())
    throw std::runtime_error("slice: Number of ranges must match tensor ndim.");

  Shape out_shape(t.ndim());
  std::vector<int64_t> offsets(t.ndim());
  for (size_t i = 0; i < t.ndim(); ++i) {
    offsets[i] = ranges[i].first;
    int64_t end = (ranges[i].second == -1) ? t.dim(i) : ranges[i].second;
    out_shape[i] = end - offsets[i];
  }

  Tensor out(out_shape, t.dtype());
  if (out.numel() == 0)
    return out;

  DeviceVector<int64_t> d_in_shape(t.shape());
  DeviceVector<int64_t> d_out_shape(out_shape);
  DeviceVector<int64_t> d_offsets(offsets);

  if (t.dtype() == DType::Float32) {
    launch_slice_forward_kernel(out.mutable_data<float>(), t.data<float>(),
                                d_in_shape.get(), d_out_shape.get(),
                                d_offsets.get(), out.numel(), t.ndim());
  } else {
    throw std::runtime_error("slice: DType not supported yet.");
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Complex Ops (LayerNorm, Embedding, etc.)
//===----------------------------------------------------------------------===//

std::tuple<Tensor, Tensor, Tensor> layer_norm_forward(const Tensor &x,
                                                      const Tensor &gain,
                                                      const Tensor &bias,
                                                      float eps) {
  if (x.dtype() != DType::Float32)
    throw std::runtime_error("layer_norm_forward: only supports Float32.");

  auto [rows, cols] = get_norm_dims(x.shape());
  if (cols != gain.numel() || cols != bias.numel()) {
    throw std::runtime_error("LayerNorm feature dimension mismatch.");
  }

  Tensor out(x.shape(), DType::Float32);
  Tensor mean({rows}, DType::Float32);
  Tensor inv_std({rows}, DType::Float32);

  launch_fused_layer_norm_forward_kernel(
      out.mutable_data<float>(), mean.mutable_data<float>(),
      inv_std.mutable_data<float>(), x.data<float>(), gain.data<float>(),
      bias.data<float>(), rows, cols, eps);

  return {std::move(out), std::move(mean), std::move(inv_std)};
}

std::tuple<Tensor, Tensor, Tensor>
layer_norm_backward(const Tensor &d_out, const Tensor &in, const Tensor &gain,
                    const Tensor &mean, const Tensor &inv_std) {
  if (d_out.dtype() != DType::Float32)
    throw std::runtime_error("layer_norm_backward: only supports Float32.");

  auto [rows, cols] = get_norm_dims(in.shape());

  Tensor dx(in.shape(), DType::Float32);
  // The gradients for gain and bias have the same shape as gain.
  Tensor d_gain(gain.shape(), DType::Float32);
  Tensor d_bias(gain.shape(), DType::Float32);

  // The backward kernel computes per-row gradients for gain/bias, which are
  // then summed. These temporary buffers hold the per-row results.
  Tensor temp_d_gain({rows, cols}, DType::Float32);
  Tensor temp_d_bias({rows, cols}, DType::Float32);

  launch_fused_layer_norm_backward_kernel(
      dx.mutable_data<float>(), d_gain.mutable_data<float>(),
      d_bias.mutable_data<float>(), temp_d_gain.mutable_data<float>(),
      temp_d_bias.mutable_data<float>(), d_out.data<float>(), in.data<float>(),
      gain.data<float>(), mean.data<float>(), inv_std.data<float>(), rows,
      cols);

  return {std::move(dx), std::move(d_gain), std::move(d_bias)};
}

Tensor embedding_forward(const Tensor &weights, const Tensor &ids) {
  if (weights.dtype() != DType::Float32 || ids.dtype() != DType::Int32) {
    throw std::runtime_error(
        "embedding_forward: requires Float32 weights and Int32 ids.");
  }

  int64_t batch_size = ids.dim(0);
  int64_t seq_len = ids.dim(1);
  int64_t d_model = weights.dim(1);
  Tensor out({batch_size, seq_len, d_model}, DType::Float32);

  launch_gather_forward_kernel(out.mutable_data<float>(), weights.data<float>(),
                               ids.data<int>(), batch_size, seq_len, d_model);
  return out;
}

Tensor embedding_backward(const Tensor &grad_out, const Tensor &ids,
                          const Shape &weights_shape) {
  if (grad_out.dtype() != DType::Float32 || ids.dtype() != DType::Int32) {
    throw std::runtime_error(
        "embedding_backward: requires Float32 grad_out and Int32 ids.");
  }

  Tensor weights_grad(weights_shape, DType::Float32);
  fill_(weights_grad,
        0.0f); // Important to zero out gradients before accumulation.

  int64_t batch_size = ids.dim(0);
  int64_t seq_len = ids.dim(1);
  int64_t d_model = weights_shape[1];

  launch_scatter_add_backward_kernel(weights_grad.mutable_data<float>(),
                                     grad_out.data<float>(), ids.data<int>(),
                                     batch_size, seq_len, d_model);
  return weights_grad;
}

Tensor sum_to(const Tensor &input, const Shape &target_shape) {
  if (input.dtype() != DType::Float32) {
    throw std::runtime_error("sum_to: operation only supports Float32.");
  }
  if (input.shape() == target_shape) {
    return input.clone();
  }

  // -------- 1. Prepare and validate shapes for reduction --------
  int in_ndim = input.ndim();
  Shape padded_target_shape = target_shape;
  while (padded_target_shape.size() < in_ndim) {
    padded_target_shape.insert(padded_target_shape.begin(), 1);
  }

  // -------- 2. Identify dimensions to be reduced --------
  std::vector<bool> h_is_dim_reduced(in_ndim);
  for (int i = 0; i < in_ndim; ++i) {
    if (input.dim(i) != padded_target_shape[i]) {
      if (padded_target_shape[i] != 1) {
        throw std::runtime_error(
            "sum_to: Mismatched dimensions must be 1 in target shape.");
      }
      h_is_dim_reduced[i] = true;
    } else {
      h_is_dim_reduced[i] = false;
    }
  }

  Tensor out(target_shape, input.dtype());
  if (out.numel() == 0)
    return out;

  // -------- 3. Calculate strides and transfer metadata to device --------
  auto in_strides = calculate_strides(input.shape());
  auto out_strides_for_in = calculate_strides(padded_target_shape);
  DeviceVector<int64_t> d_in_shape(input.shape());
  DeviceVector<int64_t> d_in_strides(in_strides);
  DeviceVector<int64_t> d_out_strides_for_in(out_strides_for_in);
  DeviceVector<bool> d_is_dim_reduced(h_is_dim_reduced);

  // -------- 4. Launch the reduction kernel --------
  launch_sum_to_kernel(out.mutable_data<float>(), input.data<float>(),
                       out.numel(), in_ndim, d_in_shape.get(),
                       d_in_strides.get(), d_out_strides_for_in.get(),
                       d_is_dim_reduced.get());
  return out;
}

std::tuple<Tensor, Tensor> cross_entropy(const Tensor &logits,
                                         const Tensor &targets) {
  if (logits.dtype() != DType::Float32 || targets.dtype() != DType::Int32) {
    throw std::runtime_error(
        "cross_entropy: requires Float32 logits and Int32 targets.");
  }
  if (logits.ndim() < 2) {
    throw std::runtime_error("cross_entropy: Logits must be at least 2D.");
  }

  int64_t rows = numel_to_last_dim(logits.shape());
  int64_t cols = logits.shape().back();
  if (targets.numel() != rows) {
    throw std::runtime_error(
        "cross_entropy: Shape mismatch between logits and targets.");
  }

  Tensor loss_per_row({rows}, DType::Float32);
  Tensor softmax_out(logits.shape(), DType::Float32);

  launch_fused_softmax_cross_entropy_forward_kernel(
      loss_per_row.mutable_data<float>(), softmax_out.mutable_data<float>(),
      logits.data<float>(), targets.data<int>(), rows, cols);

  // The final loss is the mean of the per-row losses.
  Tensor final_loss = mean(loss_per_row);
  return {std::move(final_loss), std::move(softmax_out)};
}

Tensor create_padding_mask(const Tensor &ids, int pad_id) {
  if (ids.dtype() != DType::Int32) {
    throw std::runtime_error("create_padding_mask: requires Int32 ids tensor.");
  }
  if (ids.ndim() != 2) {
    throw std::runtime_error(
        "create_padding_mask: currently expects a 2D [B, L] tensor.");
  }

  // Create a temporary 2D mask [B, L]
  Tensor mask2d(ids.shape(), DType::Float32);
  launch_create_padding_mask_kernel(mask2d.mutable_data<float>(),
                                    ids.data<int>(), ids.numel(), pad_id);

  // Reshape to [B, 1, 1, L] for broadcasting in attention mechanisms.
  int64_t batch_size = ids.dim(0);
  int64_t seq_len = ids.dim(1);
  return reshape(mask2d, {batch_size, 1, 1, seq_len});
}

Tensor create_causal_mask(int seq_len) {
  // Create a 2D mask on the host.
  std::vector<float> mask_data(seq_len * seq_len);
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      // In log-space for softmax, masked elements are -inf.
      // In value-space, they are 0. We assume log-space for attention.
      mask_data[i * seq_len + j] = (j > i) ? -1e9f : 0.0f;
    }
  }
  // Transfer to device and reshape for broadcasting.
  return from_host(mask_data, {1, 1, (int64_t)seq_len, (int64_t)seq_len});
}

void adam_update(float *param_data, const float *grad_data, float *m_data,
                 float *v_data, size_t n, float lr, float beta1, float beta2,
                 float eps, int t) {
  ::launch_adam_update_kernel(param_data, grad_data, m_data, v_data, n, lr,
                              beta1, beta2, eps, t);
}

} // namespace sylvan::tensor::ops
