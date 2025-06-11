#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "\nCUDA Error at %s:%d\n", __FILE__, __LINE__);          \
      fprintf(stderr, "Code: %d, Name: %s, Description: %s\n", err,            \
              cudaGetErrorName(err), cudaGetErrorString(err));                 \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "\ncuBLAS Error at %s:%d - Status %d\n", __FILE__,       \
              __LINE__, status);                                               \
      throw std::runtime_error("cuBLAS error");                                \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(call)                                                     \
  do {                                                                         \
    curandStatus_t status = call;                                              \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuRAND Error at %s:%d - Status %d\n", __FILE__,         \
              __LINE__, status);                                               \
      throw std::runtime_error("cuRAND error");                                \
    }                                                                          \
  } while (0)
