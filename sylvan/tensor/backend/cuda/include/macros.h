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
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS Error at %s:%d - Status %d\n", __FILE__,         \
              __LINE__, status);                                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(call)                                                     \
  do {                                                                         \
    curandStatus_t status = call;                                              \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuRAND Error at %s:%d - Status %d\n", __FILE__,         \
              __LINE__, status);                                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
