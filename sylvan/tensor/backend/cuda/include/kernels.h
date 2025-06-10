#pragma once

#include <cstddef>

/** @brief Launch element-wise addition kernel (supports broadcasting) */
void launch_add_kernel(float *out, const float *in, size_t n_out, size_t n_in);

/** @brief Launch element-wise multiplication kernel (supports broadcasting) */
void launch_mul_kernel(float *out, const float *a, const float *b, size_t n_a,
                       size_t n_b);

/** @brief Launch fill kernel */
void launch_fill_kernel(float *data, float value, size_t n);

/** @brief Launch scale kernel (used after curand to adjust range) C = A * scale + bias */
void launch_scale_kernel(float *data, float scale, float bias, size_t n);

/** @brief Launch 2D matrix transpose kernel */
void launch_transpose_kernel(float *out, const float *in, int rows, int cols);

/** @brief Launch efficient parallel reduction sum kernel */
void launch_sum_kernel(float *out, const float *in, size_t n);
