#include "kernels.h"
#include "macros.h" // Assumed path is sylvan/common/macros.h
#include <cfloat>
#include <stdexcept>

#define SYLVAN_MAX_DIMS 8

//==============================================================================
// CUDA Kernel Definitions (Internal to this .cu.cc file)
//==============================================================================

// -------- Element-wise Kernels --------

/**
 * @brief Performs element-wise addition with broadcasting.
 *
 * Each thread computes `out[idx] += in[idx % n_in]`. The modulo operator
 * `idx % n_in` enables broadcasting when the input tensor `in` is smaller
 * than the output tensor `out`, effectively repeating `in` to match `out`'s
 * size.
 */
__global__ void add_kernel(float *out, const float *in, size_t n_out,
                           size_t n_in) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_out) {
    out[idx] += in[idx % n_in];
  }
}

/**
 * @brief Performs element-wise multiplication with broadcasting.
 *
 * Each thread computes `out[idx] = a[idx] * b[idx % n_b]`. This assumes `a` is
 * the larger tensor. The modulo operator on `b`'s index allows `b` to be
 * broadcast across `a`.
 */
__global__ void mul_kernel(float *out, const float *a, const float *b,
                           size_t n_a, size_t n_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_a) { // Assumes a is the larger tensor
    out[idx] = a[idx] * b[idx % n_b];
  }
}

/**
 * @brief Fills a tensor with a scalar value.
 *
 * Each thread is assigned a unique index and writes the given `value` to that
 * position in the `data` array, guarded by a bounds check.
 */
__global__ void fill_kernel(float *data, float value, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value;
  }
}

/**
 * @brief Applies a linear transformation (scale and bias) to a tensor.
 *
 * Each thread computes `data[idx] = data[idx] * scale + bias`. This is a
 * common operation in normalization layers.
 */
__global__ void scale_kernel(float *data, float scale, float bias, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * scale + bias;
  }
}

// -------- Matrix Operation Kernels --------

// Defines the side length of the square tile used for transposition.
// Using a tile in shared memory is a key optimization for transpose operations.
constexpr int TRANSPOSE_TILE_DIM = 32;

/**
 * @brief Transposes a 2D matrix using tiled shared memory.
 *
 * This kernel improves performance by minimizing global memory accesses.
 * The operation is performed in two stages:
 * 1. Load a tile of the input matrix from global memory into fast shared
 * memory. This access pattern is coalesced along rows.
 * 2. After all threads in the block have loaded their data (`__syncthreads`),
 *    write the data from the shared memory tile back to the output matrix,
 *    but with indices swapped. This write pattern is also coalesced.
 * This avoids the uncoalesced memory access that a naive transpose would cause.
 */
__global__ void transpose_kernel(float *out, const float *in, int rows,
                                 int cols) {
  // __shared__ allocates memory visible to all threads in a block, residing
  // in fast on-chip memory. An extra column is added to mitigate bank
  // conflicts.
  __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

  // Calculate global indices for reading from the input matrix.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Load element into the shared memory tile.
  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }

  // Wait for all threads in the block to finish loading into the tile.
  __syncthreads();

  // Recalculate global indices for writing to the transposed output matrix.
  // Note that blockIdx.x and blockIdx.y are swapped.
  x = blockIdx.y * blockDim.x + threadIdx.x;
  y = blockIdx.x * blockDim.y + threadIdx.y;

  // Write the transposed element from shared memory back to global memory.
  if (x < rows && y < cols) {
    out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
  }
}

// -------- Parallel Reduction Kernels --------

/**
 * @brief A single-pass reduction kernel for summing all elements of a tensor.
 *
 * This kernel implements a highly optimized parallel reduction.
 * 1. Grid-Stride Loop: Each thread processes multiple elements from the input
 *    array, striding by the total number of threads in the grid. This ensures
 *    that all data is processed regardless of input size and improves hardware
 *    utilization.
 * 2. Shared Memory Reduction: After summing its assigned elements into a
 *    register (`sum_val`), each thread writes its partial sum to shared memory.
 *    A parallel reduction is then performed within the shared memory array,
 *    halving the number of active threads in each step until a single sum for
 *    the entire block is computed.
 * 3. Output: Thread 0 of each block writes the block's total sum to a
 *    corresponding position in the output array.
 * This kernel produces partial sums; a second pass is needed for a final
 * result.
 */
__global__ void sum_kernel(float *out, const float *in, size_t n) {
  // `extern __shared__` allows dynamic allocation of shared memory at launch.
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;

  // Grid-stride loop for initial summation into a register.
  float sum_val = 0.0f;
  while (i < n) {
    sum_val += in[i];
    if (i + blockDim.x < n) {
      sum_val += in[i + blockDim.x];
    }
    i += gridSize;
  }
  sdata[tid] = sum_val;
  __syncthreads();

  // In-block reduction using shared memory.
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 writes the block's result to global memory.
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

// -------- ReLU Kernels --------

/**
 * @brief Applies the ReLU activation function element-wise.
 *
 * Each thread computes `out[i] = max(0.f, in[i])`.
 */
__global__ void relu_forward_kernel(float *out, const float *in, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = max(0.f, in[i]);
  }
}

/**
 * @brief Computes the gradient for the ReLU activation function.
 *
 * The derivative of ReLU is 1 if the input was > 0, and 0 otherwise.
 * This kernel applies the chain rule: the output gradient is the upstream
 * gradient if the original input was positive, and zero otherwise.
 */
__global__ void relu_backward_kernel(float *out_grad, const float *in_data,
                                     const float *upstream_grad, size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_grad[i] = (in_data[i] > 0.f) ? upstream_grad[i] : 0.f;
  }
}

// -------- Embedding Kernels --------

/**
 * @brief Gathers embedding vectors based on token IDs (forward pass).
 *
 * This kernel implements the embedding lookup. The parallelization strategy is:
 * - Grid: One block per item in the batch (`batch_size`).
 * - Block: One thread per dimension of the embedding vector (`d_model`).
 * Each thread is responsible for copying one dimension of the embedding for all
 * tokens in its assigned batch item. It iterates through the sequence
 * (`seq_len`), finds the `token_id`, and copies `weights[token_id * d_model +
 * d]` to the correct output position.
 */
__global__ void gather_forward_kernel(float *out, const float *weights,
                                      const int *ids, int batch_size,
                                      int seq_len, int d_model) {
  int b = blockIdx.x;  // batch index
  int d = threadIdx.x; // dimension index
  if (b < batch_size && d < d_model) {
    for (int s = 0; s < seq_len; ++s) {
      int token_id = ids[b * seq_len + s];
      int out_idx = (b * seq_len + s) * d_model + d;
      int weights_idx = token_id * d_model + d;
      out[out_idx] = weights[weights_idx];
    }
  }
}

/**
 * @brief Scatters and adds gradients back to the embedding weight matrix.
 *
 * This is the backward pass for the embedding layer. A key challenge is that
 * multiple tokens in the input batch might have the same ID, requiring their
 * gradients to be summed, not overwritten.
 * - Parallelization: The grid is sized to `total_tokens`, and the block is
 *   sized to `d_model`. Each thread handles one dimension of one token's
 * gradient.
 * - `atomicAdd`: This CUDA intrinsic is crucial. It performs a
 * read-modify-write operation on a memory location atomically, ensuring that
 * when multiple threads try to update the gradient for the same weight, the
 * updates are summed correctly without race conditions.
 */
__global__ void scatter_add_backward_kernel(
    float *weights_grad,   // Shape: [vocab_size, d_model]
    const float *out_grad, // Shape: [batch_size, seq_len, d_model]
    const int *ids,        // Shape: [batch_size, seq_len]
    size_t total_tokens,   // batch_size * seq_len
    size_t d_model) {
  int token_idx =
      blockIdx.x;            // Index in the flattened [batch*seq_len] dimension
  int dim_idx = threadIdx.x; // Index of the embedding dimension

  if (token_idx < total_tokens && dim_idx < d_model) {
    // 1. Get the vocabulary ID for the current token.
    int vocab_id = ids[token_idx];

    // 2. Get the upstream gradient value for this token's specific dimension.
    float grad_value = out_grad[token_idx * d_model + dim_idx];

    // 3. Calculate the index in the weight gradient matrix to update.
    int weight_grad_idx = vocab_id * d_model + dim_idx;

    // 4. Use atomicAdd to safely accumulate the gradient.
    atomicAdd(&weights_grad[weight_grad_idx], grad_value);
  }
}

// -------- Masking Kernels --------

/**
 * @brief Creates a causal (lower-triangular) mask.
 *
 * Used in decoders to prevent attention to future tokens. The grid and block
 * are both sized to `seq_len`, creating a 2D grid of threads. Each thread `(r,
 * c)` writes 1.0 if `c <= r` and 0.0 otherwise, forming a lower-triangular
 * matrix.
 */
__global__ void create_causal_mask_kernel(float *mask, int seq_len) {
  int r = blockIdx.x;  // Row
  int c = threadIdx.x; // Col
  if (r < seq_len && c < seq_len) {
    mask[r * seq_len + c] = (c <= r) ? 1.0f : 0.0f;
  }
}

/**
 * @brief Applies a mask to a 4D attention score tensor.
 *
 * This kernel sets elements of the `data` tensor to `mask_value` where the
 * corresponding `mask` element is 0.
 * - Parallelization: A 4D problem is mapped to a 2D grid. `blockIdx` covers
 *   batch and head dimensions, while `threadIdx` covers the query and key
 *   sequence lengths.
 * - Broadcasting: The 2D mask `(len_q, len_k)` is implicitly broadcast across
 *   the batch and head dimensions.
 */
__global__ void apply_mask_kernel(float *data, const float *mask,
                                  float mask_value, int batch, int heads,
                                  int len_q, int len_k) {
  int b = blockIdx.x;
  int h = blockIdx.y;
  int q = threadIdx.x;
  int k = threadIdx.y;

  if (b < batch && h < heads && q < len_q && k < len_k) {
    int data_idx =
        b * (heads * len_q * len_k) + h * (len_q * len_k) + q * len_k + k;
    // Assumes the same 2D mask is applied to all batches and heads.
    int mask_idx = q * len_k + k;
    if (mask[mask_idx] == 0.f) {
      data[data_idx] = mask_value;
    }
  }
}

// -------- Layer Normalization Kernels --------

/**
 * @brief Fused kernel for Layer Normalization forward pass.
 *
 * This kernel computes the entire LayerNorm operation for a single row within
 * one block, minimizing global memory traffic and avoiding intermediate
 * tensors. Each block is assigned one row of the input tensor.
 * 1. Mean Calculation: Threads in the block collaboratively calculate the sum
 *    of elements in their assigned row using a parallel reduction in shared
 *    memory. The final sum is divided by `cols` to get the mean.
 * 2. Inv Std Dev Calculation: Using the calculated mean, threads perform a
 *    second parallel reduction to find the sum of squared differences, which
 *    gives the variance. The inverse square root is then computed.
 * 3. Normalization: With the row's mean and inverse standard deviation now
 *    known by all threads in the block, each thread normalizes its assigned
 *    elements `(x - mean) * inv_std` and applies the learned gain and bias.
 */
__global__ void fused_layer_norm_forward_kernel(
    float *__restrict__ out,
    float *__restrict__ mean,    // Output for backward pass
    float *__restrict__ inv_std, // Output for backward pass
    const float *__restrict__ in, const float *__restrict__ gain,
    const float *__restrict__ bias, int cols, float eps) {
  int row = blockIdx.x;
  const float *row_in_ptr = in + row * cols;
  float *row_out_ptr = out + row * cols;

  extern __shared__ float s_data[];

  // -------- Step 1: Calculate mean (Parallel Reduction) --------
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += row_in_ptr[i];
  }
  s_data[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_data[threadIdx.x] += s_data[threadIdx.x + s];
    __syncthreads();
  }
  float row_mean = s_data[0] / cols;

  // -------- Step 2: Calculate inverse standard deviation (Parallel Reduction)
  // --------
  float sum_sq_diff = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float diff = row_in_ptr[i] - row_mean;
    sum_sq_diff += diff * diff;
  }
  s_data[threadIdx.x] = sum_sq_diff;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      s_data[threadIdx.x] += s_data[threadIdx.x + s];
    __syncthreads();
  }
  float row_inv_std = rsqrtf(s_data[0] / cols + eps);

  // -------- Save mean and inv_std for backward pass --------
  if (threadIdx.x == 0) {
    mean[row] = row_mean;
    inv_std[row] = row_inv_std;
  }

  // -------- Step 3: Apply Normalization (Fused Part) --------
  // All threads now apply the normalization to their assigned elements.
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    row_out_ptr[i] =
        (row_in_ptr[i] - row_mean) * row_inv_std * gain[i] + bias[i];
  }
}

/**
 * @brief Fused kernel for Layer Normalization backward pass.
 *
 * This kernel is highly complex and calculates the gradients for the input
 * `dx`, `d_gain`, and `d_bias`. Each block processes one row. The derivatives
 * are derived from the chain rule. To compute `dx`, several sums over the row
 * are needed (e.g., sum of `d_out * gain`).
 * 1. Parallel Reductions: The kernel performs three parallel reductions
 *    simultaneously using partitioned shared memory to efficiently compute the
 *    necessary intermediate sums for the `dx` calculation.
 * 2. Gradient Calculation: Using these reduced sums, each thread calculates
 *    the final gradient `dx` for its assigned elements.
 * 3. d_gain/d_bias: The gradients for `gain` and `bias` are also computed for
 *    the current row. Since these gradients must be summed across all rows
 *    (the batch dimension), they are written to temporary buffers. A subsequent
 *    column-wise reduction kernel (`sum_columns_kernel`) is used to perform
 *    the final summation.
 */
__global__ void fused_layer_norm_backward_kernel(
    float *__restrict__ dx,          // Gradient of input x
    float *__restrict__ d_gain,      // Gradient of gain (temp buffer)
    float *__restrict__ d_bias,      // Gradient of bias (temp buffer)
    const float *__restrict__ d_out, // Gradient from upstream (dl_dy)
    const float *__restrict__ in,    // The original input x
    const float *__restrict__ gain, const float *__restrict__ mean,
    const float *__restrict__ inv_std, int rows, int cols) {
  int row = blockIdx.x;

  const float *row_d_out = d_out + row * cols;
  const float *row_in = in + row * cols;
  float *row_dx = dx + row * cols;
  // Gradients for gain/bias are written to temporary storage per row.
  float *row_d_gain = d_gain + row * cols;
  float *row_d_bias = d_bias + row * cols;

  float current_mean = mean[row];
  float current_inv_std = inv_std[row];

  // Partition shared memory for 3 parallel reductions.
  extern __shared__ float s_data[];
  float *s_sum_dy_gain = s_data;
  float *s_sum_dy_gain_x_hat = s_data + blockDim.x;
  float *s_sum_dy = s_data + 2 * blockDim.x;

  // -------- Step 1: Compute local sums for reduction --------
  float sum_dy_gain = 0.f;
  float sum_dy_gain_x_hat = 0.f;
  float sum_dy = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float x_hat_i = (row_in[i] - current_mean) * current_inv_std;
    float d_out_i = row_d_out[i];
    float gain_i = gain[i];

    sum_dy_gain += d_out_i * gain_i;
    sum_dy_gain_x_hat += d_out_i * gain_i * x_hat_i;
    sum_dy += d_out_i;
  }
  s_sum_dy_gain[threadIdx.x] = sum_dy_gain;
  s_sum_dy_gain_x_hat[threadIdx.x] = sum_dy_gain_x_hat;
  s_sum_dy[threadIdx.x] = sum_dy;
  __syncthreads();

  // -------- Step 2: In-shared-memory reduction --------
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum_dy_gain[threadIdx.x] += s_sum_dy_gain[threadIdx.x + s];
      s_sum_dy_gain_x_hat[threadIdx.x] += s_sum_dy_gain_x_hat[threadIdx.x + s];
      s_sum_dy[threadIdx.x] += s_sum_dy[threadIdx.x + s];
    }
    __syncthreads();
  }
  float total_sum_dy_gain = s_sum_dy_gain[0];
  float total_sum_dy_gain_x_hat = s_sum_dy_gain_x_hat[0];

  // -------- Step 3: Compute gradients using the reduced sums --------
  float inv_N = 1.0f / cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float x_hat_i = (row_in[i] - current_mean) * current_inv_std;
    float d_out_i = row_d_out[i];
    float gain_i = gain[i];

    // Calculate dx using the pre-computed sums.
    float term1 = current_inv_std * d_out_i * gain_i;
    float term2 = -inv_N * current_inv_std * total_sum_dy_gain;
    float term3 = -inv_N * current_inv_std * x_hat_i * total_sum_dy_gain_x_hat;
    row_dx[i] = term1 + term2 + term3;

    // Calculate per-row d_gain and d_bias and store in temporary space.
    row_d_gain[i] = d_out_i * x_hat_i;
    row_d_bias[i] = d_out_i;
  }
}

/**
 * @brief Reduces a matrix by summing its columns.
 *
 * This kernel is used as the second step in the LayerNorm backward pass to sum
 * the temporary `d_gain` and `d_bias` gradients across the batch dimension.
 * - Tiling: It processes the input matrix in tiles. Each block loads a tile
 *   into shared memory.
 * - Reduction: The first row of threads (`threadIdx.y == 0`) in the block then
 *   sums the values down each column of the tile.
 * - Atomic Add: The final sum for each column segment is atomically added to
 *   the output vector to ensure correctness when multiple blocks contribute to
 *   the same output column.
 */
template <int TILE_DIM>
__global__ void sum_columns_kernel_template(float *__restrict__ out,
                                            const float *__restrict__ in,
                                            int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int tile_col_idx = blockIdx.x * TILE_DIM + threadIdx.x;
  float my_sum = 0.f;

  // Iterate over the rows in tile-sized chunks.
  for (int tile_row_start = 0; tile_row_start < rows;
       tile_row_start += TILE_DIM) {
    int in_row = tile_row_start + threadIdx.y;
    int in_col = tile_col_idx;

    // Load a tile from global to shared memory.
    if (in_row < rows && in_col < cols) {
      tile[threadIdx.y][threadIdx.x] = in[in_row * cols + in_col];
    } else {
      tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Let the first row of threads perform the column-wise sum within the tile.
    if (threadIdx.y == 0) {
      float col_sum_in_tile = 0.f;
      for (int i = 0; i < TILE_DIM; ++i) {
        col_sum_in_tile += tile[i][threadIdx.x];
      }
      my_sum += col_sum_in_tile;
    }
  }

  // Atomically add the final sum for this column to the output.
  if (threadIdx.y == 0 && tile_col_idx < cols) {
    atomicAdd(&out[tile_col_idx], my_sum);
  }
}

// -------- Miscellaneous Kernels --------

/**
 * @brief Converts integer class IDs to one-hot vectors.
 *
 * Each thread handles one row of the output. It reads the `class_id` from the
 * input `ids` array and writes a `1.0f` at the corresponding column index in
 * the output matrix. The output tensor is assumed to be pre-filled with zeros.
 */
__global__ void one_hot_kernel(float *__restrict__ out,
                               const int *__restrict__ ids, int rows,
                               int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    int class_id = ids[idx];
    if (class_id >= 0 && class_id < vocab_size) {
      out[idx * vocab_size + class_id] = 1.0f;
    }
  }
}

/**
 * @brief A generic N-dimensional tensor transpose/permutation kernel.
 *
 * This kernel permutes the dimensions of an input tensor `in` according to a
 * permutation map and writes the result to `out`. The core logic is index
 * manipulation. For each element in the output tensor:
 * 1. Deconstruct Linear Index: The thread's linear output index is converted
 *    into multi-dimensional coordinates `(d0, d1, ...)` based on the output
 *    tensor's strides.
 * 2. Permute Coordinates: The `inv_perm` (inverse permutation) array is used
 *    to map the output coordinates to the corresponding input coordinates.
 * 3. Reconstruct Index: The permuted coordinates are used with the input
 *    tensor's strides to calculate the source linear index.
 * 4. Copy Data: The value is copied from `in[in_linear_idx]` to
 *    `out[out_linear_idx]`.
 */
__global__ void transpose_permute_kernel(float *out, const float *in,
                                         const int64_t *d_in_strides,
                                         const int64_t *d_out_strides,
                                         const int *d_inv_perm, int ndim,
                                         size_t n) {
  int64_t out_linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_linear_idx >= n) {
    return;
  }

  // -------- Step 1: Deconstruct output linear index to multi-dim coords
  // --------
  int64_t out_coords[SYLVAN_MAX_DIMS];
  int64_t temp_idx = out_linear_idx;
  for (int i = 0; i < ndim; ++i) {
    out_coords[i] = temp_idx / d_out_strides[i];
    temp_idx %= d_out_strides[i];
  }

  // -------- Step 2: Use inverse permutation to find corresponding input coords
  // --------
  int64_t in_coords[SYLVAN_MAX_DIMS];
  for (int i = 0; i < ndim; ++i) {
    in_coords[i] = out_coords[d_inv_perm[i]];
  }

  // -------- Step 3: Reconstruct input linear index from its coords and strides
  // --------
  int64_t in_linear_idx = 0;
  for (int i = 0; i < ndim; ++i) {
    in_linear_idx += in_coords[i] * d_in_strides[i];
  }

  // -------- Step 4: Copy data from source to destination --------
  out[out_linear_idx] = in[in_linear_idx];
}

/**
 * @brief Extracts a slice from a source tensor.
 *
 * Each thread is responsible for one element in the output (sliced) tensor.
 * 1. Deconstruct Index: The thread's linear index in the output tensor is
 *    converted to multi-dimensional coordinates.
 * 2. Calculate Source Index: These coordinates are added to the `offsets`
 *    to find the corresponding coordinates in the larger input tensor.
 * 3. Reconstruct Index: The source coordinates are converted back to a linear
 *    index using the input tensor's strides/shape.
 * 4. Copy Data: The value is copied from the calculated source index to the
 *    output index.
 */
__global__ void slice_forward_kernel(float *out, const float *in,
                                     const int64_t *in_shape,
                                     const int64_t *out_shape,
                                     const int64_t *offsets, size_t out_numel,
                                     int ndim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_numel)
    return;

  int64_t out_coords[SYLVAN_MAX_DIMS] = {0};
  int64_t temp_idx = idx;

  // Deconstruct output linear index into logical coordinates.
  for (int i = 0; i < ndim; ++i) {
    int64_t stride = 1;
    for (int j = i + 1; j < ndim; ++j) {
      stride *= out_shape[j];
    }
    out_coords[i] = temp_idx / stride;
    temp_idx %= stride;
  }

  // Construct input linear index from output coords, offsets, and input shape.
  int64_t in_idx = 0;
  int64_t in_stride_val = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    in_idx += (out_coords[i] + offsets[i]) * in_stride_val;
    in_stride_val *= in_shape[i];
  }

  out[idx] = in[in_idx];
}

/**
 * @brief Adds a gradient into a slice of a larger gradient tensor (backward
 * pass).
 *
 * This is the reverse of slicing. It adds the `grad_in` tensor into the
 * `grad_out` tensor at the specified `offsets`.
 * - `atomicAdd` is used because multiple backward passes for different slices
 *   of the same tensor could attempt to write to the same memory locations,
 *   requiring their contributions to be summed.
 */
__global__ void
slice_backward_kernel(float *grad_out, const float *grad_in,
                      const int64_t *out_shape, // grad_out shape
                      const int64_t *in_shape,  // grad_in shape
                      const int64_t *offsets, int ndim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_shape[0] * in_shape[1] * in_shape[2])
    return; // Assume ndim <= 3

  int64_t in_coords[3] = {0};
  int64_t remaining_idx = idx;

  // Deconstruct input grad index into coords
  if (ndim > 2) {
    in_coords[0] = remaining_idx / (in_shape[1] * in_shape[2]);
    remaining_idx %= (in_shape[1] * in_shape[2]);
  }
  if (ndim > 1) {
    in_coords[1] = remaining_idx / in_shape[2];
    remaining_idx %= in_shape[2];
  }
  in_coords[ndim - 1] = remaining_idx;

  // Construct output grad index from input coords and offsets
  int64_t out_idx = 0;
  if (ndim > 2)
    out_idx += (in_coords[0] + offsets[0]) * (out_shape[1] * out_shape[2]);
  if (ndim > 1)
    out_idx += (in_coords[1] + offsets[1]) * out_shape[2];
  out_idx += (in_coords[ndim - 1] + offsets[ndim - 1]);

  atomicAdd(&grad_out[out_idx], grad_in[idx]);
}

/**
 * @brief Performs element-wise addition supporting NumPy-style broadcasting.
 *
 * This generic kernel handles addition between tensors of different shapes,
 * as long as they are broadcast-compatible.
 * 1. Index Deconstruction: Each thread takes a linear index from the *output*
 *    tensor and deconstructs it into multi-dimensional coordinates.
 * 2. Source Index Calculation: It then uses these coordinates along with the
 *    *strides* of each input tensor (`a` and `b`) to find the corresponding
 *    source elements. If a dimension in an input tensor was broadcast (size 1),
 *    its stride for that dimension will be 0, causing the same element to be
 *    read repeatedly, which is the essence of broadcasting.
 * 3. Addition: The located elements from `a` and `b` are added and written to
 *    the output.
 */
__global__ void broadcast_add_kernel(float *out, const float *a, const float *b,
                                     const int64_t *out_shape, int ndim,
                                     const int64_t *a_strides,
                                     const int64_t *b_strides) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  int64_t coords[SYLVAN_MAX_DIMS] = {0};
  int64_t temp_idx = idx;
  for (int i = ndim - 1; i >= 0; --i) {
    coords[i] = temp_idx % out_shape[i];
    temp_idx /= out_shape[i];
  }
  if (temp_idx > 0)
    return;

  // Compute input indices using broadcast-aware strides.
  int64_t a_idx = 0;
  int64_t b_idx = 0;
  for (int i = 0; i < ndim; ++i) {
    a_idx += coords[i] * a_strides[i];
    b_idx += coords[i] * b_strides[i];
  }

  out[idx] = a[a_idx] + b[b_idx];
}

/**
 * @brief Fused kernel for Softmax and Cross-Entropy Loss forward pass.
 *
 * This kernel is a major optimization that computes both the softmax activation
 * and the cross-entropy loss in a single pass, avoiding the materialization of
 * the full softmax output matrix just to compute the loss. Each block handles
 * one row.
 * 1. Stable Softmax (Log-Sum-Exp Trick): It first finds the maximum value in
 * the logit row using a parallel reduction. This max value is subtracted from
 * all logits before exponentiation to prevent numerical overflow (`expf`).
 * 2. Sum of Exponentials: It then computes the sum of the exponentiated values,
 *    again using a parallel reduction. This sum is the normalization factor.
 * 3. Loss Calculation: With the max logit and the sum of exponentials, the loss
 *    for the row can be calculated directly using the formula:
 *    `loss = log(sum_exp) - (logit_correct - max_val)`.
 *    Only thread 0 in the block performs this final calculation and write.
 * 4. Softmax Output: As a side effect, the normalized softmax probabilities are
 *    computed and written to `softmax_out` for use in the backward pass.
 */
__global__ void fused_softmax_cross_entropy_forward_kernel(
    float *__restrict__ loss_per_row, float *__restrict__ softmax_out,
    const float *__restrict__ logits, const int *__restrict__ targets, int rows,
    int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  int tid = threadIdx.x;
  const float *row_logits = logits + row * cols;
  float *row_softmax_out = softmax_out + row * cols;
  extern __shared__ float s_data[];

  // -------- Part 1: Find max logit for numerical stability --------
  float max_val = -FLT_MAX;
  for (int i = tid; i < cols; i += blockDim.x) {
    max_val = max(max_val, row_logits[i]);
  }
  s_data[tid] = max_val;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      s_data[tid] = max(s_data[tid], s_data[tid + s]);
    __syncthreads();
  }
  max_val = s_data[0];
  __syncthreads();

  // -------- Part 2: Compute sum of exps and the softmax output --------
  float sum_exp = 0.f;
  for (int i = tid; i < cols; i += blockDim.x) {
    float val = expf(row_logits[i] - max_val);
    row_softmax_out[i] = val; // Store un-normalized exp
    sum_exp += val;
  }
  s_data[tid] = sum_exp;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      s_data[tid] += s_data[tid + s];
    __syncthreads();
  }
  sum_exp = s_data[0];
  __syncthreads();

  // -------- Part 3: Finalize softmax and compute loss --------
  int target_id = targets[row];
  for (int i = tid; i < cols; i += blockDim.x) {
    row_softmax_out[i] /= sum_exp; // Finalize softmax probability
  }

  if (tid == 0) {
    float logit_correct = row_logits[target_id];
    loss_per_row[row] = logf(sum_exp) - (logit_correct - max_val);
  }
}

/**
 * @brief Computes the Softmax function for each row of a 2D tensor.
 *
 * This is a standalone softmax implementation. Each block processes one row.
 * The logic is identical to the first part of the fused cross-entropy kernel:
 * 1. Find the maximum value in the row via parallel reduction for stability.
 * 2. Subtract the max, exponentiate, and sum the results via a second parallel
 *    reduction to get the normalization constant.
 * 3. Divide each exponentiated element by the sum to get the final probability.
 */
__global__ void softmax_kernel(float *out, const float *in, int rows,
                               int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  int tid = threadIdx.x;
  const float *row_in = in + row * cols;
  float *row_out = out + row * cols;
  extern __shared__ float s_data[];

  // -------- 1. Find max value in the row for numerical stability --------
  float max_val = -FLT_MAX;
  for (int i = tid; i < cols; i += blockDim.x) {
    max_val = max(max_val, row_in[i]);
  }
  s_data[tid] = max_val;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      s_data[tid] = max(s_data[tid], s_data[tid + s]);
    __syncthreads();
  }
  max_val = s_data[0];
  __syncthreads();

  // -------- 2. Compute exp(x - max_val) and sum them up --------
  float sum_exp = 0.f;
  for (int i = tid; i < cols; i += blockDim.x) {
    sum_exp += expf(row_in[i] - max_val);
  }
  s_data[tid] = sum_exp;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      s_data[tid] += s_data[tid + s];
    __syncthreads();
  }
  sum_exp = s_data[0];
  __syncthreads();

  // -------- 3. Divide by the sum to get softmax probabilities --------
  if (sum_exp > 0) {
    for (int i = tid; i < cols; i += blockDim.x) {
      row_out[i] = expf(row_in[i] - max_val) / sum_exp;
    }
  } else {
    for (int i = tid; i < cols; i += blockDim.x) {
      row_out[i] = 1.0f / cols;
    }
  }
}

/**
 * @brief Performs a scatter-add operation, generalized for N dimensions.
 *
 * This kernel adds elements from a smaller tensor `in` to a larger tensor `out`
 * at specified `offsets`. It's essentially a generalized `slice_backward`.
 * Each thread processes one element of the *input* tensor. It calculates the
 * corresponding destination index in the *output* tensor and uses `atomicAdd`
 * to accumulate the value.
 */
__global__ void scatter_add_kernel(float *out, const float *in,
                                   const int64_t *out_shape,
                                   const int64_t *in_shape,
                                   const int64_t *offsets, int ndim,
                                   size_t n_in) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_in)
    return;

  // Deconstruct index from the input tensor.
  int64_t in_coords[SYLVAN_MAX_DIMS] = {0};
  int64_t temp_idx = idx;
  for (int i = ndim - 1; i >= 0; --i) {
    in_coords[i] = temp_idx % in_shape[i];
    temp_idx /= in_shape[i];
  }

  // Construct index for the output tensor.
  int64_t out_idx = 0;
  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    out_idx += (in_coords[i] + offsets[i]) * stride;
    stride *= out_shape[i];
  }

  atomicAdd(&out[out_idx], in[idx]);
}

/**
 * @brief A simple kernel for in-place element-wise addition (a += b).
 *
 * Assumes `a` and `b` have the same number of elements `n`. Each thread
 * computes `a[idx] += b[idx]`.
 */
__global__ void elementwise_add_kernel(float *a, const float *b, size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] += b[idx];
  }
}

/**
 * @brief Reduces a tensor along a single specified axis.
 *
 * The tensor is conceptually reshaped into `[outer_dim, reduce_dim,
 * inner_dim]`. The kernel reduces along `reduce_dim`.
 * - Parallelization: Each block is assigned to compute one element of the
 *   output tensor, which corresponds to a "strip" of data in the input tensor.
 * - Reduction: Threads within the block iterate over the `reduce_dim` and sum
 *   the values. A final parallel reduction in shared memory is used to get the
 *   total sum for the strip, which thread 0 then writes to the output.
 */
__global__ void sum_axis_kernel(float *out, const float *in, int outer_dim,
                                int reduce_dim, int inner_dim) {
  int o = blockIdx.x / inner_dim;
  int i = blockIdx.x % inner_dim;
  if (o >= outer_dim || i >= inner_dim)
    return;

  float sum = 0.0f;
  // Iterate over the dimension to be reduced.
  for (int r = threadIdx.x; r < reduce_dim; r += blockDim.x) {
    int in_idx = o * (reduce_dim * inner_dim) + r * inner_dim + i;
    sum += in[in_idx];
  }

  // Shared memory reduction for the block's partial sums.
  extern __shared__ float s_data[];
  s_data[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_data[threadIdx.x] += s_data[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Thread 0 writes the final result for this strip.
  if (threadIdx.x == 0) {
    int out_idx = o * inner_dim + i;
    out[out_idx] = s_data[0];
  }
}

/**
 * @brief Performs in-place element-wise addition with broadcasting (`a += b`).
 *
 * This is the in-place version of `broadcast_add_kernel`. Each thread computes
 * its index based on the shape of the larger tensor `a`, then finds the
 * corresponding element in `b` using broadcast-aware strides, and performs
 * the in-place addition.
 */
__global__ void broadcast_add_inplace_kernel(float *a, const float *b, int ndim,
                                             const int64_t *a_shape,
                                             const int64_t *a_strides,
                                             const int64_t *b_strides) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  // Deconstruct linear index `idx` (from tensor `a`) into multi-dim coords.
  int64_t coords[SYLVAN_MAX_DIMS] = {0};
  int64_t temp_idx = idx;
  for (int i = 0; i < ndim; ++i) {
    int64_t stride = 1;
    for (int j = i + 1; j < ndim; ++j) {
      stride *= a_shape[j];
    }
    coords[i] = temp_idx / stride;
    temp_idx %= stride;
  }

  // Compute b's index using its broadcast-aware strides.
  int64_t b_idx = 0;
  for (int i = 0; i < ndim; ++i) {
    b_idx += coords[i] * b_strides[i];
  }

  a[idx] += b[b_idx];
}

/**
 * @brief Creates a padding mask from a tensor of token IDs.
 *
 * Each thread checks if the token ID at its assigned index is the `pad_id`.
 * It writes 0.0f to the mask if it is a pad token, and 1.0f otherwise.
 */
__global__ void create_padding_mask_kernel(float *mask, const int *ids,
                                           size_t n, int pad_id) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    mask[idx] = (ids[idx] == pad_id) ? 0.0f : 1.0f;
  }
}

/**
 * @brief A generic, highly optimized global reduction sum kernel.
 *
 * This is a more robust version of `sum_kernel`, intended for the public API.
 * It uses the same principles (grid-stride loop, shared memory reduction) but
 * is wrapped in a launcher that implements a two-pass reduction if necessary,
 * making it correct and efficient for any input size.
 */
__global__ void reduce_sum_kernel(float *out, const float *in, size_t n) {
  extern __shared__ float s_data[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;

  // Grid-stride loop allows a fixed-size grid to process arbitrary-sized data.
  float my_sum = 0.0f;
  while (i < n) {
    my_sum += in[i];
    if (i + blockDim.x < n) {
      my_sum += in[i + blockDim.x];
    }
    i += gridSize;
  }

  // Each thread's partial sum is written to shared memory.
  s_data[tid] = my_sum;
  __syncthreads();

  // Parallel reduction within the block's shared memory.
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 writes the block's total sum to the output array.
  if (tid == 0) {
    out[blockIdx.x] = s_data[0];
  }
}

/**
 * @brief Performs element-wise multiplication with NumPy-style broadcasting.
 *
 * This kernel is identical in principle to `broadcast_add_kernel`, but performs
 * multiplication instead of addition. It uses strides to handle broadcasting
 * between tensors of compatible but different shapes.
 */
__global__ void broadcast_mul_kernel(float *out, const float *a, const float *b,
                                     const int64_t *out_shape, int ndim,
                                     const int64_t *a_strides,
                                     const int64_t *b_strides) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  int64_t coords[SYLVAN_MAX_DIMS] = {0};
  int64_t temp_idx = idx;
  for (int i = ndim - 1; i >= 0; --i) {
    coords[i] = temp_idx % out_shape[i];
    temp_idx /= out_shape[i];
  }
  if (temp_idx > 0)
    return;

  int64_t a_idx = 0;
  int64_t b_idx = 0;
  for (int i = 0; i < ndim; ++i) {
    a_idx += coords[i] * a_strides[i];
    b_idx += coords[i] * b_strides[i];
  }

  out[idx] = a[a_idx] * b[b_idx];
}

/**
 * @brief Helper device function to advance a coordinate "odometer".
 *
 * This function increments a multi-dimensional coordinate array, but only along
 * dimensions marked as `is_dim_reduced`. It's like an odometer that only turns
 * specific wheels. It's the core of the `sum_to_kernel` logic.
 * @return `true` if the coordinates were advanced, `false` if all combinations
 *         have been exhausted.
 */
__device__ bool advance_coords(int64_t *coords, const int64_t *shape,
                               const bool *is_dim_reduced, int ndim) {
  for (int i = ndim - 1; i >= 0; --i) {
    if (is_dim_reduced[i]) { // Only advance along reduced dimensions.
      coords[i]++;
      if (coords[i] < shape[i]) {
        return true; // Successfully advanced.
      }
      coords[i] = 0; // Current dimension overflowed, reset and carry over.
    }
  }
  return false; // All reducible dimensions have been fully traversed.
}

/**
 * @brief Reduces a tensor `in` to match the shape of `out`.
 *
 * This is the backward operation for broadcasting. For each element in the
 * smaller output tensor, this kernel iterates through all corresponding
 * elements in the larger input tensor and sums them up.
 * 1. Grid-Stride Loop: Each thread is responsible for one or more elements in
 *    the *output* tensor.
 * 2. Odometer Logic: For each output element, an "odometer"
 * (`current_in_coords`) is initialized. This odometer then iterates through all
 * source elements in the input tensor that contribute to the current output
 * element. The `advance_coords` function handles the multi-dimensional
 * iteration only along the axes that were broadcasted (and now need to be
 * reduced).
 * 3. Summation: The values from the input tensor are summed into a register.
 * 4. Write Result: The final sum is written to the output tensor.
 */
__global__ void sum_to_kernel(float *out, const float *in, size_t out_numel,
                              int in_ndim, const int64_t *in_shape,
                              const int64_t *in_strides,
                              const int64_t *out_strides_for_in,
                              const bool *is_dim_reduced) {
  // Grid-stride loop over the smaller output tensor.
  for (size_t out_linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
       out_linear_idx < out_numel; out_linear_idx += blockDim.x * gridDim.x) {

    // -------- 1. Decode output index to multi-dim coordinates --------
    int64_t out_coords[SYLVAN_MAX_DIMS];
    size_t temp_idx = out_linear_idx;
    for (int i = 0; i < in_ndim; ++i) {
      out_coords[i] = temp_idx / out_strides_for_in[i];
      temp_idx %= out_strides_for_in[i];
    }

    // -------- 2. Iterate and sum using the odometer --------
    float current_sum = 0.0f;
    int64_t current_in_coords[SYLVAN_MAX_DIMS];
    for (int i = 0; i < in_ndim; ++i) {
      current_in_coords[i] = is_dim_reduced[i] ? 0 : out_coords[i];
    }

    do {
      size_t in_linear_idx = 0;
      for (int i = 0; i < in_ndim; ++i) {
        in_linear_idx += current_in_coords[i] * in_strides[i];
      }
      current_sum += in[in_linear_idx];
    } while (
        advance_coords(current_in_coords, in_shape, is_dim_reduced, in_ndim));

    // -------- 3. Write the final sum --------
    out[out_linear_idx] = current_sum;
  }
}

/**
 * @brief Performs a single update step for the Adam optimizer.
 *
 * Each thread is responsible for one parameter. It performs the complete Adam
 * update logic for that parameter using a grid-stride loop for efficiency.
 * 1. Update Biased Moments: Computes the new first moment `m` (momentum) and
 *    second moment `v` (adaptive learning rate component).
 * 2. Bias Correction: Computes the bias-corrected moments `m_hat` and `v_hat`
 *    to account for the fact that `m` and `v` are initialized to zero.
 * 3. Parameter Update: Updates the parameter using the corrected moments,
 *    learning rate `lr`, and epsilon `eps` for numerical stability.
 * 4. Store Moments: Writes the new (biased) moments `m` and `v` back to
 *    global memory for the next iteration.
 */
__global__ void adam_update_kernel(float *__restrict__ param,
                                   const float *__restrict__ grad,
                                   float *__restrict__ m, float *__restrict__ v,
                                   size_t n, float lr, float beta1, float beta2,
                                   float eps, int t) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    const float g = grad[idx];

    // -------- Update biased first and second moment estimates --------
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // -------- Compute bias-corrected estimates --------
    float beta1_t = powf(beta1, t);
    float beta2_t = powf(beta2, t);
    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);

    // -------- Update parameters --------
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);

    // -------- Store new moments back to global memory --------
    m[idx] = m_new;
    v[idx] = v_new;
  }
}

//==============================================================================
// Kernel Launchers (Public API defined in kernels.h)
//==============================================================================

constexpr int THREADS_PER_BLOCK = 256;

void launch_add_kernel(float *out, const float *in, size_t n_out, size_t n_in) {
  if (n_out == 0)
    return;
  int blocks_per_grid = (n_out + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out, in, n_out, n_in);
  CUDA_CHECK(cudaGetLastError());
}

void launch_mul_kernel(float *out, const float *a, const float *b, size_t n_a,
                       size_t n_b) {
  if (n_a == 0)
    return;
  int blocks_per_grid = (n_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  mul_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out, a, b, n_a, n_b);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fill_kernel(float *data, float value, size_t n) {
  if (n == 0)
    return;
  int blocks_per_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  fill_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(data, value, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_scale_kernel(float *data, float scale, float bias, size_t n) {
  if (n == 0)
    return;
  int blocks_per_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scale_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(data, scale, bias, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_kernel(float *out, const float *in, int rows, int cols) {
  // Configure a 2D grid of 2D blocks for the tiled transpose.
  dim3 threads(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
  dim3 blocks((cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
              (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
  transpose_kernel<<<blocks, threads>>>(out, in, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void launch_relu_forward_kernel(float *out, const float *in, size_t n) {
  if (n == 0)
    return;
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(out, in, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward_kernel(float *out_grad, const float *in,
                                 const float *grad_in, size_t n) {
  if (n == 0)
    return;
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(out_grad, in, grad_in, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_gather_forward_kernel(float *out, const float *weights,
                                  const int *ids, size_t batch_size,
                                  size_t seq_len, size_t d_model) {
  // Launch one block per batch item, with threads covering the embedding
  // dimension.
  dim3 threads(d_model);
  dim3 blocks(batch_size);
  gather_forward_kernel<<<blocks, threads>>>(out, weights, ids, batch_size,
                                             seq_len, d_model);
  CUDA_CHECK(cudaGetLastError());
}

void launch_scatter_add_backward_kernel(float *weights_grad,
                                        const float *out_grad, const int *ids,
                                        size_t batch_size, size_t seq_len,
                                        size_t d_model) {
  size_t total_tokens = batch_size * seq_len;
  if (total_tokens == 0)
    return;

  // Launch one block per token, with threads covering the embedding dimension.
  dim3 threads(d_model);
  dim3 blocks(total_tokens);
  scatter_add_backward_kernel<<<blocks, threads>>>(weights_grad, out_grad, ids,
                                                   total_tokens, d_model);
  CUDA_CHECK(cudaGetLastError());
}

void launch_create_causal_mask_kernel(float *mask, int seq_len) {
  // Launch a 2D grid to create the 2D mask.
  dim3 threads(seq_len);
  dim3 blocks(seq_len);
  create_causal_mask_kernel<<<blocks, threads>>>(mask, seq_len);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_mask_kernel(float *data, const float *mask, float mask_value,
                              int batch_size, int num_heads, int seq_len_q,
                              int seq_len_k) {
  // Launch a 2D grid of 2D blocks to map to the 4D attention tensor.
  dim3 threads(seq_len_q, seq_len_k);
  dim3 blocks(batch_size, num_heads);
  apply_mask_kernel<<<blocks, threads>>>(data, mask, mask_value, batch_size,
                                         num_heads, seq_len_q, seq_len_k);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fused_layer_norm_forward_kernel(float *out, float *mean,
                                            float *inv_std, const float *in,
                                            const float *gain,
                                            const float *bias, int rows,
                                            int cols, float eps) {
  // Choose a power-of-2 number of threads for efficient reduction.
  int threads_per_row = (cols > 512) ? 512 : ((cols > 256) ? 256 : 128);
  // Launch one block per row.
  dim3 blocks(rows);
  dim3 threads(threads_per_row);
  fused_layer_norm_forward_kernel<<<blocks, threads,
                                    threads_per_row * sizeof(float)>>>(
      out, mean, inv_std, in, gain, bias, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}

template <int TILE_DIM>
void launch_sum_columns_kernel_template(float *out, const float *in, int rows,
                                        int cols) {
  CUDA_CHECK(cudaMemset(out, 0, cols * sizeof(float)));

  dim3 grid_dim((cols + TILE_DIM - 1) / TILE_DIM);
  // Block dimension is now based on the compile-time constant TILE_DIM
  dim3 block_dim(TILE_DIM, TILE_DIM);

  // No if/else needed! We are already inside a specific compiled version.
  sum_columns_kernel_template<TILE_DIM>
      <<<grid_dim, block_dim>>>(out, in, rows, cols);

  CUDA_CHECK(cudaGetLastError());
}

void launch_fused_layer_norm_backward_kernel(
    float *dx, float *d_gain, float *d_bias, float *temp_d_gain,
    float *temp_d_bias, const float *d_out, const float *in, const float *gain,
    const float *mean, const float *inv_std, int rows, int cols) {
  int threads_per_row = (cols > 512) ? 512 : 256;
  dim3 blocks(rows);
  dim3 threads(threads_per_row);
  // Allocate shared memory for three parallel reductions.
  size_t shared_mem_size = 3 * threads_per_row * sizeof(float);

  // First pass: Calculate dx and per-row gradients for gain and bias.
  fused_layer_norm_backward_kernel<<<blocks, threads, shared_mem_size>>>(
      dx, temp_d_gain, temp_d_bias, d_out, in, gain, mean, inv_std, rows, cols);
  CUDA_CHECK(cudaGetLastError());

  // Second pass: Reduce the temporary gradient buffers across the batch
  // dimension.
  launch_sum_columns_kernel_template<32>(d_gain, temp_d_gain, rows, cols);
  launch_sum_columns_kernel_template<32>(d_bias, temp_d_bias, rows, cols);
}

void launch_one_hot_kernel(float *out, const int *ids, int rows,
                           int vocab_size) {
  if (rows == 0)
    return;
  const int blocks_per_grid =
      (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  one_hot_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out, ids, rows,
                                                         vocab_size);
  CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_permute_kernel(float *out, const float *in,
                                     const int64_t *d_in_strides,
                                     const int64_t *d_out_strides,
                                     const int *d_inv_perm, int ndim,
                                     size_t n) {
  if (n == 0)
    return;
  const size_t blocks_per_grid =
      (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  transpose_permute_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      out, in, d_in_strides, d_out_strides, d_inv_perm, ndim, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_slice_forward_kernel(float *out, const float *in,
                                 const int64_t *d_in_shape,
                                 const int64_t *d_out_shape,
                                 const int64_t *d_offsets, size_t n, int ndim) {
  if (n == 0)
    return;
  const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  slice_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      out, in, d_in_shape, d_out_shape, d_offsets, n, ndim);
  CUDA_CHECK(cudaGetLastError());
}

void launch_broadcast_add_kernel(float *out, size_t out_numel, const float *a,
                                 const float *b, const int64_t *d_out_shape,
                                 int ndim, const int64_t *d_a_strides,
                                 const int64_t *d_b_strides) {
  if (out_numel == 0)
    return;
  const size_t blocks_per_grid =
      (out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  broadcast_add_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      out, a, b, d_out_shape, ndim, d_a_strides, d_b_strides);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fused_softmax_cross_entropy_forward_kernel(float *loss_per_row,
                                                       float *softmax_out,
                                                       const float *logits,
                                                       const int *targets,
                                                       int rows, int cols) {
  if (rows == 0 || cols == 0)
    return;
  int threads_per_block = 256;
  if (cols < 256)
    threads_per_block = 128;
  if (cols < 128)
    threads_per_block = 64;

  dim3 blocks(rows);
  dim3 threads(threads_per_block);
  size_t shared_mem_size = threads_per_block * sizeof(float);
  fused_softmax_cross_entropy_forward_kernel<<<blocks, threads,
                                               shared_mem_size>>>(
      loss_per_row, softmax_out, logits, targets, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_kernel(float *out, const float *in, int rows, int cols) {
  if (rows == 0 || cols == 0)
    return;
  int threads_per_block = 256;
  if (cols < 256)
    threads_per_block = 128;
  if (cols < 128)
    threads_per_block = 64;

  dim3 blocks(rows);
  dim3 threads(threads_per_block);
  size_t shared_mem_size = threads_per_block * sizeof(float);
  softmax_kernel<<<blocks, threads, shared_mem_size>>>(out, in, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void launch_scatter_add_kernel(float *out, const float *in,
                               const int64_t *d_out_shape,
                               const int64_t *d_in_shape,
                               const int64_t *d_offsets, int ndim,
                               size_t n_in) {
  if (n_in == 0)
    return;
  const int blocks = (n_in + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  scatter_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      out, in, d_out_shape, d_in_shape, d_offsets, ndim, n_in);
  CUDA_CHECK(cudaGetLastError());
}

void launch_elementwise_add_kernel(float *a, const float *b, size_t n) {
  if (n == 0)
    return;
  const size_t blocks_per_grid =
      (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  elementwise_add_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(a, b, n);
  CUDA_CHECK(cudaGetLastError());
}

void launch_sum_axis_kernel(void *out, const void *in, int outer_dim,
                            int reduce_dim, int inner_dim) {
  int num_strips = outer_dim * inner_dim;
  if (num_strips == 0)
    return;

  int threads_per_block = 256;
  if (reduce_dim < 256)
    threads_per_block = 128;
  if (reduce_dim < 128)
    threads_per_block = 64;

  dim3 blocks(num_strips);
  dim3 threads(threads_per_block);
  size_t shared_mem_size = threads_per_block * sizeof(float);
  sum_axis_kernel<<<blocks, threads, shared_mem_size>>>(
      static_cast<float *>(out), static_cast<const float *>(in), outer_dim,
      reduce_dim, inner_dim);
  CUDA_CHECK(cudaGetLastError());
}

void launch_broadcast_add_inplace_kernel(float *a, const float *b,
                                         size_t a_numel, int ndim,
                                         const int64_t *d_a_shape,
                                         const int64_t *d_a_strides,
                                         const int64_t *d_b_strides) {
  if (a_numel == 0)
    return;
  const size_t blocks_per_grid =
      (a_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  broadcast_add_inplace_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      a, b, ndim, d_a_shape, d_a_strides, d_b_strides);
  CUDA_CHECK(cudaGetLastError());
}

void launch_create_padding_mask_kernel(float *mask, const int *ids, size_t n,
                                       int pad_id) {
  if (n == 0)
    return;
  const size_t blocks_per_grid =
      (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  create_padding_mask_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(mask, ids,
                                                                     n, pad_id);
  CUDA_CHECK(cudaGetLastError());
}

void launch_global_sum_kernel(float *out, const float *in, size_t n) {
  if (n == 0) {
    CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
    return;
  }
  if (n == 1) {
    CUDA_CHECK(cudaMemcpy(out, in, sizeof(float), cudaMemcpyDeviceToDevice));
    return;
  }

  int threads_per_block = 256;
  size_t shared_mem_size = threads_per_block * sizeof(float);
  int num_blocks = (n + (threads_per_block * 2) - 1) / (threads_per_block * 2);
  num_blocks = std::min(num_blocks, 1024);

  if (num_blocks > 1) {
    // Two-pass reduction for large inputs.
    float *d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    // First pass: Reduce large input to a smaller array of partial sums.
    reduce_sum_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_partial_sums, in, n);
    CUDA_CHECK(cudaGetLastError());

    // Second pass: Reduce the partial sums to a single final value.
    reduce_sum_kernel<<<1, threads_per_block, shared_mem_size>>>(
        out, d_partial_sums, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_partial_sums));
  } else {
    // Single-pass reduction for smaller inputs that fit in one block.
    reduce_sum_kernel<<<1, threads_per_block, shared_mem_size>>>(out, in, n);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_broadcast_mul_kernel(float *out, size_t out_numel, const float *a,
                                 const float *b, const int64_t *d_out_shape,
                                 int ndim, const int64_t *d_a_strides,
                                 const int64_t *d_b_strides) {
  if (out_numel == 0)
    return;
  const size_t blocks_per_grid =
      (out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  broadcast_mul_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      out, a, b, d_out_shape, ndim, d_a_strides, d_b_strides);
  CUDA_CHECK(cudaGetLastError());
}

void launch_sum_to_kernel(float *out, const float *in, size_t out_numel,
                          int in_ndim, const int64_t *d_in_shape,
                          const int64_t *d_in_strides,
                          const int64_t *d_out_strides_for_in,
                          const bool *d_is_dim_reduced) {
  if (out_numel == 0)
    return;
  const size_t blocks_per_grid =
      (out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  sum_to_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      out, in, out_numel, in_ndim, d_in_shape, d_in_strides,
      d_out_strides_for_in, d_is_dim_reduced);
  CUDA_CHECK(cudaGetLastError());
}

void launch_adam_update_kernel(float *param_data, const float *grad_data,
                               float *m_data, float *v_data, size_t n, float lr,
                               float beta1, float beta2, float eps, int t) {
  if (n == 0)
    return;
  const size_t blocks_per_grid =
      (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  adam_update_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
      param_data, grad_data, m_data, v_data, n, lr, beta1, beta2, eps, t);
  CUDA_CHECK(cudaGetLastError());
}
