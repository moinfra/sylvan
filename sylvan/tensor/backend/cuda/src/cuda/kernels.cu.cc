#include "kernels.h"
#include "macros.h" // 假设路径已改为 sylvan/common/macros.h

// ===========================================================================
// CUDA 核函数定义 (只存在于此.cu.cc文件中)
// ===========================================================================

// --- 逐元素操作 ---

__global__ void add_kernel(float* out, const float* in, size_t n_out, size_t n_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_out) {
        out[idx] += in[idx % n_in];
    }
}

__global__ void mul_kernel(float* out, const float* a, const float* b, size_t n_a, size_t n_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_a) { // 假设 a 是较大的张量
        out[idx] = a[idx] * b[idx % n_b];
    }
}

__global__ void fill_kernel(float* data, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void scale_kernel(float* data, float scale, float bias, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale + bias;
    }
}

// --- 矩阵操作 ---

// 将 TILE_DIM 定义为 .cu.cc 文件内的常量
constexpr int TRANSPOSE_TILE_DIM = 32;
__global__ void transpose_kernel(float* out, const float* in, int rows, int cols) {
    // __shared__ 告诉编译器这块内存是每个块私有的，并且在快速的片上内存中
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }

    __syncthreads(); // 等待块内所有线程都完成从全局内存的读取

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// --- 并行归约求和 ---

__global__ void sum_kernel(float* out, const float* in, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
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

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}


// ===========================================================================
// 核函数启动器 (Launchers)
// ===========================================================================

constexpr int THREADS_PER_BLOCK = 256;

void launch_add_kernel(float* out, const float* in, size_t n_out, size_t n_in) {
    if (n_out == 0) return;
    int blocks_per_grid = (n_out + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out, in, n_out, n_in);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_kernel(float* out, const float* a, const float* b, size_t n_a, size_t n_b) {
    if (n_a == 0) return;
    int blocks_per_grid = (n_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mul_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(out, a, b, n_a, n_b);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fill_kernel(float* data, float value, size_t n) {
    if (n == 0) return;
    int blocks_per_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(data, value, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_scale_kernel(float* data, float scale, float bias, size_t n) {
    if (n == 0) return;
    int blocks_per_grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scale_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(data, scale, bias, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_kernel(float* out, const float* in, int rows, int cols) {
    dim3 threads(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
    dim3 blocks( (cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM );
    transpose_kernel<<<blocks, threads>>>(out, in, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sum_kernel(float* out, const float* in, size_t n) {
    int threads = 512;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    blocks = std::min(blocks, 1024);

    size_t shared_mem_size = threads * sizeof(float);
    
    if (blocks > 1) {
        float* d_intermediate_sums;
        CUDA_CHECK(cudaMalloc(&d_intermediate_sums, blocks * sizeof(float)));

        sum_kernel<<<blocks, threads, shared_mem_size>>>(d_intermediate_sums, in, n);
        CUDA_CHECK(cudaGetLastError());

        int final_threads = 512;
        int final_blocks = 1;
        size_t final_shared_mem_size = final_threads * sizeof(float);
        sum_kernel<<<final_blocks, final_threads, final_shared_mem_size>>>(out, d_intermediate_sums, blocks);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_intermediate_sums));
    } else {
        sum_kernel<<<1, threads, shared_mem_size>>>(out, in, n);
        CUDA_CHECK(cudaGetLastError());
    }
}
