#include "sylvan/tensor/operators.h"
#include "kernels.h" // 包含我们自己的核函数声明
#include "macros.h"
#include <cublas_v2.h>
#include <curand.h>
#include <stdexcept>

namespace sylvan::tensor::ops {

// ---------------------------------------------------------------------------
// CUDA 库句柄管理 (RAII 风格)
// ---------------------------------------------------------------------------
// 使用函数级静态变量确保句柄在首次使用时被创建，并在程序结束时自动销毁。
// 这是线程安全的 (C++11 及以后)。

struct CudaHandles {
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;

    CudaHandles() {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL)); // 使用固定种子以保证可复现性
    }

    ~CudaHandles() {
        CURAND_CHECK(curandDestroyGenerator(curand_gen));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
};

static CudaHandles& get_handles() {
    static CudaHandles handles;
    return handles;
}


// ---------------------------------------------------------------------------
// 操作实现
// ---------------------------------------------------------------------------

Tensor from_host(const std::vector<float>& data, const Shape& shape) {
    Tensor t(shape);
    if (!data.empty()) {
        if (t.numel() != data.size()) {
            throw std::runtime_error("Shape and data size mismatch in from_host.");
        }
        CUDA_CHECK(cudaMemcpy(t.mutable_data(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    return t;
}

std::vector<float> clone_to_host(const Tensor& t) {
    std::vector<float> host_data(t.numel());
    CUDA_CHECK(cudaMemcpy(host_data.data(), t.data(), t.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    return host_data;
}

// --- In-place ops ---

void fill_(Tensor& t, float value) {
    if (t.numel() == 0) return;
    launch_fill_kernel(t.mutable_data(), value, t.numel());
}

/**
 * @brief [In-place] Adds tensor b to tensor a. A = A  B.
 * @param a The tensor to be modified.
 * @param b The tensor to add. Supports broadcasting from a smaller tensor.
 */
void add_(Tensor& a, const Tensor& b) {
    if (a.numel() == 0) return;
    // Basic broadcasting check. The kernel handles the modulo arithmetic,
    // but we add a check for clarity on what is supported.
    if (a.numel() != b.numel() && b.numel() != 1 && (a.numel() % b.numel() != 0)) {
        // For now, we only officially support element-wise or adding a scalar.
    }
    launch_add_kernel(a.mutable_data(), b.data(), a.numel(), b.numel());
}

void uniform_(Tensor& t, float from, float to) {
    if (t.numel() == 0) return;
    CURAND_CHECK(curandGenerateUniform(get_handles().curand_gen, t.mutable_data(), t.numel()));
    // curandGenerateUniform 生成 [0, 1) 的随机数，我们需要将其缩放到 [from, to)
    launch_scale_kernel(t.mutable_data(), to - from, from, t.numel());
}

// --- Out-of-place ops ---

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel() && a.numel() != 1 && b.numel() != 1) {
        throw std::runtime_error("Add op requires same shape or one to be scalar for MVP.");
    }
    const Tensor& larger = (a.numel() >= b.numel()) ? a : b;
    const Tensor& smaller = (a.numel() >= b.numel()) ? b : a;

    Tensor c(larger.shape());
    // 先将大的张量内容拷贝到结果中
    CUDA_CHECK(cudaMemcpy(c.mutable_data(), larger.data(), larger.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // 调用核函数，将小的张量加到结果上（支持广播）
    launch_add_kernel(c.mutable_data(), smaller.data(), c.numel(), smaller.numel());
    return c;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor neg_b = mul(b, from_host({-1.0f}, {1}));
    return add(a, neg_b);
}


Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel() && a.numel() != 1 && b.numel() != 1) {
        throw std::runtime_error("Mul op requires same shape or one to be scalar for MVP.");
    }
    const Tensor& larger = (a.numel() >= b.numel()) ? a : b;
    const Tensor& smaller = (a.numel() >= b.numel()) ? b : a;
    
    Tensor c(larger.shape());
    launch_mul_kernel(c.mutable_data(), larger.data(), smaller.data(), c.numel(), smaller.numel());
    return c;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("Matmul only supports 2D tensors for now.");
    }
    if (a.dim(1) != b.dim(0)) {
        throw std::runtime_error("Matmul shape mismatch.");
    }

    int m = a.dim(0);
    int k = a.dim(1);
    int n = b.dim(1);

    Tensor c({(int64_t)m, (int64_t)n});
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Sylvan (row-major): C(m,n) = A(m,k) * B(k,n)
    // cuBLAS (col-major): C_T(n,m) = B_T(n,k) * A_T(k,m)
    // 我们传入B, A的指针，告诉cuBLAS它们已经是我们想要的列主序形式了 (B_T 和 A_T)
    // lda = k, ldb = n, ldc = n
    CUBLAS_CHECK(cublasSgemm(get_handles().cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             b.data(), n,
                             a.data(), k,
                             &beta,
                             c.mutable_data(), n));
    return c;
}

Tensor transpose(const Tensor& t) {
    if (t.ndim() != 2) {
        throw std::runtime_error("Transpose only supports 2D tensors for now.");
    }
    int m = t.dim(0); // rows
    int n = t.dim(1); // cols
    Tensor transposed({(int64_t)n, (int64_t)m});

    launch_transpose_kernel(transposed.mutable_data(), t.data(), m, n);
    return transposed;
}

Tensor sum(const Tensor& t) {
    if (t.numel() == 0) {
        return from_host({0.0f}, {1});
    }
    Tensor result({1});
    launch_sum_kernel(result.mutable_data(), t.data(), t.numel());
    return result;
}

Tensor mean(const Tensor& t) {
    if (t.numel() == 0) {
        return from_host({0.0f}, {1});
    }
    Tensor sum_t = sum(t);
    Tensor scalar_n = from_host({1.0f / static_cast<float>(t.numel())}, {1});
    return mul(sum_t, scalar_n);
}

} // namespace sylvan::tensor::ops
