#pragma once

#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace sylvan::tensor {

using Shape = std::vector<int64_t>;

// 目前只关注CUDA设备
enum class Device {
    CUDA
};

// 自定义删除器，用于在 shared_ptr 销毁时调用 cudaFree
struct CudaDeleter {
    void operator()(void* ptr) const;
};

class Tensor {
public:
    // 构造函数：在GPU上分配内存
    Tensor(const Shape& shape, Device device = Device::CUDA);

    // 移动构造和移动赋值
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // 禁止拷贝，因为 Tensor 是独特的资源所有者
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 析构函数（由智能指针自动处理）
    ~Tensor();

    // --- 公共 API ---
    const Shape& shape() const { return shape_; }
    Device device() const { return device_; }
    size_t numel() const { return numel_; }
    int64_t dim(size_t i) const { return shape_.at(i); }
    size_t ndim() const { return shape_.size(); }

    // 获取底层数据指针（const版本）
    // 注意：只应由CUDA核函数或库（如cuBLAS）使用
    const float* data() const {
        return static_cast<const float*>(data_.get());
    }

    // 获取底层数据指针（非const版本）
    float* mutable_data() {
        return static_cast<float*>(data_.get());
    }

private:
    std::shared_ptr<void> data_;
    Shape shape_;
    size_t numel_;
    Device device_;
};

// 辅助函数，用于打印张量（调试用）
void print_tensor(const Tensor& t, const std::string& name = "");

} // namespace sylvan::tensor
