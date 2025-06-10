#!/bin/bash

# Sylvan 项目最终配置脚本
# 1. 修复 Bzlmod 中的 CUDA 扩展错误
# 2. 创建 README.md
# 3. 创建 LICENSE 文件

echo "🚀 开始最终的项目配置..."

# --- 1. 修复 MODULE.bazel 文件 ---
echo "🔧 正在修复 MODULE.bazel 中的 CUDA 扩展调用错误..."
cat <<'EOF' > MODULE.bazel
# Sylvan 项目的模块定义文件

module(
    name = "sylvan",
    version = "0.1.0",
)

# --- Bazel 核心依赖 ---
bazel_dep(name = "bazel_skylib", version = "1.5.0")
bazel_dep(name = "rules_cc", version = "0.0.9")

# --- Google Test 依赖 ---
bazel_dep(name = "googletest", version = "1.14.0")

# --- CUDA 依赖 ---
bazel_dep(name = "rules_cuda", version = "0.2.4") # Bazel 将解析到你缓存中的版本

# --- 配置 CUDA 工具链 ---
# FIX: 使用 'local_toolchain' 而不是 'toolkit'
cuda = use_extension("@rules_cuda//cuda:extensions.bzl", "toolchain")
cuda.local_toolchain(
    name = "local_cuda", # 明确名称
    toolkit_path = "",    # 留空以自动检测
)
# FIX: 引用正确的仓库名称 'local_cuda'
use_repo(cuda, "local_cuda")

EOF

# --- 2. 创建 README.md 文件 ---
echo "📄 正在创建 README.md 文件..."
cat <<'EOF' > README.md
# Sylvan 🌲

![Build Status](https://github.com/pluveto/sylvan/actions/workflows/ci.yml/badge.svg)

Sylvan is an educational, modern C++ deep learning framework built from scratch. It leverages CUDA for high-performance GPU computing and Bazel for a robust build system. The primary goal of this project is to gain a deep, practical understanding of how AI frameworks operate under the hood.

This project is developed with a "composition over inheritance" philosophy, favoring a functional-style API.

## Core Philosophy
- **Modern C++:** Utilizes C++17 features for clean, safe, and expressive code.
- **CUDA-First:** All core computations are designed to run on the GPU. No CPU fallback is planned to maintain focus.
- **Function-Style API:** Operations are free functions (`ops::add(a, b)`) rather than member functions (`a.add(b)`), promoting composition and testability.
- **No Inheritance for Layers/Ops:** Avoids complex class hierarchies.
- **Bazel with Bzlmod:** A modern, reproducible, and scalable build system.

## Features & Roadmap

- [x] Modern Bazel setup with Bzlmod
- [x] `sylvan_tensor` library for core tensor operations
- [ ] GPU Memory Management (Allocator/Pool)
- [ ] `sylvan_core` library with:
  - [ ] Dynamic Computation Graph
  - [ ] Autograd Engine
  - [ ] Basic Layers (FullyConnected, ReLU)
  - [ ] Optimizers (SGD)
- [ ] Convolutional Layers (Conv2D, MaxPooling) using cuDNN
- [ ] More Optimizers (Adam)
- [ ] `sylvan_infer` library for optimized inference
- [ ] Model serialization (saving/loading weights)

## Building the Project

This project uses Bazel. Ensure you have a recent version of Bazel and the NVIDIA CUDA Toolkit installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pluveto/sylvan.git
    cd sylvan
    ```

2.  **Sync dependencies (first time only):**
    ```bash
    bazel mod tidy
    ```

3.  **Build all targets:**
    All CUDA-related build flags are managed via the `.bazelrc` file.
    ```bash
    bazel build --config=cuda //...
    ```

4.  **Run all tests:**
    ```bash
    bazel test --config=cuda //...
    ```

## Directory Structure

- `sylvan/`: Main source code.
  - `tensor/`: The core tensor library. Doesn't know about autograd.
  - `core/`: The deep learning framework (autograd, layers, optimizers).
  - `infer/`: (Future) The inference-only library.
- `tests/`: Unit tests for all libraries (using GTest).
- `examples/`: Standalone examples showing how to use the framework.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
EOF

# --- 3. 创建 LICENSE 文件 ---
echo "⚖️  正在创建 MIT LICENSE 文件..."
cat <<'EOF' > LICENSE
MIT License

Copyright (c) 2024 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

echo ""
echo "✅ 项目配置修复和文档初始化完成！"
echo "--------------------------------------------------"
echo "请记得:"
echo "1. 修改 LICENSE 文件中的 `[Your Name or Organization]`."
echo "2. 修改 README.md 文件中的 GitHub 用户名和仓库链接."
echo ""
echo "现在，再次尝试运行构建命令:"
echo "   bazel build --config=cuda //..."
echo "--------------------------------------------------"
