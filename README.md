# Sylvan 🌲

🌍 [中文文档](项目介绍.md)

![Build Status](https://github.com/pluveto/sylvan/actions/workflows/ci.yml/badge.svg)

Sylvan is an educational, modern C++ deep learning framework built from scratch. It supports CUDA for high-performance GPU computing and Bazel for a robust build system. The main goal of this project is to help developers get a deep, practical understanding of how AI frameworks operate under the hood.

The core components of this framework are implemented in **under 5000 lines of C++/CUDA code**, making it a concise and approachable codebase for learning.

This project is developed with a "composition over inheritance" philosophy, favoring a functional-style API, which promotes modularity, reusability, and testability by avoiding complex class hierarchies and enabling easier combination of operations. The codebase is very easy to understand and maintain.

## Core Philosophy

- **Modern C++:** Utilizes modern C++ features for clean, safe, and expressive code.
- **CUDA-First:** All core computations are designed to run on the GPU. No CPU fallback is planned to maintain focus.
- **Function-Style API:** Operations are free functions (`ops::add(a, b)`) rather than member functions (`a.add(b)`), promoting composition and testability.
- **No Inheritance for Layers/Ops:** Avoids complex class hierarchies.
- **Bazel with Bzlmod:** A modern, reproducible, and scalable build system.

## Features & Roadmap

- [x] `sylvan_tensor` library for core tensor operations (creation, element-wise ops, matmul, sum, reshape, transpose, slice, fill, uniform initialization, ReLU, Softmax, LayerNorm, Embedding lookup)
- [ ] Advanced GPU Memory Management (Allocator/Pool) (Basic RAII via `std::shared_ptr` for `Tensor` data is implemented)
- [x] `sylvan_core` library with:
  - [x] Dynamic Computation Graph
  - [x] Autograd Engine (backward pass for all implemented ops)
  - [x] Basic Layers (Linear, ReLU, LayerNorm, Embedding)
  - [x] Attention Mechanisms (Multi-Head Attention, Scaled Dot-Product Attention)
  - [x] Transformer Architecture (Encoder, Decoder, Full Transformer)
  - [x] Optimizers (SGD, Adam)
- [ ] Convolutional Layers (Conv2D, MaxPooling) using cuDNN
- [ ] `sylvan_infer` library for optimized inference
- [ ] Model serialization (saving/loading weights)
- [ ] Dataloader (multiple formats, parallelly)

## Code Structure & Learning Focus

Sylvan's design prioritizes clarity and a hands-on understanding of deep learning internals. The codebase is extensively commented, especially in the core `sylvan/core` and `sylvan/tensor` directories. Each `Variable` operation, neural network layer, and GPU kernel includes detailed explanations of its purpose, parameters, and mathematical derivation. This focus on documentation aims to provide a clear roadmap for anyone looking to delve into the foundational concepts of modern AI frameworks.

## Dependencies

- CUDA Toolkit 11.0 or later (configured via `$CUDA_PATH`)
- Bazel 8.2.1 or later

## Building the Project

This project uses Bazel. Ensure you have a recent version of Bazel and the NVIDIA CUDA Toolkit installed.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/pluveto/sylvan.git
    cd sylvan
    ```

2. **Sync dependencies (first time only):**

    ```bash
    bazel mod tidy
    ```

3. **Build all targets:**
    All CUDA-related build flags are managed via the `.bazelrc` file.

    ```bash
    bazel build --config=cuda //...
    ```

4. **Run all tests:**

    ```bash
    bazel test --config=cuda //...
    ```

5. **Run an example:**

    ```bash
    # Run a linear regression example
    bazel run --config=cuda //examples:linear_regression
    # Run a transformer example
    bazel run --config=cuda //examples:number_translator
    ```

## Directory Structure

- `sylvan/`: Main source code.
  - `tensor/`: The core tensor library. Doesn't know about autograd.
  - `core/`: The deep learning framework (autograd, layers, optimizers).
  - `infer/`: (Future) The inference-only library.
- `tests/`: Unit tests for all libraries (using GTest).
- `examples/`: Standalone examples showing how to use the framework.

Run `CC=clang bazel run @hedron_compile_commands//:refresh_all` to generate a compilation database for your editor.

Install `Nsight Visual Studio Code Edition` for better debugging experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
