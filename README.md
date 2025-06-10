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
    bazel run --config=cuda //examples:linear_regression
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
