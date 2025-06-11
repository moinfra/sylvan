# Sylvan C++ Style Guide

This document outlines the coding and commenting standards for the Sylvan C++ project. Adhering to these guidelines is crucial for maintaining code readability, consistency, and maintainability across the entire codebase.

## 1. Naming Conventions

* **File Names:** `snake_case.h` for headers, `snake_case.cc` for source files.
  * *Good Example:* `tensor_ops.h`, `layer_norm.cc`
  * *Bad Example:* `TensorOps.h`, `layernorm.cpp`
* **Directory Names:** `snake_case`.
  * *Good Example:* `sylvan/core/layers`
  * *Bad Example:* `sylvan/Core/Layers`
* **Namespace Names:** `snake_case`. Nested namespaces are allowed.
  * *Good Example:* `sylvan::tensor::ops`
  * *Bad Example:* `Sylvan::TensorOps`
* **Class/Struct Names:** `PascalCase`.
  * *Good Example:* `LayerNorm`, `GraphContext`
  * *Bad Example:* `layer_norm`, `graph_context`
* **Function Names (member and free):** `snake_case`.
  * *Good Example:* `forward`, `create_variable`, `layer_norm_forward`
  * *Bad Example:* `Forward`, `CreateVariable`, `LayerNormForward`
* **Variable Names (local and member):** `snake_case`.
  * *Good Example:* `input_data`, `feature_dim`, `eps`
  * *Bad Example:* `InputData`, `featureDim`, `EPS`
* **Constants/Macros:** `ALL_CAPS_WITH_UNDERSCORES`.
  * *Good Example:* `MAX_BATCH_SIZE`, `PI`
  * *Bad Example:* `MaxBatchSize`, `pi`
* **Enum Names:** `PascalCase`.
  * *Good Example:* `DType`, `Kind`
  * *Bad Example:* `dtype`, `kind`
* **Enum Member Names:** `PascalCase` or `ALL_CAPS_WITH_UNDERSCORES` (consistency within an enum is key).
  * *Good Example:* `DType::Float32`, `DType::Int32`
  * *Good Example:* `Kind::Void`, `Kind::Integer`
  * *Bad Example:* `DType::float32`

## 2. Formatting

* **Indentation:** 4 spaces, no tabs.
* **Braces:** K&R style (opening brace on the same line as the statement, closing brace on its own line).
  * *Good Example:*

        ```cpp
        if (condition) {
            // ...
        } else {
            // ...
        }
        ```

  * *Bad Example:*

        ```cpp
        if (condition)
        {
            // ...
        }
        ```

* **Line Length:** Max 120 characters. Break long lines if necessary.
* **Spaces:**
  * Around binary operators (`+`, `-`, `*`, `/`, `=`, `==`, etc.).
  * After commas in argument lists.
  * After keywords like `if`, `for`, `while`, `switch`.
  * *Good Example:* `int a = b + c;`
  * *Bad Example:* `int a=b+c;`
* **Pointers and References:** Asterisk/ampersand next to the type.
  * *Good Example:* `const float* data_ptr;`, `Variable& var;`
  * *Bad Example:* `const float *data_ptr;`, `Variable &var;`
* **Include Guards:** Use `#pragma once`.
  * *Good Example:* `#pragma once`
  * *Bad Example:* `#ifndef MY_HEADER_H_`
* **Include Order:**
    1. Related header (e.g., `foo.h` for `foo.cc`).
    2. Other project headers.
    3. Third-party library headers.
    4. Standard library headers.
  * Grouped by blank lines.

## 3. Commenting Guidelines

This section provides detailed rules and examples for commenting. All new code and modified code **must** conform to these rules.

This section provides detailed rules and examples for commenting. All new code and modified code **must** conform to these rules.

### 3.1. File Header Comments

**Rule:**
Every `.h` and `.cc` file **must** include a standard file header comment providing essential information about the file.

**Good Example:**

```cpp
// sylvan/core/tensor/operators.cc
//
// Implements core tensor operations (e.g., element-wise, linear algebra,
// normalization) primarily optimized for CUDA-enabled GPUs.
// This file provides the concrete implementations for the abstract tensor
// operations declared in sylvan/tensor/operators.h.
//
// Author: Zijing Zhang
// Date: 2025-06-11
// Copyright: (c) 2025 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT
```

**Bad Example:**

```cpp
// operators.cc
// This is an implementation of tensor operations
// October 27, 2023
```

(Lacks detailed description, author, copyright, license, etc.)

### 3.2. Namespace Comments

**Rule:**
Namespaces should have a brief Doxygen-style comment describing their contained functionality or module.

**Good Example:**

```cpp
namespace sylvan::tensor::ops {
/**
 * @brief Provides a collection of fundamental tensor operations.
 *
 * This namespace encapsulates various tensor manipulation functions,
 * including element-wise operations, matrix multiplication,
 * normalization, and data transfer utilities. Operations are
 * primarily implemented using CUDA kernels for GPU acceleration.
 */
// ... code ...
} // namespace sylvan::tensor::ops
```

**Bad Example:**

```cpp
namespace sylvan::tensor::ops {
// Tensor operations namespace
// ... code ...
}
```

(Too brief, lacks detailed description)

### 3.3. Class/Struct Comments

**Rule:**
Every class or struct definition should be preceded by a Doxygen-style comment explaining its purpose, responsibilities, and key characteristics.

**Good Example:**

```cpp
/**
 * @brief Manages CUDA runtime and cuBLAS/cuRAND library handles.
 *
 * This RAII (Resource Acquisition Is Initialization) style struct ensures
 * proper creation and destruction of necessary CUDA library handles,
 * preventing resource leaks. It is designed as a singleton to provide
 * global access to these handles.
 */
struct CudaHandles {
    // ...
};
```

**Bad Example:**

```cpp
struct CudaHandles { // Manages CUDA handles
    // ...
};
```

(Insufficient information, not Doxygen-compliant)

### 3.4. Function Comments

**Rule:**
All `public` and `protected` functions (including constructors, destructors, member functions, and free functions) **must** have Doxygen-style comments. These comments should explain the function's purpose, parameters, return value, thrown exceptions, and any pre/post-conditions. `private` functions should also be commented if their logic is complex or if they have non-obvious side effects.

**Good Example:**

```cpp
/**
 * @brief Adds two tensors element-wise, supporting broadcasting.
 *
 * This function computes the sum of two tensors `a` and `b` according to
 * standard broadcasting rules. The resulting tensor `c` will have a shape
 * compatible with both input tensors.
 *
 * @param a The first input tensor (Float32).
 * @param b The second input tensor (Float32).
 * @return A new Tensor containing the element-wise sum.
 * @throws std::runtime_error if input tensors are not Float32 or
 *         are not broadcast-compatible.
 * @note This operation is out-of-place; a new tensor is allocated for the result.
 */
Tensor add(const Tensor &a, const Tensor &b);
```

**Bad Example:**

```cpp
Tensor add(const Tensor &a, const Tensor &b) { // Adds two tensors
    // ...
}
```

### 3.5. Member Variable Comments

**Rule:**
Member variables should be commented **only when their purpose is not immediately obvious** from their name or the context of the class. Avoid commenting on self-explanatory variables. When a comment is necessary, it should briefly explain the variable's role or any non-obvious characteristics.

**Good Example:**

```cpp
struct CudaHandles {
  cublasHandle_t cublas_handle;
  curandGenerator_t curand_gen;
  float eps;                      // Epsilon value for numerical stability
};
```

**Bad Example:**

```cpp
struct CudaHandles {
  cublasHandle_t cublas_handle;   ///< Handle for the cuBLAS library. // Redundant Doxygen for obvious name
  curandGenerator_t curand_gen;   ///< Generator for cuRAND operations. // Redundant Doxygen for obvious name
  float eps;                      // Epsilon value. // Too vague, "epsilon" could mean many things.
  int feature_dim;                // Feature dimension. // Redundant, name is clear.
};
```

### 3.6. Commenting Levels and Separators

This section defines a hierarchical system for comments and code block separators to enhance readability and organization.

* **Level 0: File/Major Class/Module Separator**
  * **Purpose:** To clearly delineate major independent classes, modules, or significant logical parts within a single `.cc` file. This is the highest level of separation within a file.
  * **Format:**

        ```cpp
        //===----------------------------------------------------------------------===//
        // ClassName/ModuleName
        //===----------------------------------------------------------------------===//
        ```

  * **Usage:** Place before the definition/implementation of a new major class, struct, or a distinct logical module within a source file. The descriptive text should be concise and centered.

* **Level 1: Major Logical Region Separator within a Function**
  * **Purpose:** To divide a large function into several distinct, high-level logical regions, especially when these regions themselves contain further nested logic or sub-steps. This indicates a significant shift in the function's flow or purpose.
  * **Format:** (NOTE: must eight `=`s)

        ```cpp
        // ======== Region Title ========
        ```

  * **Usage:** Place before a new major logical block within a function (e.g., "Data Preparation", "Forward Pass", "Backward Pass", "Optimization Step").

* **Level 2: Flat Logical Region Separator within a Function**
  * **Purpose:** To divide a function into logical regions where the sub-regions are relatively flat in their logical hierarchy, or to highlight a specific, important algorithm/step that doesn't require a Level 1 separator's visual weight.
  * **Format:** (NOTE: must eight `-`s)

        ```cpp
        // -------- Sub-Region Title / Algorithm Description --------
        ```

  * **Usage:** Place before a distinct sub-section of code within a Level 1 region, or for smaller functions where Level 1 is too heavy.

* **Level 3: Inline/Normal Comment**
  * **Purpose:** To explain code that is not immediately obvious, specific steps of an algorithm, design decisions, known limitations, or potential pitfalls. Avoid commenting on self-explanatory code or content already clearly expressed by the code itself.
  * **Format:**

        ```cpp
        // Brief explanation for non-obvious line/small block.
        ```

  * **Usage:** Use sparingly. Only when the code itself is not clear enough. Prioritize writing self-documenting code.

* **Level 4: No Comment**
  * For most of the code, which is self-explanatory, comments are unnecessary and distracting. Avoid excessive comments.
  * NO MORE than 10% percent of code should have comments (generally).

---

### 3.7. TODO/FIXME Comments

**Rule:**
Use `TODO:` or `FIXME:` tags to mark pending tasks, known issues, or future improvements. They should include a concise description and (optionally) the assignee's name and date.

**Good Example:**

```cpp
// TODO(JohnDoe): Refactor softmax backward to correctly handle arbitrary 'axis' values (2023-10-27).
// Currently assumes axis=-1 for simplicity.
```

**Bad Example:**

```cpp
// Improve later ...
// For now ...
```

(Unspecific, untraceable information)

### 3.8. Error Handling Comments

**Rule:**
Error `throw` statements should be accompanied by clear, concise, and sufficiently contextual error messages.

**Good Example:**

```cpp
if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("add: Operation requires Float32 tensors. "
                             "Got a.dtype()=" + to_string(a.dtype()) +
                             ", b.dtype()=" + to_string(b.dtype()) + ".");
}
```

**Bad Example:**

```cpp
if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32) {
    throw std::runtime_error("Type error"); // Insufficient information
}
```

### 3.9. Separators for Multiple Classes/Modules within a File

**Rule:**
When a `.cc` file implements **multiple** classes or modules, the following specific separator **must** be used to clearly delineate them.

**Good Example:**

```cpp
#include "ast_type.h"

namespace ast {

    //===----------------------------------------------------------------------===//
    // TypedField
    //===----------------------------------------------------------------------===//

    TypedField::TypedField(std::string name, TypePtr type)
        : name(std::move(name)), type(std::move(type)) {}

    // ... other TypedField implementations ...

    //===----------------------------------------------------------------------===//
    // Type
    //===----------------------------------------------------------------------===//
    TypePtr Type::create_placeholder() {
        return std::make_unique<PlaceholderType>();
    }

    // ... other Type implementations ...

    //===----------------------------------------------------------------------===//
    // SomeOtherModule
    //===----------------------------------------------------------------------===//
    void SomeOtherModule::do_something() {
        // ...
    }

} // namespace ast
```

**Bad Example:**

```cpp
namespace ast {

    // TypedField
    TypedField::TypedField(std::string name, TypePtr type)
        : name(std::move(name)), type(std::move(type)) {}

    // Type
    TypePtr Type::create_placeholder() {
        return std::make_unique<PlaceholderType>();
    }

} // namespace ast
```

(Separator does not conform to the specified format, poor visual separation)

### 3.10. Complex Logic Blocks

**Rule:**
For blocks of code implementing complex algorithms, non-trivial mathematical formulas, or critical performance optimizations, a concise comment block **must** precede the code. This comment should explain the **"why"** and **"how"** of the logic, including the underlying principle, the purpose of the block, or any important assumptions/trade-offs. It should aim to provide a high-level overview rather than line-by-line explanations. **For simple, straightforward code blocks, especially in example or demonstration files where the overall flow is the primary focus, such detailed comment blocks may be omitted if the logic is immediately apparent.**

**Good Example:**

```cpp
// This block implements the Jacobian-vector product for softmax backward pass.
// The formula used is: dL/dx = (dL/dy * y) - (y * sum(dL/dy * y, axis=last)).
// Note: This assumes the softmax was applied along the last dimension.
// Future work: Generalize to support arbitrary 'axis' for sum.
{
    const tensor::Tensor &upstream_grad = out_var->grad;
    tensor::Tensor s_mult_grad = tensor::ops::mul(s, upstream_grad);
    tensor::Tensor sum_s_grad =
        tensor::ops::sum(s_mult_grad, axis, /*keep_dims=*/true);
    tensor::Tensor sub_term = tensor::ops::mul(s, sum_s_grad);
    tensor::Tensor x_grad = tensor::ops::sub(s_mult_grad, sub_term);
    accumulate_grad(x, x_grad);
}
```

**Acceptable Example (for simple, self-explanatory blocks in examples/demos):**

```cpp
// Standard training steps
optimizer.zero_grad();
// Forward pass
auto *Y_pred = model.forward(graph, X);
// Calculate loss
auto *loss = core::mse_loss(graph, Y_pred, Y_true);
// Backward pass
loss->backward();
// Update parameters
optimizer.step();
```

*(Explanation: The sequence of `zero_grad`, `forward`, `loss`, `backward`, `step` is a standard deep learning training loop pattern. In a simple example, the "why" is implicit in the standard pattern, and the "how" is clear from the function names. A high-level comment like "Standard training steps" is sufficient, or even omitted if the context is very clear.)*

**Bad Example (Still bad, as it's redundant and clutters):**

```cpp
// Calculate softmax gradient
{
    const tensor::Tensor &upstream_grad = out_var->grad; // Upstream gradient
    tensor::Tensor s_mult_grad = tensor::ops::mul(s, upstream_grad); // Multiply s and upstream_grad
    // ... (rest of redundant line-by-line comments)
}
```

---

## 4. General Best Practices

* **RAII:** Resource Acquisition Is Initialization. Manage resources (memory, file handles, etc.) using RAII principles (e.g., `std::unique_ptr`, custom RAII wrappers).
* **Error Handling:** Use exceptions for exceptional conditions. Validate input parameters at the function boundaries.
* **Const Correctness:** Use `const` aggressively for parameters, member functions, and variables to ensure data integrity and enable compiler optimizations.
* **Avoid Global State:** Minimize the use of global variables. If necessary, encapsulate them within singletons or well-defined modules.
* **Smart Pointers:** Prefer `std::unique_ptr` for exclusive ownership and `std::shared_ptr` for shared ownership. Avoid raw pointers unless managing C-style arrays or interacting with C APIs.
* **`auto` Keyword:** Use `auto` when it improves readability (e.g., for complex iterator types) but avoid it when it obscures the type.
* **`override` and `final`:** Use `override` for virtual functions to catch errors at compile time. Use `final` for classes or virtual functions that should not be further derived/overridden.
* **`noexcept`:** Use `noexcept` for functions that are guaranteed not to throw exceptions.
* **`std::move` and `std::forward`:** Use correctly for efficient resource transfer.
* **Avoid `using namespace std;` in headers.** It pollutes the global namespace. In `.cc` files, it's generally acceptable within function scopes or at the top of the file.
* **Header Inclusion:** Include only what is necessary. Minimize transitive dependencies.
