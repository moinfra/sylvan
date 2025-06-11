// sylvan/core/graph.h
//
// Defines the GraphContext class, which manages the computational graph
// for automatic differentiation. It utilizes an internal memory arena
// to efficiently allocate and manage Variable objects.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "sylvan/core/arena.h"
#include "sylvan/core/autograd.h"

namespace sylvan::core {

/**
 * @brief Manages the computational graph for automatic differentiation.
 *
 * `GraphContext` is responsible for creating and owning `Variable` objects
 * that form the nodes of the computation graph. It uses an internal `Arena`
 * for efficient memory allocation of `Variable`s, ensuring all associated
 * memory is deallocated when the `GraphContext` is destroyed or reset.
 *
 * This class adheres to RAII principles for memory management of `Variable`s.
 * It is non-copyable and non-movable to prevent issues with pointer invalidation
 * and double-freeing resources managed by the internal `Arena`.
 */
class GraphContext {
public:
  /**
   * @brief Constructs a new GraphContext.
   *
   * Initializes an empty computational graph.
   */
  GraphContext() = default;

  /**
   * @brief Destroys the GraphContext and deallocates all managed Variable objects.
   *
   * This destructor manually calls the destructors for all `Variable` objects
   * created within this context's `Arena`. This is crucial because `placement new`
   * was used for their construction, and the `Arena`'s memory is freed without
   * explicit destructor calls otherwise, leading to potential resource leaks
   * if `Variable`s hold complex resources. Variables are destructed in reverse
   * order of creation, which is a common practice.
   */
  ~GraphContext() {
    for (auto it = variables_.rbegin(); it != variables_.rend(); ++it) {
      (*it)->~Variable();
    }
  }

  /**
   * @brief Creates a new Variable within this GraphContext's memory arena.
   *
   * This templated method uses placement new to construct a `Variable`
   * directly within the `GraphContext`'s `Arena`. The created `Variable`
   * is tracked internally and its lifetime is managed by this context.
   *
   * @tparam Args Variadic template arguments for the Variable constructor.
   * @param args Arguments forwarded to the Variable constructor.
   * @return A pointer to the newly created `Variable` object.
   */
  template <typename... Args> Variable *create_variable(Args &&...args) {
    void *mem = arena_.allocate(sizeof(Variable));
    Variable *var = new (mem) Variable(std::forward<Args>(args)...);
    variables_.push_back(var); // Track the created Variable.
    return var;
  }

  // Delete copy constructor and copy assignment operator to prevent unintended
  // copying, which would lead to issues with the Arena-managed memory.
  GraphContext(const GraphContext &) = delete;
  GraphContext &operator=(const GraphContext &) = delete;
  // Delete move constructor and move assignment operator for consistency and
  // to avoid complex deep-copying logic or invalidating internal pointers.
  GraphContext(GraphContext &&) = delete;
  GraphContext &operator=(GraphContext &&) = delete;


  /**
   * @brief Resets the GraphContext, clearing all tracked Variables and freeing Arena memory.
   *
   * This method effectively clears the computational graph, making the context
   * ready for building a new graph. It manually destructs all `Variable`s
   * and then resets the underlying `Arena`.
   */
  void reset() {
    // Manually call destructors for all variables before resetting the arena.
    for (auto it = variables_.rbegin(); it != variables_.rend(); ++it) {
      (*it)->~Variable();
    }
    variables_.clear(); // Clear the tracking vector.
    arena_.reset();     // Reset the memory arena, freeing all memory.
  }

private:
  Arena arena_;                       ///< Memory arena for efficient Variable allocation.
  std::vector<Variable *> variables_; ///< Tracks all Variable objects created in this context.
};

} // namespace sylvan::core
