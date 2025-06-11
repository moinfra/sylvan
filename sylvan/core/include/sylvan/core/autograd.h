#pragma once

#include "sylvan/tensor/tensor.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sylvan::core {
/**
 * @brief Core Autograd engine and fundamental operation wrappers.
 *
 * This namespace defines the `Variable` class, which is the cornerstone
 * of the dynamic computation graph, and provides wrappers for tensor operations
 * that automatically build the backward graph (Autograd).
 */

class Variable;     // Forward declaration for Variable.
class GraphContext; // Forward declaration for GraphContext.

/**
 * @brief Represents a single step in the backward pass.
 *
 * Each `BackwardStep` holds a lambda function (`pass`) that defines how to compute
 * gradients for its inputs, given the gradient from its output. This lambda
 * captures necessary intermediate values from the forward pass.
 */
struct BackwardStep {
  std::function<void(Variable *)> pass; ///< The function that performs the backward calculation.
};

/**
 * @brief Represents a node in the dynamic computation graph.
 *
 * A `Variable` encapsulates a tensor (`data`), its gradient (`grad`),
 * pointers to its parent `Variable`s (`prev_`), and a `BackwardStep`
 * (`grad_fn`) that defines its backward pass behavior.
 */
class Variable {
public:
  tensor::Tensor data; ///< The actual tensor data held by this variable.
  tensor::Tensor grad; ///< The gradient of the loss with respect to this variable. Lazily initialized to zero.
  std::vector<Variable *> prev_; ///< Pointers to variables that are inputs to the operation that produced this variable.
  std::shared_ptr<BackwardStep> grad_fn; ///< The backward function to be called during backpropagation. Null if this is a leaf node or `requires_grad` is false.
  std::string op_name; ///< The name of the operation that created this variable (for debugging).
  bool requires_grad;  ///< Indicates if gradients should be computed for this variable.

  /**
   * @brief Constructs a Variable node.
   * @param data The tensor data for this variable (moved).
   * @param requires_grad True if gradients should be computed for this variable; false otherwise.
   * @param prev A vector of input variables to the operation that produced this variable.
   * @param grad_fn The backward function associated with the operation that produced this variable.
   * @param op_name A descriptive name for the operation.
   */
  explicit Variable(tensor::Tensor &&data, bool requires_grad = true,
                    std::vector<Variable *> prev = {},
                    std::shared_ptr<BackwardStep> grad_fn = nullptr,
                    std::string op_name = "");

  /**
   * @brief Initiates the backpropagation process from this variable.
   *
   * This method builds a topological sort of the computation graph leading
   * to this variable and then traverses it in reverse order, calling
   * `grad_fn->pass` for each node to compute and accumulate gradients.
   * Can only be called on variables that `requires_grad`.
   */
  void backward();
};

/**
 * @brief Performs element-wise addition of two variables (A + B).
 * @param ctx The GraphContext for creating new variables.
 * @param a The first input Variable.
 * @param b The second input Variable.
 * @return A new Variable representing the sum.
 */
Variable *add(GraphContext &ctx, Variable *a, Variable *b);

/**
 * @brief Performs matrix multiplication of two variables (A @ B).
 * @param ctx The GraphContext for creating new variables.
 * @param a The first input Variable (matrix).
 * @param b The second input Variable (matrix).
 * @return A new Variable representing the matrix product.
 */
Variable *matmul(GraphContext &ctx, Variable *a, Variable *b);

/**
 * @brief Computes the Mean Squared Error (MSE) loss between predictions and targets.
 * @param ctx The GraphContext for creating new variables.
 * @param pred The predicted values Variable.
 * @param target The true target values Variable.
 * @return A new scalar Variable representing the MSE loss.
 */
Variable *mse_loss(GraphContext &ctx, Variable *pred, Variable *target); // TODO: Implement MSE loss.

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function.
 * @param ctx The GraphContext for creating new variables.
 * @param x The input Variable.
 * @return A new Variable with ReLU applied.
 */
Variable *relu(GraphContext &ctx, Variable *x);

/**
 * @brief Applies the Softmax function along a specified axis.
 * @param ctx The GraphContext for creating new variables.
 * @param x The input Variable.
 * @param axis The dimension along which to apply softmax. Default is the last dimension (-1).
 * @return A new Variable with Softmax applied.
 */
Variable *softmax(GraphContext &ctx, Variable *x, int axis = -1);

/**
 * @brief Performs Layer Normalization on the input variable.
 * @param ctx The GraphContext for creating new variables.
 * @param x The input Variable to normalize.
 * @param gain The learnable gain parameter (gamma).
 * @param bias The learnable bias parameter (beta).
 * @param eps A small value added to the variance for numerical stability.
 * @return A new Variable representing the normalized output.
 */
Variable *layer_norm(GraphContext &ctx, Variable *x, Variable *gain,
                     Variable *bias, float eps = 1e-5f);

/**
 * @brief Computes the Cross-Entropy Loss between logits and targets.
 * @param ctx The GraphContext for creating new variables.
 * @param logits The raw model output (logits).
 * @param targets The true class indices (Int32).
 * @return A new scalar Variable representing the cross-entropy loss.
 */
Variable *cross_entropy_loss(GraphContext &ctx, Variable *logits,
                             Variable *targets);

/**
 * @brief Reshapes a variable to a new shape.
 * @param ctx The GraphContext for creating new variables.
 * @param v The input Variable.
 * @param new_shape The target shape.
 * @return A new Variable representing the reshaped tensor (zero-copy view if possible).
 */
Variable *reshape(GraphContext &ctx, Variable *v,
                  const tensor::Shape &new_shape);

/**
 * @brief Transposes the dimensions of a variable.
 * @param ctx The GraphContext for creating new variables.
 * @param v The input Variable.
 * @param perm A vector specifying the new order of dimensions.
 * @return A new Variable with permuted dimensions.
 */
Variable *transpose(GraphContext &ctx, Variable *v,
                    const std::vector<int> &perm);

/**
 * @brief Scales a variable by a constant factor.
 * @param ctx The GraphContext for creating new variables.
 * @param v The input Variable.
 * @param factor The scalar factor to multiply by.
 * @return A new Variable representing the scaled tensor.
 */
Variable *scale(GraphContext &ctx, Variable *v, float factor);

/**
 * @brief Extracts a slice from a variable.
 * @param ctx The GraphContext for creating new variables.
 * @param v The input Variable.
 * @param ranges A vector of pairs, where each pair {offset, length} defines the slice for a dimension.
 * @return A new Variable representing the sliced tensor.
 */
Variable *slice(GraphContext &ctx, Variable *v,
                const std::vector<std::pair<int64_t, int64_t>> &ranges);

/**
 * @brief Accumulates an incoming gradient into a target variable's gradient.
 *
 * This function handles broadcasting rules for gradient accumulation. If the
 * incoming gradient has a different shape (e.g., from a broadcasted forward op),
 * it is summed down to match the target variable's shape before accumulation.
 *
 * @param target_var The Variable whose gradient needs to be updated.
 * @param incoming_grad The gradient tensor to be accumulated.
 */
void accumulate_grad(Variable* target_var,
                     const tensor::Tensor &incoming_grad);

} // namespace sylvan::core
