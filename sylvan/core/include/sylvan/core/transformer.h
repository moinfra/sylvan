// sylvan/core/transformer.h
//
// Defines the core components of the Transformer neural network architecture,
// including attention mechanisms, multi-head attention, feed-forward networks,
// encoder/decoder blocks, and the full Encoder-Decoder Transformer model.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include "autograd.h" 
#include "sylvan/core/layers/embedding.h"
#include "sylvan/core/layers/layernorm.h"
#include "sylvan/core/layers/linear.h"

#include <vector> 

namespace sylvan::core {

/**
 * @brief Performs Scaled Dot-Product Attention.
 *
 * This function calculates the attention scores between query (Q) and key (K)
 * matrices, scales them by the square root of the key dimension, applies an
 * optional mask, and then multiplies the resulting attention weights by the
 * value (V) matrix.
 *
 * Mathematically: `Attention(Q, K, V) = softmax((Q * K.T) / sqrt(d_k)) * V`
 *
 * @param ctx The graph context for tracing operations.
 * @param q The query tensor. Shape: `[..., seq_len_q, d_k]`.
 * @param k The key tensor. Shape: `[..., seq_len_k, d_k]`.
 * @param v The value tensor. Shape: `[..., seq_len_k, d_v]`.
 * @param mask An optional attention mask. Shape: `[..., seq_len_q, seq_len_k]`.
 *             Mask values should be 0 (for no masking) or -inf (for masking).
 *             If nullptr, no mask is applied.
 * @return A Variable containing the attention output. Shape: `[..., seq_len_q, d_v]`.
 */
Variable *scaled_dot_product_attention(GraphContext &ctx, Variable *q,
                                       Variable *k, Variable *v,
                                       Variable *mask = nullptr);

/**
 * @brief Implements Multi-Head Attention mechanism.
 *
 * This module projects queries, keys, and values `num_heads` times, performs
 * scaled dot-product attention in parallel for each head, and then concatenates
 * the results and projects them back into the original dimension.
 */
struct MultiHeadAttention {
  Linear w_q, w_k, w_v, w_o; ///< Linear layers for Q, K, V projections and final output projection.
  int num_heads;             ///< Number of attention heads.
  int d_model;               ///< The input and output feature dimension (d_model).

  /**
   * @brief Constructs a MultiHeadAttention module.
   *
   * @param d_model The input/output feature dimension. Must be divisible by `num_heads`.
   * @param num_heads The number of parallel attention heads.
   * @throws std::runtime_error if `d_model` is not divisible by `num_heads`.
   */
  MultiHeadAttention(int d_model, int num_heads);

  /**
   * @brief Performs the forward pass of Multi-Head Attention.
   *
   * Projects Q, K, V into `num_heads` sub-spaces, applies scaled dot-product attention
   * in parallel, concatenates results, and applies final linear projection.
   *
   * @param ctx The graph context.
   * @param q Query input tensor. Shape: `[batch_size, seq_len_q, d_model]`.
   * @param k Key input tensor. Shape: `[batch_size, seq_len_k, d_model]`.
   * @param v Value input tensor. Shape: `[batch_size, seq_len_k, d_model]`.
   * @param mask Optional attention mask. Shape: `[batch_size, 1, seq_len_q, seq_len_k]`
   *             or `[batch_size, seq_len_q, seq_len_k]` (broadcastable).
   * @return The output tensor after multi-head attention. Shape: `[batch_size, seq_len_q, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *q, Variable *k, Variable *v,
                    Variable *mask = nullptr);

  /**
   * @brief Returns a vector of learnable parameters for this module.
   *
   * Includes parameters from `w_q`, `w_k`, `w_v`, and `w_o`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();

private:
  /**
   * @brief Helper function to split the input tensor into multiple heads.
   *
   * Reshapes the input from `[batch_size, seq_len, d_model]` to
   * `[batch_size, num_heads, seq_len, d_k]`.
   * @param ctx The graph context.
   * @param x The input tensor.
   * @return The reshaped tensor.
   */
  Variable *split_heads(GraphContext &ctx, Variable *x);

  /**
   * @brief Helper function to combine outputs from multiple heads.
   *
   * Reshapes the input from `[batch_size, num_heads, seq_len_q, d_v]` to
   * `[batch_size, seq_len_q, d_model]`.
   * @param ctx The graph context.
   * @param x The input tensor.
   * @return The reshaped tensor.
   */
  Variable *combine_heads(GraphContext &ctx, Variable *x);
};

/**
 * @brief Implements a position-wise Feed-Forward Network (FFN).
 *
 * This module consists of two linear transformations with a ReLU activation
 * in between, applied independently to each position.
 * Mathematically: `FFN(x) = max(0, x * W1 + b1) * W2 + b2`.
 */
struct FeedForward {
  Linear layer1, layer2; ///< The two linear layers in the FFN.

  /**
   * @brief Constructs a FeedForward module.
   *
   * @param d_model The input and output feature dimension of the block.
   * @param d_ff The inner dimension of the feed-forward network.
   */
  FeedForward(int d_model, int d_ff);

  /**
   * @brief Performs the forward pass of the FeedForward network.
   *
   * Applies the first linear layer, ReLU activation, and then the second linear layer.
   *
   * @param ctx The graph context.
   * @param x The input tensor. Shape: `[batch_size, seq_len, d_model]`.
   * @return The output tensor. Shape: `[batch_size, seq_len, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *x);

  /**
   * @brief Returns a vector of learnable parameters for this module.
   *
   * Includes parameters from `layer1` and `layer2`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

/**
 * @brief Represents a single Encoder Block in the Transformer.
 *
 * An Encoder Block consists of a Multi-Head Self-Attention layer,
 * followed by a Feed-Forward Network, with residual connections and Layer Normalization
 * applied after each sub-layer.
 */
struct EncoderBlock {
  MultiHeadAttention self_attn; ///< Multi-head self-attention sub-layer.
  FeedForward ffn;              ///< Position-wise feed-forward network sub-layer.
  LayerNorm norm1, norm2;       ///< Layer Normalization layers for the two sub-layers.

  /**
   * @brief Constructs an EncoderBlock.
   *
   * @param d_model The input/output feature dimension.
   * @param num_heads The number of attention heads.
   * @param d_ff The inner dimension of the feed-forward network.
   */
  EncoderBlock(int d_model, int num_heads, int d_ff);

  /**
   * @brief Performs the forward pass of the Encoder Block.
   *
   * Applies self-attention, then a feed-forward network, each followed by
   * a residual connection and layer normalization.
   *
   * @param ctx The graph context.
   * @param x The input tensor to the encoder block. Shape: `[batch_size, seq_len, d_model]`.
   * @param src_mask The source padding mask for self-attention.
   *                 Shape: `[batch_size, 1, 1, seq_len]` (broadcastable).
   * @return The output tensor of the encoder block. Shape: `[batch_size, seq_len, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *x, Variable *src_mask);

  /**
   * @brief Returns a vector of learnable parameters for this block.
   *
   * Includes parameters from `self_attn`, `ffn`, `norm1`, and `norm2`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

/**
 * @brief Represents a single Decoder Block in the Transformer.
 *
 * A Decoder Block consists of a Masked Multi-Head Self-Attention layer,
 * a Multi-Head Cross-Attention layer (attending to encoder output),
 * followed by a Feed-Forward Network, with residual connections and Layer Normalization
 * applied after each sub-layer.
 */
struct DecoderBlock {
  MultiHeadAttention self_attn;  ///< Masked multi-head self-attention sub-layer.
  MultiHeadAttention cross_attn; ///< Multi-head attention over encoder output.
  FeedForward ffn;               ///< Position-wise feed-forward network sub-layer.
  LayerNorm norm1, norm2, norm3; ///< Layer Normalization layers for the three sub-layers.

  /**
   * @brief Constructs a DecoderBlock.
   *
   * @param d_model The input/output feature dimension.
   * @param num_heads The number of attention heads.
   * @param d_ff The inner dimension of the feed-forward network.
   */
  DecoderBlock(int d_model, int num_heads, int d_ff);

  /**
   * @brief Performs the forward pass of the Decoder Block.
   *
   * Applies masked self-attention, then cross-attention to encoder output,
   * then a feed-forward network, each followed by residual connections and layer normalization.
   *
   * @param ctx The graph context.
   * @param x The input tensor to the decoder block (target sequence). Shape: `[batch_size, tgt_seq_len, d_model]`.
   * @param enc_output The output from the encoder (source sequence representation).
   *                   Shape: `[batch_size, src_seq_len, d_model]`.
   * @param src_mask The source padding mask for cross-attention.
   *                 Shape: `[batch_size, 1, 1, src_seq_len]` (broadcastable).
   * @param tgt_mask The target (causal) mask for self-attention.
   *                 Shape: `[batch_size, 1, tgt_seq_len, tgt_seq_len]` (broadcastable).
   * @return The output tensor of the decoder block. Shape: `[batch_size, tgt_seq_len, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *x, Variable *enc_output,
                    Variable *src_mask, Variable *tgt_mask);

  /**
   * @brief Returns a vector of learnable parameters for this block.
   *
   * Includes parameters from `self_attn`, `cross_attn`, `ffn`, `norm1`, `norm2`, and `norm3`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

/**
 * @brief Implements the Encoder part of the Transformer model.
 *
 * The Encoder consists of an Embedding layer, a position encoding, and a stack
 * of `EncoderBlock`s. It processes the input source sequence.
 */
struct Encoder {
  Embedding embedding;          ///< Layer for converting token IDs to dense embeddings.
  Variable pos_encoding;        ///< Positional encoding tensor (not trained).
  std::vector<EncoderBlock> layers; ///< Stack of encoder blocks.

  /**
   * @brief Constructs an Encoder module.
   *
   * @param vocab_size The size of the source vocabulary.
   * @param d_model The embedding dimension and hidden size of the model.
   * @param num_heads The number of attention heads in each EncoderBlock.
   * @param num_layers The number of EncoderBlock layers.
   * @param d_ff The inner dimension of the feed-forward network in each EncoderBlock.
   * @param max_len The maximum sequence length for positional encoding.
   */
  Encoder(int vocab_size, int d_model, int num_heads, int num_layers, int d_ff,
          int max_len);

  /**
   * @brief Performs the forward pass of the Encoder.
   *
   * Embeds input IDs, adds positional encoding, and passes through all encoder blocks.
   *
   * @param ctx The graph context.
   * @param src_ids A Variable containing the source token IDs. Shape: `[batch_size, src_seq_len]`.
   * @param src_mask The source padding mask.
   *                 Shape: `[batch_size, 1, 1, src_seq_len]` (broadcastable).
   * @return The encoded representation of the source sequence.
   *         Shape: `[batch_size, src_seq_len, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *src_ids, Variable *src_mask);

  /**
   * @brief Returns a vector of learnable parameters for the Encoder.
   *
   * Includes parameters from the `embedding` layer and all `layers`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

/**
 * @brief Implements the Decoder part of the Transformer model.
 *
 * The Decoder consists of an Embedding layer, a position encoding, and a stack
 * of `DecoderBlock`s. It processes the target sequence and attends to the
 * encoder output.
 */
struct Decoder {
  Embedding embedding;          ///< Layer for converting token IDs to dense embeddings.
  Variable pos_encoding;        ///< Positional encoding tensor (not trained).
  std::vector<DecoderBlock> layers; ///< Stack of decoder blocks.

  /**
   * @brief Constructs a Decoder module.
   *
   * @param vocab_size The size of the target vocabulary.
   * @param d_model The embedding dimension and hidden size of the model.
   * @param num_heads The number of attention heads in each DecoderBlock.
   * @param num_layers The number of DecoderBlock layers.
   * @param d_ff The inner dimension of the feed-forward network in each DecoderBlock.
   * @param max_len The maximum sequence length for positional encoding.
   */
  Decoder(int vocab_size, int d_model, int num_heads, int num_layers, int d_ff,
          int max_len);

  /**
   * @brief Performs the forward pass of the Decoder.
   *
   * Embeds target IDs, adds positional encoding, and passes through all decoder blocks,
   * attending to the encoder output.
   *
   * @param ctx The graph context.
   * @param tgt_ids A Variable containing the target token IDs. Shape: `[batch_size, tgt_seq_len]`.
   * @param enc_output The output from the Encoder. Shape: `[batch_size, src_seq_len, d_model]`.
   * @param src_mask The source padding mask (for cross-attention).
   *                 Shape: `[batch_size, 1, 1, src_seq_len]` (broadcastable).
   * @param tgt_mask The target (causal) mask (for self-attention).
   *                 Shape: `[batch_size, 1, tgt_seq_len, tgt_seq_len]` (broadcastable).
   * @return The decoded representation of the target sequence.
   *         Shape: `[batch_size, tgt_seq_len, d_model]`.
   */
  Variable *forward(GraphContext &ctx, Variable *tgt_ids, Variable *enc_output,
                    Variable *src_mask, Variable *tgt_mask);

  /**
   * @brief Returns a vector of learnable parameters for the Decoder.
   *
   * Includes parameters from the `embedding` layer and all `layers`.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

/**
 * @brief Implements the full Encoder-Decoder Transformer model.
 *
 * This struct integrates the Encoder and Decoder modules and adds a final
 * linear projection layer to map the decoder's output to the target vocabulary size.
 * It is suitable for sequence-to-sequence tasks like machine translation.
 */
struct Transformer {
  Encoder encoder;         ///< The Transformer Encoder module.
  Decoder decoder;         ///< The Transformer Decoder module.
  Linear final_projection; ///< Final linear layer to project decoder output to vocabulary logits.

  /**
   * @brief Constructs a Transformer model.
   *
   * @param src_vocab_size The size of the source input vocabulary.
   * @param tgt_vocab_size The size of the target output vocabulary.
   * @param d_model The embedding dimension and hidden size of the model.
   * @param num_heads The number of attention heads in each block.
   * @param num_layers The number of encoder and decoder layers.
   * @param d_ff The inner dimension of the feed-forward network.
   * @param max_len The maximum sequence length for positional encoding.
   */
  Transformer(int src_vocab_size, int tgt_vocab_size, int d_model,
              int num_heads, int num_layers, int d_ff, int max_len);

  /**
   * @brief Performs the full forward pass of the Transformer model.
   *
   * Passes the source sequence through the encoder and the target sequence
   * through the decoder, using the encoder's output for cross-attention.
   * Finally, applies a linear projection to get vocabulary logits.
   *
   * @param ctx The graph context.
   * @param src_ids A Variable containing source token IDs. Shape: `[batch_size, src_seq_len]`.
   * @param tgt_ids A Variable containing target token IDs (decoder input). Shape: `[batch_size, tgt_seq_len]`.
   * @param src_mask The source padding mask.
   * @param tgt_mask The target (causal) mask.
   * @return A Variable containing the logits for the target vocabulary.
   *         Shape: `[batch_size, tgt_seq_len, tgt_vocab_size]`.
   */
  Variable *forward(GraphContext &ctx, Variable *src_ids, Variable *tgt_ids,
                    Variable *src_mask, Variable *tgt_mask);

  /**
   * @brief Returns a vector of all learnable parameters in the Transformer model.
   *
   * Includes parameters from the encoder, decoder, and final projection layer.
   * @return A `std::vector` of pointers to `Variable` parameters.
   */
  std::vector<Variable *> parameters();
};

} // namespace sylvan::core
