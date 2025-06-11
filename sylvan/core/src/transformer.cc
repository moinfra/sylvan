// sylvan/core/transformer.cc
//
// Implements the core components of the Transformer neural network
// architecture, including attention mechanisms, multi-head attention,
// feed-forward networks, encoder/decoder blocks, and the full Encoder-Decoder
// Transformer model. This file provides the concrete implementations of the
// abstract interfaces declared in sylvan/core/transformer.h.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/transformer.h"
#include "sylvan/core/graph.h"
#include "sylvan/core/position.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sylvan::core {

Variable *scaled_dot_product_attention(GraphContext &ctx, Variable *q,
                                       Variable *k, Variable *v,
                                       Variable *mask) {
  int d_k = q->data.dim(q->data.ndim() - 1);
  float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

  // Compute Q * K.T
  auto k_T = transpose(
      ctx, k, {0, 1, 3, 2}); // Transpose last two dimensions for dot product
  auto scores = matmul(ctx, q, k_T);

  // Scale scores
  auto scaled_scores = scale(ctx, scores, scale_factor);

  // Apply mask if provided
  if (mask != nullptr) {
    scaled_scores = add(ctx, scaled_scores, mask);
  }

  // Apply softmax to get attention weights
  auto attention_weights = softmax(ctx, scaled_scores, -1);
  // Multiply weights by V
  return matmul(ctx, attention_weights, v);
}

//===----------------------------------------------------------------------===//
// MultiHeadAttention Implementation
//===----------------------------------------------------------------------===//

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : w_q(d_model, d_model), w_k(d_model, d_model), w_v(d_model, d_model),
      w_o(d_model, d_model), num_heads(num_heads), d_model(d_model) {
  if (d_model % num_heads != 0) {
    throw std::runtime_error("d_model must be divisible by num_heads");
  }
}

Variable *MultiHeadAttention::split_heads(GraphContext &ctx, Variable *x) {
  int64_t batch_size = x->data.dim(0);
  int64_t seq_len = x->data.dim(1);
  int64_t d_head = d_model / num_heads;
  // Reshape to [batch, seq_len, num_heads, d_head]
  auto reshaped =
      reshape(ctx, x, {batch_size, seq_len, (int64_t)num_heads, d_head});
  // Permute to [batch, num_heads, seq_len, d_head] for attention calculation
  return transpose(ctx, reshaped, {0, 2, 1, 3});
}

Variable *MultiHeadAttention::combine_heads(GraphContext &ctx, Variable *x) {
  // Permute back to [batch, seq_len, num_heads, d_head]
  auto permuted = transpose(ctx, x, {0, 2, 1, 3});
  int64_t batch_size = permuted->data.dim(0);
  int64_t seq_len = permuted->data.dim(1);
  // Reshape back to [batch, seq_len, d_model]
  return reshape(ctx, permuted, {batch_size, seq_len, (int64_t)d_model});
}

Variable *MultiHeadAttention::forward(GraphContext &ctx, Variable *q,
                                      Variable *k, Variable *v,
                                      Variable *mask) {
  // Linearly project Q, K, V
  auto q_proj = w_q.forward(ctx, q);
  auto k_proj = w_k.forward(ctx, k);
  auto v_proj = w_v.forward(ctx, v);

  // Split projected Q, K, V into multiple heads
  auto q_split = split_heads(ctx, q_proj);
  auto k_split = split_heads(ctx, k_proj);
  auto v_split = split_heads(ctx, v_proj);

  // Perform scaled dot-product attention for each head
  auto attn_output =
      scaled_dot_product_attention(ctx, q_split, k_split, v_split, mask);
  // Combine outputs from multiple heads
  auto combined = combine_heads(ctx, attn_output);
  // Final linear projection
  return w_o.forward(ctx, combined);
}

/**
 * @brief Returns a vector of learnable parameters for this module.
 *
 * Includes parameters from `w_q`, `w_k`, `w_v`, and `w_o`.
 * @return A `std::vector` of pointers to `Variable` parameters.
 */
std::vector<Variable *> MultiHeadAttention::parameters() {
  auto params = w_q.parameters();
  auto p_k = w_k.parameters();
  auto p_v = w_v.parameters();
  auto p_o = w_o.parameters();
  params.insert(params.end(), p_k.begin(), p_k.end());
  params.insert(params.end(), p_v.begin(), p_v.end());
  params.insert(params.end(), p_o.begin(), p_o.end());
  return params;
}

//===----------------------------------------------------------------------===//
// FeedForward Implementation
//===----------------------------------------------------------------------===//

FeedForward::FeedForward(int d_model, int d_ff)
    : layer1(d_model, d_ff), layer2(d_ff, d_model) {}

Variable *FeedForward::forward(GraphContext &ctx, Variable *x) {
  auto x1 = layer1.forward(ctx, x);
  auto relu_out = relu(ctx, x1);
  return layer2.forward(ctx, relu_out);
}

std::vector<Variable *> FeedForward::parameters() {
  auto params = layer1.parameters();
  auto p2 = layer2.parameters();
  params.insert(params.end(), p2.begin(), p2.end());
  return params;
}

//===----------------------------------------------------------------------===//
// EncoderBlock Implementation
//===----------------------------------------------------------------------===//

EncoderBlock::EncoderBlock(int d_model, int num_heads, int d_ff)
    : self_attn(d_model, num_heads), ffn(d_model, d_ff), norm1(d_model),
      norm2(d_model) {}

Variable *EncoderBlock::forward(GraphContext &ctx, Variable *x,
                                Variable *src_mask) {
  // Self-attention sub-layer with residual connection and LayerNorm
  auto attn_out = self_attn.forward(ctx, x, x, x, src_mask);
  auto x1 = norm1.forward(ctx, add(ctx, x, attn_out));

  // Feed-forward sub-layer with residual connection and LayerNorm
  auto ffn_out = ffn.forward(ctx, x1);
  auto x2 = norm2.forward(ctx, add(ctx, x1, ffn_out));
  return x2;
}

std::vector<Variable *> EncoderBlock::parameters() {
  auto params = self_attn.parameters();
  auto p_ffn = ffn.parameters();
  auto p_n1 = norm1.parameters();
  auto p_n2 = norm2.parameters();
  params.insert(params.end(), p_ffn.begin(), p_ffn.end());
  params.insert(params.end(), p_n1.begin(), p_n1.end());
  params.insert(params.end(), p_n2.begin(), p_n2.end());
  return params;
}

//===----------------------------------------------------------------------===//
// DecoderBlock Implementation
//===----------------------------------------------------------------------===//

DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff)
    : self_attn(d_model, num_heads), cross_attn(d_model, num_heads),
      ffn(d_model, d_ff), norm1(d_model), norm2(d_model), norm3(d_model) {}

Variable *DecoderBlock::forward(GraphContext &ctx, Variable *x,
                                Variable *enc_output, Variable *src_mask,
                                Variable *tgt_mask) {
  // Masked self-attention sub-layer
  auto self_attn_out = self_attn.forward(ctx, x, x, x, tgt_mask);
  auto x1 = norm1.forward(ctx, add(ctx, x, self_attn_out));

  // Cross-attention sub-layer (attending to encoder output)
  auto cross_attn_out =
      cross_attn.forward(ctx, x1, enc_output, enc_output, src_mask);
  auto x2 = norm2.forward(ctx, add(ctx, x1, cross_attn_out));

  // Feed-forward sub-layer
  auto ffn_out = ffn.forward(ctx, x2);
  auto x3 = norm3.forward(ctx, add(ctx, x2, ffn_out));
  return x3;
}

std::vector<Variable *> DecoderBlock::parameters() {
  auto params = self_attn.parameters();
  auto p_cross = cross_attn.parameters();
  auto p_ffn = ffn.parameters();
  auto p_n1 = norm1.parameters();
  auto p_n2 = norm2.parameters();
  auto p_n3 = norm3.parameters();
  params.insert(params.end(), p_cross.begin(), p_cross.end());
  params.insert(params.end(), p_ffn.begin(), p_ffn.end());
  params.insert(params.end(), p_n1.begin(), p_n1.end());
  params.insert(params.end(), p_n2.begin(), p_n2.end());
  params.insert(params.end(), p_n3.begin(), p_n3.end());
  return params;
}

//===----------------------------------------------------------------------===//
// Encoder Implementation
//===
Encoder::Encoder(int vocab_size, int d_model, int num_heads, int num_layers,
                 int d_ff, int max_len)
    : embedding(vocab_size, d_model),
      pos_encoding(create_positional_encoding(max_len, d_model),
                   /*requires_grad=*/false) {
  for (int i = 0; i < num_layers; ++i) {
    layers.emplace_back(d_model, num_heads, d_ff);
  }
}

Variable *Encoder::forward(GraphContext &ctx, Variable *src_ids,
                           Variable *src_mask) {
  // Embed source IDs and scale embeddings
  auto embed_out = embedding.forward(ctx, src_ids);
  float scale_factor =
      std::sqrt(static_cast<float>(embedding.weights.data.dim(1)));
  auto scaled_embed = scale(ctx, embed_out, scale_factor);

  // Slice positional encoding to match current sequence length
  int64_t seq_len = src_ids->data.dim(1);
  auto pos_encoding_sliced = slice(
      ctx, &pos_encoding, {{0, 1}, {0, seq_len}, {0, -1}}); // Slice to seq len

  // Add positional encoding to scaled embeddings
  auto x = add(ctx, scaled_embed, pos_encoding_sliced);

  // Pass through all encoder blocks
  for (auto &layer : layers) {
    x = layer.forward(ctx, x, src_mask);
  }
  return x;
}

std::vector<Variable *> Encoder::parameters() {
  auto params = embedding.parameters();
  // pos_encoding is not a trainable parameter
  for (auto &layer : layers) {
    auto p_layer = layer.parameters();
    params.insert(params.end(), p_layer.begin(), p_layer.end());
  }
  return params;
}

//===----------------------------------------------------------------------===//
// Decoder Implementation
//===----------------------------------------------------------------------===//

Decoder::Decoder(int vocab_size, int d_model, int num_heads, int num_layers,
                 int d_ff, int max_len)
    : embedding(vocab_size, d_model),
      pos_encoding(create_positional_encoding(max_len, d_model),
                   /*requires_grad=*/false) {
  for (int i = 0; i < num_layers; ++i) {
    layers.emplace_back(d_model, num_heads, d_ff);
  }
}

Variable *Decoder::forward(GraphContext &ctx, Variable *tgt_ids,
                           Variable *enc_output, Variable *src_mask,
                           Variable *tgt_mask) {
  // Embed target IDs and scale embeddings
  auto embed_out = embedding.forward(ctx, tgt_ids);
  float scale_factor =
      std::sqrt(static_cast<float>(embedding.weights.data.dim(1)));
  auto scaled_embed = scale(ctx, embed_out, scale_factor);

  // Slice positional encoding to match current sequence length
  int64_t seq_len = tgt_ids->data.dim(1);
  auto pos_encoding_sliced = slice(
      ctx, &pos_encoding, {{0, 1}, {0, seq_len}, {0, -1}}); // Slice to seq len

  // Add positional encoding to scaled embeddings
  auto x = add(ctx, scaled_embed, pos_encoding_sliced);

  // Pass through all decoder blocks
  for (auto &layer : layers) {
    x = layer.forward(ctx, x, enc_output, src_mask, tgt_mask);
  }
  return x;
}

std::vector<Variable *> Decoder::parameters() {
  auto params = embedding.parameters();
  for (auto &layer : layers) {
    auto p_layer = layer.parameters();
    params.insert(params.end(), p_layer.begin(), p_layer.end());
  }
  return params;
}

//===----------------------------------------------------------------------===//
// Transformer Implementation
//===----------------------------------------------------------------------===//

Transformer::Transformer(int src_vocab_size, int tgt_vocab_size, int d_model,
                         int num_heads, int num_layers, int d_ff, int max_len)
    : encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len),
      decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len),
      final_projection(d_model, tgt_vocab_size) {}

Variable *Transformer::forward(GraphContext &ctx, Variable *src_ids,
                               Variable *tgt_ids, Variable *src_mask,
                               Variable *tgt_mask) {
  // Encode the source sequence
  auto enc_output = encoder.forward(ctx, src_ids, src_mask);
  // Decode the target sequence using encoder output
  auto dec_output =
      decoder.forward(ctx, tgt_ids, enc_output, src_mask, tgt_mask);
  // Project decoder output to vocabulary logits
  return final_projection.forward(ctx, dec_output);
}

std::vector<Variable *> Transformer::parameters() {
  auto params = encoder.parameters();
  auto p_dec = decoder.parameters();
  auto p_proj = final_projection.parameters();
  params.insert(params.end(), p_dec.begin(), p_dec.end());
  params.insert(params.end(), p_proj.begin(), p_proj.end());
  return params;
}

} // namespace sylvan::core
