// examples/number_translator/main.cc
//
// This example demonstrates how to build and train a Transformer model
// using the Sylvan framework to translate integer numbers into their
// English word representations (e.g., 123 -> "one hundred twenty three").
// It showcases data preparation, model definition, training loop,
// and auto-regressive inference.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/autograd.h"
#include "sylvan/core/graph.h"
#include "sylvan/core/optimizer.h"
#include "sylvan/core/transformer.h"
#include "sylvan/tensor/operators.h"
#include <cfloat>        
#include <cuda_runtime.h>
#include <functional>    
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

/**
 * @brief Top-level namespace for the Sylvan deep learning framework.
 *
 * This namespace contains all core components of Sylvan, including tensor
 * operations, automatic differentiation, graph management, and neural network
 * modules.
 */
using namespace sylvan;

/**
 * @brief Prints the current GPU memory usage to standard output.
 *
 * This utility function queries the CUDA runtime API for the amount of free,
 * used, and total GPU memory on the default device. The memory values are
 * converted to megabytes (MB) for readability.
 */
void print_gpu_memory_usage() {
  size_t free_byte, total_byte;
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if (cuda_status != cudaSuccess) {
    printf("Error getting CUDA memory info: %s\n",
           cudaGetErrorString(cuda_status));
    return;
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}

//===----------------------------------------------------------------------===//
// ALGORITHMIC NUMBER-TO-WORDS CONVERTER
//===----------------------------------------------------------------------===//

/**
 * @brief Converts an integer number into its English word representation.
 *
 * This function provides a rule-based conversion for integers up to 99,999.
 * It handles digits, teens, tens, hundreds, and thousands using predefined
 * string arrays and a recursive helper function for hierarchical breakdown.
 *
 * @param n The integer number to convert. Must be between 0 and 99,999 (inclusive).
 * @return The English word representation of the number (e.g., 123 -> "one hundred twenty three").
 *         Returns "zero" for input 0. Returns "out of range" if the number is
 *         negative or exceeds the supported maximum (99,999).
 */
std::string number_to_words(int n) {
  if (n == 0)
    return "zero";
  if (n < 0 || n > 99999)
    return "out of range";

  const std::vector<std::string> ones = {"",      "one",  "two", "three",
                                         "four",  "five", "six", "seven",
                                         "eight", "nine"};
  const std::vector<std::string> teens = {
      "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
      "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
  const std::vector<std::string> tens = {"",       "",      "twenty", "thirty",
                                         "forty",  "fifty", "sixty",  "seventy",
                                         "eighty", "ninety"};

  // -------- Recursive Helper for Number Conversion --------
  // This lambda recursively breaks down the number into smaller parts
  // (e.g., thousands, hundreds, tens, ones) and combines their word representations.
  std::function<std::string(int)> helper = [&](int num) -> std::string {
    if (num == 0)
      return "";
    if (num < 10)
      return ones[num];
    if (num < 20)
      return teens[num - 10];
    if (num < 100)
      return tens[num / 10] + (num % 10 != 0 ? " " + ones[num % 10] : "");
    if (num < 1000)
      return ones[num / 100] + " hundred" +
             (num % 100 != 0 ? " " + helper(num % 100) : "");
    if (num < 100000)
      return helper(num / 1000) + " thousand" +
             (num % 1000 != 0 ? " " + helper(num % 1000) : "");
    return ""; // Should not be reached for valid inputs within range.
  };

  std::string result = helper(n);
  // Trim leading space that might result from recursive calls (e.g., " thousand ...").
  if (!result.empty() && result[0] == ' ') {
    result = result.substr(1);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// VOCABULARY/DATA PREPARATION
//===----------------------------------------------------------------------===//

/**
 * @brief Manages character-to-integer ID mappings for the number translator model.
 *
 * This struct provides a bidirectional mapping between textual characters
 * (digits '0'-'9', hyphen '-', space ' ', and lowercase alphabet 'a'-'z')
 * and their corresponding unique integer IDs. It also includes special tokens
 * for padding (PAD), start-of-sequence (SOS), and end-of-sequence (EOS).
 *
 * The special tokens are assigned low integer IDs and use non-printable ASCII
 * characters as their string representations internally to avoid collisions
 * with standard text characters.
 */
struct CharMap {
  std::map<char, int> c_to_i; ///< Maps textual characters to their integer IDs.
  std::map<int, char> i_to_c; ///< Maps integer IDs back to their textual characters.
  int pad_id;                 ///< The integer ID for the padding token (value 0).
  int sos_id;                 ///< The integer ID for the start-of-sequence token (value 1).
  int eos_id;                 ///< The integer ID for the end-of-sequence token (value 2).

  /**
   * @brief Constructs a CharMap instance, initializing its vocabulary.
   *
   * This constructor populates the character-to-ID maps. It first adds
   * special tokens (PAD, SOS, EOS) with unique, non-printable character
   * representations and then adds standard characters used in number-to-word
   * translation: digits ('0'-'9'), hyphen ('-'), space (' '), and
   * lowercase English letters ('a'-'z').
   */
  CharMap() : pad_id(0), sos_id(1), eos_id(2) {
    // Add special tokens first with unique, non-printable char representations.
    // Using ASCII control characters (Null, SOH, STX) to avoid conflicts.
    i_to_c[pad_id] = '\0'; // Null character
    c_to_i['\0'] = pad_id;
    i_to_c[sos_id] = '\1'; // Start of Header (SOH)
    c_to_i['\1'] = sos_id;
    i_to_c[eos_id] = '\2'; // Start of Text (STX)
    c_to_i['\2'] = eos_id;

    // Add standard characters: digits, hyphen, space, and lowercase alphabet.
    std::string chars = "0123456789- abcdefghijklmnopqrstuvwxyz";
    for (char c : chars) {
      // Only add if not already present (e.g., special tokens are already mapped).
      if (c_to_i.find(c) == c_to_i.end()) {
        int id = i_to_c.size(); // Assign the next available integer ID
        i_to_c[id] = c;
        c_to_i[c] = id;
      }
    }
  }

  /**
   * @brief Returns the total number of unique tokens (vocabulary size).
   *
   * This includes all standard characters and special tokens (PAD, SOS, EOS).
   *
   * @return The size of the vocabulary.
   */
  int size() const { return i_to_c.size(); }

  /**
   * @brief Converts a string into a vector of integer IDs, with optional SOS/EOS and padding.
   *
   * This method processes an input string, converting each character to its
   * corresponding integer ID. It can optionally prepend a Start-of-Sequence (SOS)
   * token and append an End-of-Sequence (EOS) token. The resulting sequence
   * is then padded with the PAD token up to `max_len`.
   *
   * @param s The input string to convert.
   * @param max_len The maximum desired length of the output ID sequence, including
   *                any special tokens and padding.
   * @param add_sos If true, the SOS token is prepended to the sequence.
   * @param add_eos If true, the EOS token is appended to the sequence (before padding).
   * @return A `std::vector<int>` containing the integer IDs of the string,
   *         padded to `max_len`.
   * @throws std::runtime_error if the resulting sequence length (after adding
   *         SOS/EOS and characters) exceeds `max_len` before padding.
   */
  std::vector<int> to_ids(const std::string &s, size_t max_len,
                          bool add_sos = false, bool add_eos = false) const {
    std::vector<int> ids;
    if (add_sos)
      ids.push_back(sos_id);
    for (char c : s) {
      if (c_to_i.count(c)) // Only add characters that are present in the map (i.e., known vocabulary).
        ids.push_back(c_to_i.at(c));
    }
    if (add_eos)
      ids.push_back(eos_id);

    if (ids.size() > max_len) {
      throw std::runtime_error("Sequence longer than max_len after adding tokens!");
    }

    // Pad with PAD_ID until max_len is reached.
    while (ids.size() < max_len)
      ids.push_back(pad_id);
    return ids;
  }

  /**
   * @brief Converts a vector of integer IDs back into a human-readable string.
   *
   * This method decodes a sequence of integer IDs back into a string.
   * Decoding stops at the first End-of-Sequence (EOS) token encountered.
   * Start-of-Sequence (SOS) and Padding (PAD) tokens are skipped.
   * Unknown IDs (not found in `i_to_c` map) are ignored.
   *
   * @param ids The `std::vector<int>` containing the integer IDs to decode.
   * @return The decoded string, representing the original text.
   */
  std::string to_string(const std::vector<int> &ids) const {
    std::string s;
    for (int id : ids) {
      if (id == eos_id) // Stop decoding at the first EOS token.
        break;
      if (id != sos_id && id != pad_id) { // Skip SOS and PAD tokens from the output string.
        if (i_to_c.count(id))
          s += i_to_c.at(id);
      }
    }
    return s;
  }
};

/**
 * @brief Helper function to convert a vector of integers to a Sylvan Tensor.
 *
 * This utility function creates a new Sylvan Tensor with `Int32` DType
 * initialized with the provided integer IDs. It is useful for preparing
 * tokenized input sequences for the Transformer model.
 *
 * @param ids The `std::vector<int>` containing the integer IDs.
 * @param shape The desired `sylvan::tensor::Shape` for the output tensor.
 *              The total number of elements in `ids` must match `shape.num_elements()`.
 * @return A new `sylvan::tensor::Tensor` with DType `Int32` and the specified shape.
 */
sylvan::tensor::Tensor ids_to_int_tensor(const std::vector<int> &ids,
                                         sylvan::tensor::Shape shape) {
  return sylvan::tensor::ops::from_host(ids, shape);
}

//===----------------------------------------------------------------------===//
// INFERENCE UTILITY
//===----------------------------------------------------------------------===//

/**
 * @brief Translates a given integer number into its English word representation
 *        using a trained Transformer model with greedy decoding.
 *
 * This function performs auto-regressive inference. It first encodes the
 * numerical input, then iteratively generates the output word sequence
 * token by token. At each step, it feeds the previously generated tokens
 * back into the decoder, and the token with the highest predicted logit
 * is chosen as the next token. The process continues until an EOS token
 * is predicted or the maximum sequence length (`max_len`) is reached.
 *
 * @param model A reference to the trained `sylvan::core::Transformer` model.
 * @param number The integer number to translate (e.g., 123).
 * @param char_map The `CharMap` instance used for converting between
 *                 characters and integer IDs.
 * @param max_len The maximum sequence length allowed for both input and output
 *                sequences, including special tokens and padding.
 * @return A `std::string` containing the translated English word representation.
 */
std::string translate_number(sylvan::core::Transformer &model, int number,
                             const CharMap &char_map, int max_len) {

  std::string input_str = std::to_string(number);

  // -------- Prepare Encoder Input --------
  // Convert the numerical input string into a tensor of integer IDs, padded to max_len.
  std::vector<int> input_ids = char_map.to_ids(input_str, max_len);
  auto src_tensor =
      ids_to_int_tensor(input_ids, {1, (int64_t)input_ids.size()});

  // -------- Prepare Encoder Padding Mask --------
  // This mask informs the self-attention mechanism in the encoder to ignore
  // interactions with padding tokens.
  auto src_mask_tensor =
      sylvan::tensor::ops::create_padding_mask(src_tensor, char_map.pad_id);

  // -------- Initialize Decoder Input --------
  // The auto-regressive decoding process starts with only the Start-of-Sequence (SOS) token.
  std::vector<int> output_ids = {char_map.sos_id};

  // ======== Auto-Regressive Decoding Loop ========
  // Generate one token at each step until EOS is predicted or max_len is reached.
  for (int i = 0; i < max_len - 1; ++i) {
    // A new graph context is created for each decoding step.
    sylvan::core::GraphContext graph;

    // -------- Prepare Inputs for Current Decoding Step --------
    // Create variables for encoder input (src) and its padding mask.
    // These do not require gradients during inference.
    auto *src_var =
        graph.create_variable(src_tensor.clone(), /*requires_grad=*/false);
    auto *src_mask_var =
        graph.create_variable(src_mask_tensor.clone(), /*requires_grad=*/false);

    // Create variable for the decoder target input, which is the sequence
    // of tokens generated so far. This also does not require gradients.
    auto tgt_tensor =
        ids_to_int_tensor(output_ids, {1, (int64_t)output_ids.size()});
    auto *tgt_var =
        graph.create_variable(std::move(tgt_tensor), /*requires_grad=*/false);

    // Create a causal mask for the decoder target input.
    // This mask prevents the decoder from "looking into the future" tokens
    // when predicting the next token.
    auto tgt_mask_tensor =
        sylvan::tensor::ops::create_causal_mask(output_ids.size());
    auto *tgt_mask_var = graph.create_variable(std::move(tgt_mask_tensor),
                                               /*requires_grad=*/false);

    // -------- Forward Pass: Get Logits for the Next Token --------
    // The model predicts a distribution over the vocabulary for the next token.
    auto *logits =
        model.forward(graph, src_var, tgt_var, src_mask_var, tgt_mask_var);

    // -------- Greedy Decoding: Select the Token with Highest Logit --------
    // Clone the logits from device to host to find the maximum.
    auto all_logits_host =
        sylvan::tensor::ops::clone_to_host<float>(logits->data);

    // The logits tensor has shape [batch_size, sequence_length, vocab_size].
    // For batch_size = 1, we are interested in the logits of the *last* token in the sequence.
    int vocab_size = model.final_projection.W.data.dim(1); // W shape is [d_model, vocab_size]
    int last_token_logits_start_index = (output_ids.size() - 1) * vocab_size;

    float max_logit = -FLT_MAX;
    int next_token_id = char_map.eos_id; // Default to EOS if no better token is found.

    for (int j = 0; j < vocab_size; ++j) {
      float current_logit = all_logits_host[last_token_logits_start_index + j];
      if (current_logit > max_logit) {
        max_logit = current_logit;
        next_token_id = j;
      }
    }

    // Stop decoding if the predicted token is the End of Sequence (EOS) token.
    if (next_token_id == char_map.eos_id) {
      break;
    }

    // Add the predicted token to the output sequence and continue decoding.
    output_ids.push_back(next_token_id);
  }

  // Convert the final sequence of integer IDs back to a human-readable string.
  return char_map.to_string(output_ids);
}

//===----------------------------------------------------------------------===//
// MAIN TRAINING AND EVALUATION LOOP
//===----------------------------------------------------------------------===//

/**
 * @brief Main entry point for the Sylvan Number Translator example.
 *
 * This function orchestrates the entire training and evaluation process:
 * 1. Initializes a Transformer model and an Adam optimizer.
 * 2. Sets up progressive training, starting with simpler numbers and gradually
 *    increasing complexity (number of digits) as the model performs well.
 * 3. Manages data generation, batching, forward/backward passes, and parameter updates.
 * 4. Logs training progress, loss, and performs periodic inference tests.
 * 5. Conducts a final evaluation of the trained model.
 *
 * @return 0 upon successful execution.
 */
int main() {
  std::cout << "🚀 Sylvan Number Translator Example" << std::endl;
  std::cout
      << "This example trains a Transformer model to translate numbers into "
         "their English word representations (e.g., 123 -> \"one hundred twenty three\")."
      << std::endl;
  std::cout << "Example: It translates " << 123 << " to '" << number_to_words(123)
            << "'" << std::endl;

  CharMap char_map;

  // -------- Hyperparameters for the Transformer model and training --------
  const int vocab_size = char_map.size(); // Total number of unique tokens.
  const int d_model = 128;    // Dimension of model embeddings and internal representations.
  const int num_layers = 4;   // Number of stacked encoder and decoder layers.
  const int num_heads = 4;    // Number of attention heads in multi-head attention.
  const int d_ff = 256;       // Dimension of the feed-forward network in each layer.
  const int max_len = 64;     // Maximum sequence length for both input and output.
  const int total_epochs = 20000; // Total training epochs, adjusted for progressive training.
  const float learning_rate = 1e-4f;
  const int batch_size = 512; // Number of samples processed in one training iteration.

  // -------- Progressive Training Parameters --------
  // The model starts training on simpler (fewer digits) numbers and
  // progresses to more complex ones as it achieves a certain loss threshold.
  int current_magnitude = 1; // Start with 1-digit numbers (1-9).
  const float loss_threshold = 0.5f; // Loss threshold to move to the next magnitude.
  const int check_loss_interval = 100; // Check loss every N epochs to decide magnitude progression.
  float current_avg_loss = FLT_MAX; // Moving average loss for the current magnitude.

  // -------- Model and Optimizer Initialization --------
  // Instantiate the Transformer model with defined hyperparameters.
  sylvan::core::Transformer model(vocab_size, vocab_size, d_model, num_heads,
                                  num_layers, d_ff, max_len);
  // Initialize the Adam optimizer with the model's trainable parameters.
  sylvan::core::Adam optimizer(model.parameters(), learning_rate);

  // Pre-create the causal mask for the decoder.
  // This mask is static and applied to all batches during training.
  auto tgt_causal_mask = sylvan::tensor::ops::create_causal_mask(max_len);

  // -------- Random Number Generators Setup --------
  std::random_device rd;  // Used to seed the random number generator.
  std::mt19937 gen(rd()); // Mersenne Twister engine for random numbers.
  // Distribution for random test numbers across the full training range (1-9999).
  std::uniform_int_distribution<> distrib(1, 9999);

  // Specific distributions for each magnitude (1-digit, 2-digit, etc.)
  // used for generating training data progressively.
  std::vector<std::uniform_int_distribution<>> number_dist_by_magnitude = {
      std::uniform_int_distribution<>(1, 9),      // Index 0: 1-digit (1-9)
      std::uniform_int_distribution<>(10, 99),     // Index 1: 2-digit (10-99)
      std::uniform_int_distribution<>(100, 999),   // Index 2: 3-digit (100-999)
      std::uniform_int_distribution<>(1000, 9999)}; // Index 3: 4-digit (1000-9999)

  std::cout << "Model created. Starting progressive training..." << std::endl;

  // ======== Main Training Loop ========
  for (int epoch = 0; epoch <= total_epochs; ++epoch) {
    // Determine the maximum magnitude of numbers to include in this training epoch's batch.
    // This value increases progressively.
    int max_train_magnitude = current_magnitude;
    if (current_magnitude > 4) { // Cap at 4-digits as the maximum magnitude to train on.
        max_train_magnitude = 4;
    }

    // Distribution to pick a magnitude for training data, ensuring numbers
    // from 1-digit up to `max_train_magnitude`-digits are included.
    std::uniform_int_distribution<> magnitude_dist(1, max_train_magnitude);

    // -------- Batch Data Generation --------
    // Prepare vectors to hold the token IDs for the entire batch.
    std::vector<int> batched_input_ids;          // Encoder input (numerical string IDs)
    std::vector<int> batched_target_input_ids;   // Decoder input (word string IDs with SOS)
    std::vector<int> batched_target_output_ids;  // Decoder target (word string IDs with EOS, for loss calculation)

    batched_input_ids.reserve(batch_size * max_len);
    batched_target_input_ids.reserve(batch_size * max_len);
    batched_target_output_ids.reserve(batch_size * max_len);

    for (int i = 0; i < batch_size; ++i) {
      // Randomly select a magnitude for the current number based on `magnitude_dist`.
      int magnitude_choice = magnitude_dist(gen) - 1; // Adjust to 0-indexed for vector access.
      int number = number_dist_by_magnitude[magnitude_choice](gen); // Generate a random number of that magnitude.

      std::string input_str = std::to_string(number);
      std::string target_str = number_to_words(number);

      // Convert the input and target strings for a single sample into ID sequences.
      std::vector<int> input_ids = char_map.to_ids(input_str, max_len);
      // Decoder input (tgt_in): Includes SOS, no EOS.
      std::vector<int> target_input_ids =
          char_map.to_ids(target_str, max_len, true, false);
      // Decoder target (tgt_out): No SOS, includes EOS, used for loss calculation.
      std::vector<int> target_output_ids =
          char_map.to_ids(target_str, max_len, false, true);

      // Append (flatten) the single sample's IDs to the batched vectors.
      batched_input_ids.insert(batched_input_ids.end(), input_ids.begin(),
                               input_ids.end());
      batched_target_input_ids.insert(batched_target_input_ids.end(),
                                      target_input_ids.begin(),
                                      target_input_ids.end());
      batched_target_output_ids.insert(batched_target_output_ids.end(),
                                       target_output_ids.begin(),
                                       target_output_ids.end());
    }

    // -------- Training Step --------
    sylvan::core::GraphContext graph; // A new computational graph for each training step.
    optimizer.zero_grad();            // Reset gradients for all model parameters.

    // Define the shape for batched tensors: [batch_size, max_len].
    sylvan::tensor::Shape tensor_shape = {(int64_t)batch_size,
                                          (int64_t)max_len};

    // Create Sylvan variables from the batched data.
    // Inputs and targets do not require gradients themselves, as they are fixed data.
    auto *src_var = graph.create_variable(
        ids_to_int_tensor(batched_input_ids, tensor_shape), false);
    auto *tgt_in_var = graph.create_variable(
        ids_to_int_tensor(batched_target_input_ids, tensor_shape), false);
    auto *tgt_out_var = graph.create_variable(
        ids_to_int_tensor(batched_target_output_ids, tensor_shape), false);

    // Create padding mask for the encoder source input.
    // This mask prevents attention to padding tokens in the encoder.
    auto src_padding_mask = sylvan::tensor::ops::create_padding_mask(src_var->data, char_map.pad_id);
    auto *src_mask_var = graph.create_variable(std::move(src_padding_mask));

    // The causal mask for the decoder target input is pre-created and shared.
    // It prevents looking into future tokens.
    auto *mask_var = graph.create_variable(tgt_causal_mask.clone());

    // Forward pass through the Transformer model.
    auto *logits = model.forward(graph, src_var, tgt_in_var, src_mask_var, mask_var);

    // Calculate Cross-Entropy Loss.
    auto *loss = core::cross_entropy_loss(graph, logits, tgt_out_var);

    loss->backward();
    optimizer.step();

    // -------- Logging and Evaluation --------
    // Clone loss value to host for printing.
    auto loss_val = sylvan::tensor::ops::clone_to_host<float>(loss->data);
    // Update simple moving average of loss for progressive training.
    current_avg_loss = (current_avg_loss == FLT_MAX) ? loss_val[0] : (current_avg_loss * 0.9 + loss_val[0] * 0.1);

    if (epoch > 0 && epoch % 5 == 0) {
      std::cout << "Epoch " << epoch << " | Current Magnitude: " << current_magnitude
                << " | Loss: " << loss_val[0] << " | Avg Loss (last " << check_loss_interval << " epochs): " << current_avg_loss << std::endl;
      print_gpu_memory_usage(); // Monitor GPU memory usage.

      // Perform a random inference test during training to observe model's progress.
      // Select a random number within the currently trained magnitude range.
      int test_magnitude = magnitude_dist(gen) - 1;
      int test_num_random = number_dist_by_magnitude[test_magnitude](gen);

      std::string translation =
          translate_number(model, test_num_random, char_map, max_len);
      std::cout << "  Translate(" << test_num_random << ") -> '" << translation
                << "'" << std::endl;
      std::cout << "  Gold: '" << number_to_words(test_num_random) << "'" << std::endl;
    }

    // -------- Progressive Training Logic --------
    // Check if the average loss for the current magnitude is below the threshold.
    // If so, increment `current_magnitude` to start training on more complex numbers.
    if (epoch > 0 && current_magnitude < 4 && epoch % check_loss_interval == 0) {
        if (current_avg_loss < loss_threshold) {
            current_magnitude++;
            std::cout << "\n🎉 Loss (" << current_avg_loss << ") below threshold (" << loss_threshold
                      << ") for " << (current_magnitude - 1) << "-digit numbers. Moving to training "
                      << current_magnitude << "-digit numbers!\n" << std::endl;
            current_avg_loss = FLT_MAX; // Reset average loss for the new magnitude.
        } else {
            std::cout << "Loss (" << current_avg_loss << ") still above threshold (" << loss_threshold
                      << ") for " << current_magnitude << "-digit numbers. Continuing training on current magnitude." << std::endl;
        }
    }
  }

  std::cout << "\n✅ Training finished. Final showcase:\n" << std::endl;
  // -------- Final Evaluation --------
  // Test the fully trained model with a few random numbers across the full range (1-9999).
  for (int i = 0; i < 5; ++i) {
    int test_num = distrib(gen);
    std::cout << "Input:  " << test_num << std::endl;
    std::cout << "Gold:   '" << number_to_words(test_num) << "'" << std::endl;
    std::cout << "Sylvan: '"
              << translate_number(model, test_num, char_map, max_len)
              << "'" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
