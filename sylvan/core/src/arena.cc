// sylvan/core/arena.cc
//
// Implements a simple memory arena for fast, contiguous memory allocation.
// This arena is designed for objects with known lifetimes within a graph
// context, offering very quick allocation and deallocation by simply resetting
// a pointer.
//
// Author: Zijing Zhang
// Date: 2024-07-27
// Copyright: (c) 2024 Sylvan Framework. All rights reserved.
// SPDX-License-Identifier: MIT

#include "sylvan/core/arena.h"

namespace sylvan::core {

Arena::Arena(size_t block_size)
    : block_size_(block_size), current_block_(nullptr), current_offset_(0) {}

Arena::~Arena() {
  // Release all allocated memory blocks.
  for (char *block : blocks_) {
    delete[] block;
  }
}

// -------- Internal Helper: new_block --------
void Arena::new_block() {
  char *block = new char[block_size_];
  blocks_.push_back(block);
  current_block_ = block;
  current_offset_ = 0;
}

void *Arena::allocate(size_t size) {
  // Simple 8-byte alignment for allocated memory.
  size = (size + 7) & ~7;

  // Check if the current block has enough space, or if no block exists.
  if (!current_block_ || current_offset_ + size > block_size_) {
    // If the requested size is larger than a standard block, allocate a
    // special, dedicated large block to avoid wasting space in smaller blocks.
    if (size > block_size_) {
      char *block = new char[size];
      blocks_.push_back(block);
      return block;
    }
    // Otherwise, allocate a new standard-sized block.
    new_block();
  }

  // Allocate memory from the current block.
  char *ptr = current_block_ + current_offset_;
  current_offset_ += size;
  return ptr;
}

void Arena::reset() {
  // Reset the allocation pointer to reuse memory from the beginning of the
  // first block. This is a very fast way to "deallocate" all memory managed by
  // the arena. NOTE: This operation does NOT call destructors for objects
  // previously allocated on the arena. This is a deliberate design choice,
  // making it suitable for POD types or objects like `Variable` whose internal
  // resources (e.g., `Tensor`'s GPU memory) are managed independently (e.g., by
  // smart pointers or custom destructors).
  current_offset_ = 0;
  if (!blocks_.empty()) {
    current_block_ = blocks_[0]; // Reset to the first block for reuse.
  }
}

} // namespace sylvan::core
