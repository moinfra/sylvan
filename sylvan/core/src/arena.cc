#include "sylvan/core/arena.h"
#include <algorithm> // For std::max

namespace sylvan::core {

Arena::Arena(size_t block_size) 
    : block_size_(block_size), current_block_(nullptr), current_offset_(0) {}

Arena::~Arena() {
    for (char* block : blocks_) {
        delete[] block;
    }
}

void Arena::new_block() {
    char* block = new char[block_size_];
    blocks_.push_back(block);
    current_block_ = block;
    current_offset_ = 0;
}

void* Arena::allocate(size_t size) {
    // 简单的对齐到 8 字节
    size = (size + 7) & ~7;

    if (!current_block_ || current_offset_ + size > block_size_) {
        // 如果请求的大小大于整个块，就分配一个特殊的大块
        if (size > block_size_) {
            char* block = new char[size];
            blocks_.push_back(block);
            return block;
        }
        new_block();
    }
    
    char* ptr = current_block_ + current_offset_;
    current_offset_ += size;
    return ptr;
}

void Arena::reset() {
    // 只是重置指针，内存并不释放，下次分配时会覆盖
    // 注意：这使得 arena 中对象的析构函数不会被调用。
    // 这对于我们的 Variable 是可以接受的，因为它的主要资源 Tensor 是通过 shared_ptr 管理的。
    current_offset_ = 0;
    if (!blocks_.empty()) {
        current_block_ = blocks_[0];
    }
}

} // namespace sylvan::core
