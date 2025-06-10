#pragma once

#include <cstddef>
#include <vector>

namespace sylvan::core {

class Arena {
public:
  Arena(size_t block_size = 4096);
  ~Arena();

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  void *allocate(size_t size);

  template <typename T, typename... Args> T *create(Args &&...args) {
    void *mem = allocate(sizeof(T));
    return new (mem) T(std::forward<Args>(args)...);
  }

  void reset();

private:
  void new_block();

  size_t block_size_;
  std::vector<char *> blocks_;
  char *current_block_;
  size_t current_offset_;
};

} // namespace sylvan::core
