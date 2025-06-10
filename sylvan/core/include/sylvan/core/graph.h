#pragma once

#include "sylvan/core/arena.h"
#include "sylvan/core/autograd.h"

namespace sylvan::core {

class GraphContext {
public:
  GraphContext() = default;

  template <typename... Args> Variable *create_variable(Args &&...args) {
    return arena_.create<Variable>(std::forward<Args>(args)...);
  }

  void reset() { arena_.reset(); }

private:
  Arena arena_;
};

} // namespace sylvan::core
