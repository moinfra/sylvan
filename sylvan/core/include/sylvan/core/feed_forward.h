#pragma once
#include "sylvan/core/layers/linear.h"
#include "sylvan/core/autograd.h"

namespace sylvan::core {

struct FeedForward {
    Linear layer1;
    Linear layer2;

    FeedForward(int d_model, int d_ff);
    
    Variable* forward(GraphContext& ctx, Variable* x);
    std::vector<Variable*> parameters();
};

} // namespace sylvan::core
