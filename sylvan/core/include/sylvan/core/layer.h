#pragma once

#include "sylvan/core/autograd.h"
#include <vector>

namespace sylvan::core {

struct Linear {
    // Parameters are persistent and owned by the layer.
    Variable W;
    Variable b;

    Linear(int in_features, int out_features);

    // The forward pass operates within a given graph context.
    Variable* forward(GraphContext& ctx, Variable* x);
    std::vector<Variable*> parameters();
};

}
