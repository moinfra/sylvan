#pragma once

#include "sylvan/core/autograd.h"
#include <vector>

namespace sylvan::core {

struct Linear {
    Variable W;
    Variable b;

    Linear(int in_features, int out_features);

    Variable forward(Variable& x);
    std::vector<Variable*> parameters();
};

}
