#pragma once

#include "sylvan/core/autograd.h"
#include <vector>

namespace sylvan::core {

class SGD {
public:
    SGD(std::vector<Variable*> params, float lr = 0.01);
    void step();
    void zero_grad();

private:
    std::vector<Variable*> params_;
    float lr_;
};

}
