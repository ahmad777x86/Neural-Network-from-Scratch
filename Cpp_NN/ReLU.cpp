#include "ReLU.h"
#include <vector>

std::vector<std::vector<double>> ReLU::forward(std::vector<std::vector<double>> &X)
{
    for (auto &i : X)
    {
        for (auto &j : i)
        {
            if (j < 0)
                j = 0.0;
        }
    }
    return X;
}

/* std::vector<std::vector<double>> ReLU::backward(std::vector<std::vector<double>> &grad)
{
} */
