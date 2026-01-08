#include "Sigmoid.h"
#include <cmath>

std::vector<std::vector<double>> Sigmoid::forward(std::vector<std::vector<double>> &X)
{
    for (auto &i : X)
    {
        for (auto &j : i)
        {
            j = 1 / (1 + exp(-j));
        }
    }
    return X;
}

/* std::vector<std::vector<double>> Sigmoid::backward(std::vector<std::vector<double>> &X)
{
} */
