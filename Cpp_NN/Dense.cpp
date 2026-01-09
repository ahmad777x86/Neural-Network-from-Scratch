#include "Dense.h"
#include <vector>

Dense::Dense(int n_inputs, int n_neurons)
{
    std::vector<std::vector<double>> weight(n_inputs, std::vector<double>(n_neurons));
    std::vector<std::vector<double>> bias(1, std::vector<double>(n_neurons));
    std::vector<std::vector<double>> w_gradient(n_inputs, std::vector<double>(n_neurons));
    std::vector<std::vector<double>> b_gradient(1, std::vector<double>(n_neurons));
}

std::vector<std::vector<double>> Dense::forward(std::vector<std::vector<double>> &X)
{
    std::vector<std::vector<double>> output(weight.size(), std::vector<double>(X[0].size(), 0.0));
    /* for (auto i = X.begin(); i != X.end(); i++)
    {
        for (auto j = i->begin(); j != i->end(); j++)
        {
            output[i][j] +=
        }
    } */
    return X;
}

std::vector<std::vector<double>> Dense::backward(std::vector<std::vector<double>> &grad)
{
    return grad;
}