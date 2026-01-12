#include "Dense.h"
#include "Utilities/dot_product.h"
#include <vector>
#include <iostream>

Dense::Dense(int n_inputs, int n_neurons)
{
    weight = std::vector<std::vector<double>>(n_inputs, std::vector<double>(n_neurons, std::rand()));
    bias = std::vector<std::vector<double>>(1, std::vector<double>(n_neurons));
    w_gradient = std::vector<std::vector<double>>(n_inputs, std::vector<double>(n_neurons));
    b_gradient = std::vector<std::vector<double>>(1, std::vector<double>(n_neurons));
}

std::vector<std::vector<double>> Dense::forward(std::vector<std::vector<double>> &X)
{
    std::vector<std::vector<double>> output(X.size(), std::vector<double>(weight[0].size(), std::rand()));
    output = Dot_Product::compute(X, weight);
    return output;
}

std::vector<std::vector<double>> Dense::backward(std::vector<std::vector<double>> &grad)
{
    return grad;
}