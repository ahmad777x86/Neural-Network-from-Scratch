#include "Dense.h"
#include "Utilities/Matrix_Op.h"
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
    input = X;
    std::vector<std::vector<double>> output(X.size(), std::vector<double>(weight[0].size(), std::rand()));
    output = Matrix::dot(X, weight);
    return output;
}

std::vector<std::vector<double>> Dense::backward(std::vector<std::vector<double>> &grad)
{
    w_gradient = Matrix::dot(Matrix::transpose(input), grad);
    b_gradient = Matrix::sum(grad);
    auto input_gradient = Matrix::dot(grad, Matrix::transpose(weight));
    return grad;
}