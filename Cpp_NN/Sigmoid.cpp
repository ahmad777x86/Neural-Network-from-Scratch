#include "Sigmoid.h"
#include "Utilities/Matrix_Op.h"
#include <cmath>

std::vector<std::vector<double>> Sigmoid::forward(std::vector<std::vector<double>> &X)
{
    output = X;
    for (auto &i : output)
    {
        for (auto &j : i)
        {
            j = 1 / (1 + exp(-j));
        }
    }
    return output;
}

std::vector<std::vector<double>> Sigmoid::backward(std::vector<std::vector<double>> &grad)
{
    std::vector<std::vector<double>> output_clipped = Matrix::clip(output);
    auto sigmoid_grad = Matrix::multiply_matrix(grad, Matrix::multiply_matrix(output_clipped, Matrix::multiply_scalar(Matrix::add_scalar(output_clipped, -1), -1)));
    return sigmoid_grad;
}
