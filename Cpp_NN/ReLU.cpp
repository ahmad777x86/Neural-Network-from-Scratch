#include "ReLU.h"
#include "Utilities/Matrix_Op.h"
#include <vector>

std::vector<std::vector<double>> ReLU::forward(std::vector<std::vector<double>> &X)
{
    input = X;
    auto output = input;
    for (auto &i : output)
    {
        for (auto &j : i)
        {
            if (j < 0)
                j = 0.0;
        }
    }
    return output;
}

std::vector<std::vector<double>> ReLU::backward(std::vector<std::vector<double>> &grad)
{
    auto mask = input;
    for (auto &i : mask)
    {
        for (auto &j : i)
        {
            if (j < 0)
                j = 0;
            else
                j = 1;
        }
    }
    auto input_gradient = Matrix::multiply_matrix(grad, mask);
    return input_gradient;
}
