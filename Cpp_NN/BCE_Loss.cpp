#include "BCE_Loss.h"
#include "Utilities/Matrix_Op.h"
#include <cmath>
#include <iostream>

std::vector<std::vector<double>> BCE::forward(std::vector<std::vector<double>> &y, std::vector<std::vector<double>> &preds)
{
    std::vector<std::vector<double>> preds_clipped = Matrix::clip(preds);
    std::vector<std::vector<double>> preds_log = Matrix::Log(preds_clipped);
    std::vector<std::vector<double>> preds_log_minused = Matrix::Log(Matrix::multiply_scalar(Matrix::add_scalar(preds_clipped, -1), -1));
    std::vector<std::vector<double>> first_multiple = Matrix::multiply_matrix(y, preds_log);
    std::vector<std::vector<double>> second_multiple = Matrix::multiply_matrix(Matrix::multiply_scalar(Matrix::add_scalar(y, -1), -1), preds_log_minused);
    std::vector<std::vector<double>> loss = Matrix::multiply_scalar(Matrix::add_matrix(first_multiple, second_multiple), -1);
    return loss;
}

std::vector<std::vector<double>> BCE::backward(std::vector<std::vector<double>> &y, std::vector<std::vector<double>> &preds)
{
    std::vector<std::vector<double>> preds_clipped = Matrix::clip(preds);
    auto first_multiple = Matrix::multiply_matrix(y, Matrix::reciprocate(preds_clipped));
    auto second_multiple = Matrix::multiply_matrix(Matrix::multiply_scalar(Matrix::add_scalar(y, -1), -1), Matrix::reciprocate(Matrix::multiply_scalar(Matrix::add_scalar(preds_clipped, -1), -1)));
    auto grad = Matrix::multiply_scalar(Matrix::add_matrix(first_multiple, Matrix::multiply_scalar(second_multiple, -1)), -1);
    return grad;
}