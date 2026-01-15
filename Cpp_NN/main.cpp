#include <iostream>
#include <vector>

#include "ReLU.h"
#include "Sigmoid.h"
#include "Dense.h"
#include "BCE_Loss.h"
#include "Utilities/Matrix_Op.h"

int main()
{
    std::vector<std::vector<double>> X = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    std::vector<std::vector<double>> y = {{1}, {1}, {0}, {0}};

    // Model
    ReLU relu;
    Sigmoid sigmoid;
    Dense dense1(2, 4);
    Dense dense2(4, 1);

    // Forward Pass
    auto X_densed1 = dense1.forward(X);
    std::cout << "\n1st Dense Layer Output: " << std::endl;
    Matrix::print(X_densed1);

    auto X_densed2 = dense2.forward(X_densed1);
    std::cout << "\n2nd Dense Layer Output: " << std::endl;
    Matrix::print(X_densed2);

    auto X_relued = relu.forward(X_densed2);
    auto X_sigmoid = sigmoid.forward(X_relued);

    std::cout << "\nFinal Output: " << std::endl;
    Matrix::print(X_sigmoid);

    BCE Loss;
    auto loss = Loss.forward(y, X_sigmoid);

    std::cout << "\nLoss: " << std::endl;
    Matrix::print(loss);

    // Backward Pass
    auto loss_grad = Loss.backward(y, X_sigmoid);
    std::cout << "\nLoss Gradient: " << std::endl;
    Matrix::print(loss_grad);

    auto sigmoid_grad = sigmoid.backward(loss_grad);
    std::cout << "\nSigmoid Gradient: " << std::endl;
    Matrix::print(sigmoid_grad);

    auto dense2_grad = dense2.backward(sigmoid_grad);
    std::cout << "\nWeight Gradient: " << std::endl;
    Matrix::print(dense2.w_gradient);
    std::cout << "\nBias Gradient: " << std::endl;
    Matrix::print(dense2.b_gradient);
    std::cout << "\nInput Gradient: " << std::endl;
    Matrix::print(dense2_grad);
    return 0;
}