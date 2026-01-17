#include <iostream>
#include <vector>
#include <cmath>

#include "ReLU.h"
#include "Sigmoid.h"
#include "Dense.h"
#include "BCE_Loss.h"
#include "Utilities/Matrix_Op.h"
#include "Utilities/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
    std::vector<std::vector<double>> X = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    std::vector<std::vector<double>> y = {{0}, {0}, {1}, {1}};

    std::vector<double> loss_history(500);
    std::vector<double> epochs(500);

    // Model
    ReLU relu;
    Sigmoid sigmoid;
    Dense dense1(2, 4);
    Dense dense2(4, 1);

    for (int i = 0; i < 500; i++)
    {
        // Forward Pass
        auto X_densed1 = dense1.forward(X);
        std::cout << "\n1st Dense Layer Output: " << std::endl;
        Matrix::print(X_densed1);

        auto X_relued = relu.forward(X_densed1);

        auto X_densed2 = dense2.forward(X_relued);
        std::cout << "\n2nd Dense Layer Output: " << std::endl;
        Matrix::print(X_densed2);

        auto X_sigmoid = sigmoid.forward(X_densed2);

        std::cout << "\nFinal Output: " << std::endl;
        Matrix::print(X_sigmoid);

        BCE Loss;
        auto loss = Loss.forward(y, X_sigmoid);
        auto loss_sum = Matrix::sum(loss);
        loss_history.push_back(loss_sum[0][0] / 4);

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
        std::cout << "\nWeight Hidden Layer Gradient: " << std::endl;
        Matrix::print(dense2.w_gradient);
        std::cout << "\nBias Hidden Layer Gradient: " << std::endl;
        Matrix::print(dense2.b_gradient);
        std::cout << "\nInput Gradient: " << std::endl;
        Matrix::print(dense2_grad);

        auto relu_grad = relu.backward(dense2_grad);
        std::cout << "\nReLU Gradient: " << std::endl;
        Matrix::print(relu_grad);

        auto dense1_grad = dense1.backward(relu_grad);
        std::cout << "\nWeight 1st Layer Gradient: " << std::endl;
        Matrix::print(dense1.w_gradient);
        std::cout << "\nBias 1st Layer Gradient: " << std::endl;
        Matrix::print(dense1.b_gradient);
        std::cout << "\nInput Gradient: " << std::endl;
        Matrix::print(dense1_grad);

        dense1.weight = Matrix::add_matrix(Matrix::multiply_scalar(dense1.w_gradient, -0.1), dense1.weight);
        dense1.bias = Matrix::add_matrix(Matrix::multiply_scalar(dense1.b_gradient, -0.1), dense1.bias);
        dense2.weight = Matrix::add_matrix(Matrix::multiply_scalar(dense2.w_gradient, -0.1), dense2.weight);
        dense2.bias = Matrix::add_matrix(Matrix::multiply_scalar(dense2.b_gradient, -0.1), dense2.bias);
    }
    // Forward Pass
    auto X_densed1 = dense1.forward(X);
    std::cout << "\n1st Dense Layer Output: " << std::endl;
    Matrix::print(X_densed1);

    auto X_relued = relu.forward(X_densed1);

    auto X_densed2 = dense2.forward(X_relued);
    std::cout << "\n2nd Dense Layer Output: " << std::endl;
    Matrix::print(X_densed2);

    auto X_sigmoid = sigmoid.forward(X_densed2);

    std::cout << "\nFinal Output: " << std::endl;
    Matrix::print(X_sigmoid);

    BCE Loss;
    auto loss = Loss.forward(y, X_sigmoid);

    std::cout << "\nLoss: " << std::endl;
    Matrix::print(loss);

    for (int i = 0; i < 500; i++)
    {
        epochs.push_back((double)i + 1);
    }

    plt::figure();
    plt::plot(epochs, loss_history);
    plt::xlabel("Epochs");
    plt::ylabel("Loss");
    plt::title("Binary Cross Entropy Loss over Epochs");

    plt::savefig("Model_Evaluation_Classification_XOR.png");

    return 0;
}