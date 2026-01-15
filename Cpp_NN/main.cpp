#include <iostream>
#include <vector>

#include "ReLU.h"
#include "Sigmoid.h"
#include "Dense.h"
#include "BCE_Loss.h"

int main()
{
    std::vector<std::vector<double>> X = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    std::vector<std::vector<double>> y = {{1}, {1}, {0}, {0}};

    ReLU relu;
    Sigmoid sigmoid;
    Dense dense1(2, 4);
    Dense dense2(4, 1);

    auto X_densed1 = dense1.forward(X);
    std::cout << "\n1st Dense Layer Output: " << std::endl;
    for (auto &i : X_densed1)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    auto X_densed2 = dense2.forward(X_densed1);
    std::cout << "\n2nd Dense Layer Output: " << std::endl;
    for (auto &i : X_densed2)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    auto X_relued = relu.forward(X_densed2);
    auto X_sigmoid = sigmoid.forward(X_relued);

    std::cout << "\nFinal Output: " << std::endl;
    for (auto &i : X_sigmoid)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    BCE Loss;
    auto loss = Loss.forward(y, X_sigmoid);

    std::cout << "\nLoss: " << std::endl;
    for (auto &i : loss)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    auto loss_grad = Loss.backward(y, X_sigmoid);
    std::cout << "\nLoss Gradient: " << std::endl;
    for (auto &i : loss_grad)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}