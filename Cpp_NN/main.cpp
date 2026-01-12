#include <iostream>
#include <vector>

#include "ReLU.h"
#include "Sigmoid.h"
#include "Dense.h"
#include "BCE_Loss.h"
#include "Utilities/dot_product.h"

int main()
{
    std::vector<std::vector<double>> X = {{1, 4}, {-5, 3}};
    ReLU relu;
    Sigmoid sigmoid;
    Dense dense(2, 2);

    auto X_densed = dense.forward(X);
    for (auto &i : X_densed)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    auto X_relued = relu.forward(X_densed);
    auto X_sigmoid = sigmoid.forward(X_relued);

    for (auto &i : X_sigmoid)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}