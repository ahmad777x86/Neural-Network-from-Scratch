#include <iostream>
#include <vector>
#include "ReLU.h"

int main()
{
    std::vector<std::vector<double>> X = {{1, 4}, {-5, 3}};
    ReLU relu;
    auto X_relued = relu.forward(X);

    for (auto &i : X_relued)
    {
        for (auto &j : i)
        {
            std::cout << j;
        }
        std::cout << std::endl;
    }
    return 0;
}