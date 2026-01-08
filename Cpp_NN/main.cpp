#include <iostream>
#include <vector>

#include "ReLU.h"
#include "Sigmoid.h"
#include "Dense.h"
#include "BCE_Loss.h"

int main()
{
    std::vector<std::vector<double>> X = {{1, 4}, {-5, 3}};
    ReLU relu;
    Sigmoid sigmoid;

    auto X_relued = relu.forward(X);
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