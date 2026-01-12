#include "dot_product.h"
#include <iostream>

int LINK_TEST = 123;

std::vector<std::vector<double>> Dot_Product::compute(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B)
{
    int n = A.size();
    int m = B[0].size();
    int p = A[0].size();

    if (A[0].size() != B.size())
    {
        std::cerr << "Matrix shape mismatch: " << A[0].size() << " != " << B.size() << std::endl;
        std::exit(1);
    }

    std::cout << "A matrix" << std::endl;
    for (auto &i : A)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "B matrix" << std::endl;
    for (auto &i : B)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < p; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}