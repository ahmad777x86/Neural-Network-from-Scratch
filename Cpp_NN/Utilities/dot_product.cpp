#include "dot_product.h"

std::vector<std::vector<double>> Dot_Product::compute(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B)
{
    int n = A.size();
    int m = B[0].size();
    int p = A[0].size();

    std::vector<std::vector<double>> C;
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