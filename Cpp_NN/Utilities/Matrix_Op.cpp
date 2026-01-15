#include "Matrix_Op.h"
#include <iostream>
#include <cmath>

int LINK_TEST = 123;

std::vector<std::vector<double>> Matrix::dot(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
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

std::vector<std::vector<double>> Matrix::Log(const std::vector<std::vector<double>> &A)
{
    std::vector<std::vector<double>> C(A.size(), std::vector<double>(A[0].size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = log(A[i][j]);
        }
    }

    return C;
}

std::vector<std::vector<double>> Matrix::multiply_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
{
    if (A.size() != B.size() || A[0].size() != B[0].size())
    {
        std::cerr << "Dimensions not same for matrix entry wise multiplication" << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<double>> C(A.size(), std::vector<double>(A[0].size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = A[i][j] * B[i][j];
        }
    }

    return C;
}

std::vector<std::vector<double>> Matrix::add_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B)
{
    if (A.size() != B.size() || A[0].size() != B[0].size())
    {
        std::cerr << "Dimensions not same for matrix entry wise addition" << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<double>> C(A.size(), std::vector<double>(A[0].size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    return C;
}

std::vector<std::vector<double>> Matrix::add_scalar(const std::vector<std::vector<double>> &A, double S)
{

    std::vector<std::vector<double>> C(A.size(), std::vector<double>(A[0].size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = A[i][j] + S;
        }
    }

    return C;
}

std::vector<std::vector<double>> Matrix::multiply_scalar(const std::vector<std::vector<double>> &A, double S)
{

    std::vector<std::vector<double>> C(A.size(), std::vector<double>(A[0].size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            C[i][j] = A[i][j] * S;
        }
    }

    return C;
}

std::vector<std::vector<double>> Matrix::clip(const std::vector<std::vector<double>> &A)
{
    auto B = A;
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            if (B[i][j] < 10e-7)
            {
                B[i][j] = 10e-7;
            }
            else if (B[i][j] > 1 - 10e-7)
            {
                B[i][j] = 1 - 10e-7;
            }
        }
    }
    return B;
}

std::vector<std::vector<double>> Matrix::reciprocate(const std::vector<std::vector<double>> &A)
{
    auto A_r = A;
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            A_r[i][j] = 1 / A[i][j];
        }
    }
    return A_r;
}
std::vector<std::vector<double>> Matrix::transpose(const std::vector<std::vector<double>> &A)
{
    std::vector<std::vector<double>> A_t(A[0].size(), std::vector<double>(A.size(), 0.0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[0].size(); j++)
        {
            A_t[j][i] = A[i][j];
        }
    }
    return A_t;
}

std::vector<std::vector<double>> Matrix::sum(const std::vector<std::vector<double>> &A, int axis)
{
    if (axis == 0)
    {
        std::vector<std::vector<double>> output(1, std::vector<double>(A[0].size(), 0.0));

        for (int i = 0; i < A[0].size(); i++)
        {
            for (int j = 0; j < A.size(); j++)
            {
                output[0][i] += A[j][i];
            }
        }

        return output;
    }
    else
    {
        std::vector<std::vector<double>> output(A.size(), std::vector<double>(1, 0.0));

        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < A[0].size(); j++)
            {
                output[i][0] += A[i][j];
            }
        }
        return output;
    }
}

void Matrix::print(const std::vector<std::vector<double>> &A)
{
    for (auto &i : A)
    {
        for (auto &j : i)
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}