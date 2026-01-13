#include "Matrix_Op.h"
#include <iostream>
#include <cmath>

int LINK_TEST = 123;

std::vector<std::vector<double>> Matrix::dot(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B)
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