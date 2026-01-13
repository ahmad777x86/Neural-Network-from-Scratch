#include <vector>

class Matrix
{
public:
    static std::vector<std::vector<double>> dot(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B);
    static std::vector<std::vector<double>> Log(const std::vector<std::vector<double>> &A);
    static std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
    static std::vector<std::vector<double>> add_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
    static std::vector<std::vector<double>> add_scalar(const std::vector<std::vector<double>> &A, double S);
    static std::vector<std::vector<double>> multiply_scalar(const std::vector<std::vector<double>> &A, double S);
    static std::vector<std::vector<double>> clip(const std::vector<std::vector<double>> &A);
};