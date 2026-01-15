#include <vector>

class Matrix
{
public:
    static std::vector<std::vector<double>> dot(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B, bool verbose = false);
    static std::vector<std::vector<double>> Log(const std::vector<std::vector<double>> &A);
    static std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B);
    static std::vector<std::vector<double>> add_matrix(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B, bool broadcast = false);
    static std::vector<std::vector<double>> add_scalar(const std::vector<std::vector<double>> &A, double S);
    static std::vector<std::vector<double>> multiply_scalar(const std::vector<std::vector<double>> &A, double S);
    static std::vector<std::vector<double>> clip(const std::vector<std::vector<double>> &A);
    static std::vector<std::vector<double>> reciprocate(const std::vector<std::vector<double>> &A);
    static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>> &A);
    static std::vector<std::vector<double>> sum(const std::vector<std::vector<double>> &A, int axis = 0);
    static void print(const std::vector<std::vector<double>> &A);
};