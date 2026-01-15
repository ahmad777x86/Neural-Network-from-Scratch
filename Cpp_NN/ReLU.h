#include <vector>

class ReLU
{
public:
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &X);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &grad);
    std::vector<std::vector<double>> input;
};