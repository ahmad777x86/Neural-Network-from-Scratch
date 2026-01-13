#include <vector>

class BCE
{
public:
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &y, std::vector<std::vector<double>> &preds);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &grad);
};