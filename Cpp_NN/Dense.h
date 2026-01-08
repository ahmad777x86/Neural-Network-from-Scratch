#include <vector>

class Dense
{
public:
    Dense(int n_inputs, int n_neurons);
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &X);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &grad);
};