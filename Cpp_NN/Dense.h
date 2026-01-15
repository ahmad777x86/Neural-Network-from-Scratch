#include <vector>

class Dense
{
public:
    std::vector<std::vector<double>> weight;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<double>> w_gradient;
    std::vector<std::vector<double>> b_gradient;
    std::vector<std::vector<double>> input;
    Dense(int n_inputs, int n_neurons);
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &X);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &grad);
};