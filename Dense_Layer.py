import numpy as np

X = [[1,2],[3,4],[3,5],[5,1]]

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs,n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, X):
        output = np.dot(X, self.weights) + self.bias
        return output
    

layer1 = Dense_Layer(2,3)
layer2 = Dense_Layer(3,1)

output1 = layer1.forward(X)
output2 = layer2.forward(output1)
print(output2)