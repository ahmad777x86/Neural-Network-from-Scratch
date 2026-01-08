import numpy as np

X = [[1,0],[1,1],[0,1],[0,0]]
y = [[0],[1],[0],[1]]

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons) * np.sqrt(2.0/n_inputs)
        self.bias = np.zeros((1, n_neurons))
        print("Initial Weights: ", self.weights)
        print("Initial Bias: ", self.bias)

    def forward(self, X):
        self.X = X
        self.output = np.dot(self.X, self.weights) + self.bias
        return self.output
    
    def backward(self,grad):
        self.w_gradient = np.dot(np.transpose(self.X),grad)
        self.b_gradient = np.sum(grad, axis=0, keepdims=True)
        self.inp_gradient = np.dot(grad,np.transpose(self.weights))
        return self.inp_gradient
    
    @property
    def params(self):
        return [(self.weights, self.w_gradient),(self.bias,self.b_gradient)]
    
if __name__ == "__main__":
    layer1 = Dense_Layer(2,3)
    layer2 = Dense_Layer(3,1)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print("Forward: ", layer1.output)
    print("Forward: ", layer2.output)

    grad_2 = layer2.backward(layer2.output - y)
    print("Input Gradient: ", grad_2)
    grad_1 = layer1.backward(grad_2)
    print("Input Gradient: ", grad_1)
