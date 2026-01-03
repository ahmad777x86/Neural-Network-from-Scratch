import numpy as np

X = np.array([[2],[-4]])
y = np.array([[1],[0]])

class Sigmoid_Layer:
    def forward(self, X):
        self.X = X
        self.output = 1/(1+np.exp(-X))
        return self.output
    
    def backward(self,grad):
        self.inp_gradient = grad*(self.output) * (1 - self.output)
        return self.inp_gradient
    
if __name__ == "__main__":
    layer1 = Sigmoid_Layer()

    layer1.forward(X)
    print("Forward: ", layer1.output)

    grad_1 = layer1.backward(layer1.output - y)
    print("Input Gradient: ", grad_1)