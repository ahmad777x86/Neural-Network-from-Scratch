import numpy as np

X = [[2],[-2]]
y = [[0],[1]]

class ReLU_Layer:
    def forward(self, X):
        self.X = X
        self.output = np.maximum(0,X)
        return self.output
    
    def backward(self,grad):
        mask = (np.array(self.X) > 0).astype(float)
        self.inp_gradient = grad * mask
        return self.inp_gradient
    
if __name__ == "__main__":
    layer1 = ReLU_Layer()

    layer1.forward(X)
    print("Forward: ", layer1.output)

    grad_1 = layer1.backward(layer1.output)
    print("Input Gradient: ", grad_1)
