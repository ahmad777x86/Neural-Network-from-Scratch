class Sequential_Model:
    def __init__(self, Layers):
        self.Layers = Layers
    
    def forward(self, X, verbose=True):
        self.signal = X
        for layer in self.Layers:
            self.signal = layer.forward(self.signal)
        if verbose:
            print("Prediction: ", self.Layers[-1].output)

    def backward(self, loss_grad, verbose=True):
        self.grad = loss_grad
        for layer in self.Layers[::-1]:
            self.grad = layer.backward(self.grad)
            if verbose:
                print("Gradient: ", self.grad)
        if verbose:
            print("Gradients Calculated!")