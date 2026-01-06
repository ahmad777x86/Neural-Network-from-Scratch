class Sequential_Model:
    def __init__(self, Layers):
        self.Layers = Layers
    
    def forward(self, X):
        self.signal = X
        for layer in self.Layers:
            self.signal = layer.forward(self.signal)
        print("Prediction: ", self.Layers[-1].output)

    def backward(self, loss_grad):
        self.grad = loss_grad
        for layer in self.Layers[::-1]:
            self.grad = layer.backward(self.grad)
        print("Gradients Calculated!")