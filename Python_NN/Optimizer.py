
class Optimizer:
    def __init__(self, lr=0.1):
        self.lr = lr 

    def step(self, model):
        for layer in model.Layers:
            for param, grad in layer.params:
                param -= self.lr * grad