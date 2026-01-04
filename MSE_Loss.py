import numpy as np

class Mean_Squared_Loss:
    def forward(self, y, preds):
        loss = (preds - y) ** 2 / len(preds)
        return loss

    def backward(self, y, preds):
        grad = 2 * (preds - y) / len(preds)
        return grad