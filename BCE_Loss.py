import numpy as np

class Binary_Cross_Entropy_Loss:
    def forward(self,y,preds):
        preds = np.clip(preds,1e-7, 1 - 1e-7)
        loss = - (y * np.log(preds) + (1-y) * np.log(1-preds))
        return loss

    def backward(self, y, preds):
        grad = - (y/preds - (1-y)/(1-preds))
        return grad