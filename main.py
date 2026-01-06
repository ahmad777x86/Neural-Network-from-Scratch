import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from ReLU_Layer import ReLU_Layer
from Sigmoid_Layer import Sigmoid_Layer
from Dense_Layer import Dense_Layer
from BCE_Loss import Binary_Cross_Entropy_Loss
from Seq_NN_Model import Sequential_Model

X = np.array([[1,1],[0,0],[0,1],[1,0]])
y = np.array([[1],[1],[0],[0]])

# model
layer1 = Dense_Layer(2,4)
layer2 = ReLU_Layer()
layer3 = Dense_Layer(4,1)
layer4 = Sigmoid_Layer()
Layers = [layer1, layer2, layer3, layer4]
model = Sequential_Model(Layers)

# Loss
loss = Binary_Cross_Entropy_Loss()

# forward pass
model.forward(X)

# backward pass
Loss = []
for _ in range(1000):

    model.forward(X)
    Loss.append(loss.forward(y,layer4.output)[0])
    print(f"Loss: {Loss[_]}")

    loss_gradient = loss.backward(y,layer4.output)
    print("Loss grad shape: ", loss_gradient.shape)

    model.backward(loss_gradient)

    layer3.weights -= 0.1 * layer3.w_gradient
    layer3.bias -= 0.1 * layer3.b_gradient
    layer1.weights -= 0.1 * layer1.w_gradient
    layer1.bias -= 0.1 * layer1.b_gradient

model.forward(X)

plt.plot(range(1000), Loss, 'r-')
plt.title("Binary Cross Entropy Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()