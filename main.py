import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from ReLU_Layer import ReLU_Layer
from Sigmoid_Layer import Sigmoid_Layer
from Dense_Layer import Dense_Layer
from BCE_Loss import Binary_Cross_Entropy_Loss

X = np.array([[1,1],[0,0],[0,1],[1,0]])
y = np.array([[1],[1],[0],[0]])

# model
layer1 = Dense_Layer(2,4)
layer2 = ReLU_Layer()
layer3 = Dense_Layer(4,1)
layer4 = Sigmoid_Layer()
Layers = [layer1, layer2, layer3, layer4]

# Loss
loss = Binary_Cross_Entropy_Loss()

# forward pass
def predict():
    layer1.forward(X)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    layer4.forward(layer3.output)

    print("Prediction: ", layer4.output)



# backward pass
Loss = []
for _ in range(1000):
    predict()
    Loss.append(loss.forward(y,layer4.output)[0])
    print(f"Loss: {Loss[_]}")
    loss_gradient = loss.backward(y,layer4.output)
    print("Loss grad shape: ", loss_gradient.shape)
    grad_4 = layer4.backward(loss_gradient)
    grad_3 = layer3.backward(grad_4)
    print("Input Gradient 3: ", grad_3)
    print("Weight Gradient 3: ",layer3.w_gradient)
    layer3.weights -= 0.1 * layer3.w_gradient
    layer3.bias -= 0.1 * layer3.b_gradient

    grad_2 = layer2.backward(grad_3)
    print("Gradient 2: ", grad_2)

    grad_1 = layer1.backward(grad_2)
    print("Gradient 1: ", grad_1)
    print("Weight Gradient 1: ",layer1.w_gradient)
    layer1.weights -= 0.1 * layer1.w_gradient
    layer1.bias -= 0.1 * layer1.b_gradient

predict()

plt.plot(range(1000), Loss, 'r-')
plt.title("Binary Cross Entropy Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()