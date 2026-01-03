import numpy as np
from sklearn.datasets import make_regression

from ReLU_Layer import ReLU_Layer
from Sigmoid_Layer import Sigmoid_Layer
from Dense_Layer import Dense_Layer

X, y = make_regression(n_samples = 4, n_features=2,random_state=42)
print(f"Features: {X}, X shape: {X.shape}")
print(f"Target: {y}, y shape: {y.shape}")


# model
layer1 = Dense_Layer(2,4)
layer2 = ReLU_Layer()
layer3 = Dense_Layer(4,1)

# forward pass
def predict():
    layer1.forward(X)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)

    print(layer3.output)


# backward pass
for _ in range(10):
    predict()
    grad_3 = layer3.backward(layer3.output - y.reshape(4,1))
    print("Input Gradient 3: ", grad_3)
    print("Weight Gradient 3: ",layer3.w_gradient)
    layer3.weights -= 0.01 * layer3.w_gradient
    layer3.bias -= 0.01 * layer3.b_gradient

    grad_2 = layer2.backward(grad_3)
    print("Gradient 2: ", grad_2)

    grad_1 = layer1.backward(grad_2)
    print("Gradient 1: ", grad_1)
    print("Weight Gradient 1: ",layer3.w_gradient)
    layer1.weights -= 0.01 * layer1.w_gradient
    layer1.bias -= 0.01 * layer1.b_gradient

predict()