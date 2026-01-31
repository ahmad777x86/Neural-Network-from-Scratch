import numpy as np
import matplotlib.pyplot as plt

from ReLU_Layer import ReLU_Layer
from Sigmoid_Layer import Sigmoid_Layer
from Dense_Layer import Dense_Layer
from BCE_Loss import Binary_Cross_Entropy_Loss
from Seq_NN_Model import Sequential_Model
from Optimizer import Optimizer

X = np.array([[1,1],[0,0],[0,1],[1,0]])
y = np.array([[1],[1],[0],[0]])

# model
layer1 = Dense_Layer(2,4)
layer2 = ReLU_Layer()
layer3 = Dense_Layer(4,1)
layer4 = Sigmoid_Layer()
Layers = [layer1, layer2, layer3, layer4]
model = Sequential_Model(Layers)

# Optimizer
optimizer = Optimizer(0.01)

# Loss
loss = Binary_Cross_Entropy_Loss()

# forward pass
model.forward(X)

# backward pass
Loss = []
for _ in range(1000):

    model.forward(X, verbose=False)
    Loss.append(loss.forward(y,layer4.output)[0])

    loss_gradient = loss.backward(y,layer4.output)

    model.backward(loss_gradient, verbose=False)

    optimizer.step(model)

model.forward(X)

plt.plot(range(1000), Loss, 'r-')
plt.title("Binary Cross Entropy Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()