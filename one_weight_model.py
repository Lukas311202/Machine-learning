import random
import numpy as np
import matplotlib.pyplot as plt



X = []
Y = []

distribution = 0.1

for i in range(100):
    Xval = random.random()
    # Yval = Xval
    Yval = random.uniform(Xval - 0.2, Xval + 0.2)
    X.append(Xval)
    Y.append(Yval)

xPoints = np.array(X)
yPoints = np.array(Y)

class Model():
    weights = 0.0
    bias = 0.0
    
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias
    
    def predict(self, input : float) -> int:
        return self.weights * input + self.bias

def train(inputs : list[float], outputs : list[float], epochs = 100, learning_rate = 0.3):
    weight = 0.0
    bias = 0.0
    
    for epoch in range(epochs):
        print("Epoch ", epoch, ":\n")
        
        delta_weight = 0.0
        delta_bias = 0.0
        
        for i in range(len(inputs)):
            prediction = inputs[i] * weight + bias
            loss = pow(outputs[i] - prediction, 2) * 0.5
            delta_weight += ((prediction - outputs[i]) * inputs[i])
            delta_bias += prediction - outputs[i]
            print("loss: ", loss)
        
        
        weight -= (learning_rate / len(inputs)) * delta_weight 
        bias -= (learning_rate / len(inputs)) * delta_bias
        
        
        print("weights: {0}, Bias: {1}".format(weight, bias))
           
    m = Model(weight, bias)
    
    return m

model = train(X, Y)
# print("average loss:", model)

predition_line = []
for i in X:
    predition_line.append(model.predict(i))

plt.scatter(xPoints, yPoints)
plt.plot(xPoints, np.array(predition_line))
plt.show()