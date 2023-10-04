import numpy as np
import matplotlib.pyplot as plt
import random


class Network():
    def __init__(self, sizes : list[int], activation = "sigmoid") -> None:
        self.weights = [np.random.randn(n, m) for n, m in zip(sizes[:-1], sizes[1:])]
        self.bias = [np.random.randn(1, n) for n in sizes[1:]]
        self.activation_fn = globals()[activation]
        self.actication_derivative_fn = globals()[activation+"_derivative"]
        
        print("weights shape: ", [x.shape for x in self.weights])
        print("Bias shape: ", [x.shape for x in self.bias])
        
    def predict(self, a):
        for w, b in zip(self.weights, self.bias):
            a = self.activation_fn(np.dot(a, w) + b)
        return a
    
    def Train(self, train_set : list[tuple], learning_rate, epochs = 100, batch_size = 10):
        mini_batches = []
        k = 0
        for x in range(int(len(train_set) / batch_size)):
            mini_batches.append(train_set[k : batch_size + k])
            k += batch_size
            
            
        for x in range(epochs):
            for batch in mini_batches:
                self.update_batch(batch, learning_rate)
            print("epoch {x} completed".format(x=x))
        
        print("average loss: ", self.calculate_average_loss(train_set))
    
    def calculate_average_loss(self, dataset : list[tuple]):
        loss = 0.0
        for x, y in dataset:
            loss += pow(y - self.predict(x), 2)
        loss /= len(dataset)
        return loss
        
        # print(mini_batches)
    def update_batch(self, batch : list[tuple], learning_rate):
        weight_update = [np.zeros(x.shape) for x in self.weights]
        bias_update = [np.zeros(x.shape) for x in self.bias]
        
        for t in batch:
            new_weights, new_bias = self.backprop(t[0], t[1])
            weight_update = [w + nw for w, nw in zip(weight_update, new_weights)]
            bias_update = [b + nb for b, nb in zip(bias_update, new_bias)]
        
        self.weights = [w - nw * (learning_rate / len(batch)) for w, nw in zip(self.weights, weight_update)]
        self.bias = [b - nb * (learning_rate / len(batch)) for b, nb in zip(self.bias, bias_update)]
    
    def backprop(self, x, y):
        if isinstance(x, float): x = np.array(x)
        activations = [x]
        zs = []
        
        weight_update = [np.zeros(x.shape) for x in self.weights]
        bias_update = [np.zeros(x.shape) for x in self.bias]
        
        z = x
        for w, b in zip(self.weights, self.bias):
            z = np.dot(z, w) + b
            zs.append(z)
            activations.append(self.activation_fn(z))
        
        # print(activations)
        
        delta = self.error_derivative(activations[-1], y) * self.actication_derivative_fn(zs[-1])
        weight_update[-1] = np.dot(delta, activations[-2]).transpose()
        bias_update[-1] = delta
        
        for l in range(2, len(self.weights)):
            z = zs[-l]
            sp = self.actication_derivative_fn(z)
            delta = np.dot(delta, self.weights[-l+1].transpose()) * sp
            bias_update[-l] = delta
            weight_update[-l] = np.dot(delta.transpose(), activations[-l-1]).transpose()
        
        return weight_update, bias_update

    def error_derivative(self, prediction, target):
        return prediction - target
        
def relu(x):
    return max(0, x)        

def relu_derivative(x):
    return 0.0 if x <= 0.0 else 1.0

def sigmoid(x):
    # return x
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    # return 1.0
    return sigmoid(x) * (1 - sigmoid(x))

def linear(x):
    return x

def linear_derivative(x):
    return 1.0

model = Network([1, 1], "linear")
print(model.predict(1.0))

X = []
Y = []

for i in range(100):
    Xval = random.random()
    # Yval = Xval
    Yval = random.uniform(Xval - 0.2, Xval + 0.2)
    X.append(Xval)
    Y.append(Yval)
    
trainings_data = list(zip(X, Y))
model.Train(trainings_data, 0.1)

prediction_line = [model.predict(x)[0] for x in X]

plt.scatter(np.array(X), np.array(Y))
plt.plot(np.array(X), np.array(prediction_line))
plt.show()
