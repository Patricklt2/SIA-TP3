# /home/pipemind/sia/SIA-TP3/perceptrons/multicapa/activation.py
import numpy as np
from .layers import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient):
        return output_gradient * self.activation_prime(self.input)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        def tanh_prime(x):
            return 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input_data):
        exps = np.exp(input_data - np.max(input_data, axis=0, keepdims=True))
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        return output_gradient

class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: np.maximum(0, x),
                         lambda x: (x > 0).astype(float))
