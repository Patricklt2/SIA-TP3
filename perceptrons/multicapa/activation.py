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
    """Función de activación Softmax para la capa de salida."""
    def forward(self, input_data):
        # Se resta el máximo para estabilidad numérica
        tmp = np.exp(input_data - np.max(input_data))
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient):
        # La derivada de Softmax es compleja y usualmente se combina
        # con la derivada de la Cross-Entropy Categórica.
        # El gradiente que llega (output_gradient) ya está combinado.
        # Esta es una simplificación común en la implementación.
        return output_gradient