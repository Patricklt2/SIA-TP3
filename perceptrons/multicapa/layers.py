# /home/pipemind/sia/SIA-TP3/perceptrons/multicapa/layers.py
import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        # Gradiente de los pesos
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        # LA CORRECCIÓN DEFINITIVA ESTÁ AQUÍ:
        # El gradiente del sesgo es la suma de los gradientes de salida a lo largo del eje del lote.
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        
        # Gradiente para propagar a la capa anterior
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Devolver los 3 gradientes
        return input_gradient, weights_gradient, bias_gradient