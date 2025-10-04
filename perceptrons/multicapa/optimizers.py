# /home/pipemind/sia/SIA-TP3/perceptrons/multicapa/optimizers.py
import numpy as np

class Optimizer:
    """Clase base para los optimizadores."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer, grad_weights, grad_bias):
        raise NotImplementedError

class SGD(Optimizer):
    """Optimizador de Gradiente Descendente Estocástico."""
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layer, grad_weights, grad_bias):
        layer.weights -= self.learning_rate * grad_weights
        # LA CORRECCIÓN ESTÁ AQUÍ:
        layer.bias -= self.learning_rate * grad_bias.reshape(layer.bias.shape)

class Momentum(Optimizer):
    """Optimizador con Momentum."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_weights = {}
        self.v_bias = {}

    def update(self, layer, grad_weights, grad_bias):
        layer_id = id(layer)
        if layer_id not in self.v_weights:
            self.v_weights[layer_id] = np.zeros_like(layer.weights)
            self.v_bias[layer_id] = np.zeros_like(layer.bias)

        # Actualizar velocidad (velocity)
        self.v_weights[layer_id] = self.momentum * self.v_weights[layer_id] - self.learning_rate * grad_weights
        # LA CORRECCIÓN ESTÁ AQUÍ:
        self.v_bias[layer_id] = self.momentum * self.v_bias[layer_id] - self.learning_rate * grad_bias.reshape(self.v_bias[layer_id].shape)
        
        # Actualizar pesos
        layer.weights += self.v_weights[layer_id]
        layer.bias += self.v_bias[layer_id]

class Adam(Optimizer):
    """Optimizador Adam."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = {}
        self.v_weights = {}
        self.m_bias = {}
        self.v_bias = {}
        self.t = 0

    def update(self, layer, grad_weights, grad_bias):
        self.t += 1
        layer_id = id(layer)

        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(layer.weights)
            self.v_weights[layer_id] = np.zeros_like(layer.weights)
            self.m_bias[layer_id] = np.zeros_like(layer.bias)
            self.v_bias[layer_id] = np.zeros_like(layer.bias)

        # Actualizar momentos de primer y segundo orden
        self.m_weights[layer_id] = self.beta1 * self.m_weights[layer_id] + (1 - self.beta1) * grad_weights
        self.m_bias[layer_id] = self.beta1 * self.m_bias[layer_id] + (1 - self.beta1) * grad_bias

        self.v_weights[layer_id] = self.beta2 * self.v_weights[layer_id] + (1 - self.beta2) * (grad_weights ** 2)
        self.v_bias[layer_id] = self.beta2 * self.v_bias[layer_id] + (1 - self.beta2) * (grad_bias ** 2)

        # Corregir sesgo de los momentos
        m_weights_hat = self.m_weights[layer_id] / (1 - self.beta1 ** self.t)
        m_bias_hat = self.m_bias[layer_id] / (1 - self.beta1 ** self.t)
        v_weights_hat = self.v_weights[layer_id] / (1 - self.beta2 ** self.t)
        v_bias_hat = self.v_bias[layer_id] / (1 - self.beta2 ** self.t)

        # Actualizar pesos
        layer.weights -= self.learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
        layer.bias -= self.learning_rate * m_bias_hat / (np.sqrt(v_bias_hat) + self.epsilon)
