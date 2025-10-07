
import numpy as np
from .layers import Dense
from .loss import mse, mse_prime
import os

class MLP:
    def __init__(self, layers, loss, loss_prime, optimizer):
        """
        Inicializa el Perceptrón Multicapa.

        Args:
            layers (list): Una lista de objetos de capa que componen la red.
            loss: La función de pérdida a utilizar (ej. mse).
            loss_prime: La derivada de la función de pérdida (ej. mse_prime).
            optimizer: El optimizador para actualizar los pesos.
        """
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.optimizer = optimizer

    def forward(self, input_data):
        """Realiza la pasada hacia adelante (forward pass)."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient):
        # Este método ahora solo devuelve los gradientes, no actualiza.
        grad = loss_gradient
        layer_gradients = {}
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                grad, grad_w, grad_b = layer.backward(grad)
                layer_gradients[id(layer)] = (grad_w, grad_b)
            else:
                grad = layer.backward(grad)
        return layer_gradients

    def train(self, X_train, y_train, epochs, batch_size=1, verbose=True):
        history = []
        num_samples = len(X_train)

        for epoch in range(epochs):
            epoch_loss = 0
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                # Inicializar gradientes para el lote
                accumulated_gradients = {id(layer): [np.zeros_like(layer.weights), np.zeros_like(layer.bias)]
                                         for layer in self.layers if isinstance(layer, Dense)}
                
                x_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Procesar cada muestra en el lote y acumular gradientes
                for x, y in zip(x_batch, y_batch):
                    output = self.forward(x)
                    epoch_loss += self.loss(y, output)
                    loss_gradient = self.loss_prime(y, output)
                    layer_gradients = self.backward(loss_gradient)
                    
                    for layer_id, (grad_w, grad_b) in layer_gradients.items():
                        accumulated_gradients[layer_id][0] += grad_w
                        accumulated_gradients[layer_id][1] += grad_b

                # Actualizar pesos UNA VEZ por lote con los gradientes promediados
                for layer in self.layers:
                    if isinstance(layer, Dense):
                        grad_w_avg = accumulated_gradients[id(layer)][0] / batch_size
                        grad_b_avg = accumulated_gradients[id(layer)][1] / batch_size
                        self.optimizer.update(layer, grad_w_avg, grad_b_avg)

            epoch_loss /= num_samples
            history.append(epoch_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.6f}")
        
        return history

    def predict(self, input_data):
        """Realiza una predicción para una o más entradas."""
        result = []
        for i in range(len(input_data)):
            output = self.forward(input_data[i])
            result.append(output)
        return np.array(result)

    def save_weights(self, file_path):
        """Saves the weights and biases of all dense layers to a file."""
        weights_to_save = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                weights_to_save[f'weights_{i}'] = layer.weights
                weights_to_save[f'bias_{i}'] = layer.bias
        np.savez(file_path, **weights_to_save)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path):
        """Loads weights and biases from a file into the dense layers."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No weights file found at {file_path}")
            
        data = np.load(file_path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.weights = data[f'weights_{i}']
                layer.bias = data[f'bias_{i}']
        print(f"Model weights loaded from {file_path}")
