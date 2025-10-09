import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SimplePerceptron:
    """
    Implementación de un Perceptrón con activaciones no lineales.
    """
    def __init__(self, input_size, learning_rate=0.001, activation='tanh', beta=0.5):
        """
        Inicializa el Perceptrón no lineal.

        Args:
            input_size (int): Número de características de entrada.
            learning_rate (float): Tasa de aprendizaje.
            activation (str): Tipo de activación ('tanh', 'sigmoid', 'relu', 'linear').
            beta (float): Parámetro beta para las funciones de activación.
        """
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.activation_type = activation
        self.beta = beta
        self.errors_history_scaled = []
        self.errors_history_real = []
        self._yscaler = None

    def activation_function(self, x):
        """
        Aplica la función de activación seleccionada.
        """
        if self.activation_type == 'linear':
            return x
        elif self.activation_type == 'tanh':
            return np.tanh(self.beta * x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-self.beta * x))
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        return x

    def activation_derivative(self, x):
        """
        Calcula la derivada de la función de activación.
        """
        if self.activation_type == 'linear':
            return np.ones_like(x)
        elif self.activation_type == 'tanh':
            return self.beta * (1 - np.tanh(self.beta * x)**2)
        elif self.activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.beta * x))
            return self.beta * sig * (1 - sig)
        elif self.activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        return 1

    def predict(self, inputs):
        """
        Realiza una predicción para las entradas dadas.
        """
        summation = np.dot(inputs, self.weights) + self.bias
        out = self.activation_function(summation)
        if self._yscaler is not None:
            out = self._yscaler.inverse_transform(out.reshape(-1,1)).ravel()
        return out
    
    def _range_for_act(self):
        if self.activation_type == "sigmoid":
            return (0, 1)
        if self.activation_type == "tanh":
            return (-1, 1)
        return None  # lineal -> no escalo

    def train(self, training_inputs, y, epochs=1000, verbose=True):
        """
        Entrena el perceptrón no lineal.
        """
        self.errors_history_scaled = []
        self.errors_history_real = []
        
        if verbose:
            print(f"Entrenando perceptrón no lineal ({self.activation_type})...")

        y_orig = np.asarray(y).ravel()

        # Escalado de y si corresponde
        rng = self._range_for_act()
        if rng is not None:
            self._yscaler = MinMaxScaler(feature_range=rng)
            y_scaled = self._yscaler.fit_transform(y_orig.reshape(-1, 1)).ravel()
        else:
            self._yscaler = None
            y_scaled = y_orig
        
        for epoch in range(epochs):
            # FASE 1: Actualización de pesos (patrón por patrón)
            for inputs, label in zip(training_inputs, y_scaled):
                summation = np.dot(inputs, self.weights) + self.bias
                prediction = self.activation_function(summation)
                error = label - prediction
                
                # Actualización usando gradiente descendente
                delta = error * self.activation_derivative(np.dot(inputs, self.weights) + self.bias)
                self.weights += self.learning_rate * delta * inputs
                self.bias += self.learning_rate * delta
            
            # FASE 2: Cálculo del error (después de actualizar todos los pesos)
            total_error = 0
            total_error_real = 0
            
            for inputs, label, label_real in zip(training_inputs, y_scaled, y_orig):
                summation = np.dot(inputs, self.weights) + self.bias
                prediction = self.activation_function(summation)
                error = label - prediction
                total_error += error**2
                
                if rng is not None:
                    prediction_real = self._yscaler.inverse_transform(prediction.reshape(-1,1)).ravel()
                    error_real = label_real - prediction_real
                    total_error_real += error_real**2
            
            mse = float(total_error / len(training_inputs))
            self.errors_history_scaled.append(mse)
            
            if rng is not None:
                mse_real = float(total_error_real / len(training_inputs))
                self.errors_history_real.append(mse_real)
            else:
                self.errors_history_real.append(mse)

            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}/{epochs} - MSE: {mse:.6f}")
                
            if mse < 0.0001:
                if verbose:
                    print("Convergencia alcanzada.")
                return True
                
        if verbose:
            print("No se alcanzó la convergencia deseada.")
        return False