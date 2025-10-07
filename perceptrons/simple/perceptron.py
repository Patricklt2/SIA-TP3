import numpy as np

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
        self.errors_history = []

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
        return self.activation_function(summation)

    def train(self, training_inputs, labels, epochs=1000, verbose=True):
        """
        Entrena el perceptrón no lineal.

        Args:
            training_inputs (np.array): Datos de entrenamiento.
            labels (np.array): Etiquetas esperadas.
            epochs (int): Número máximo de épocas.
            verbose (bool): Si es True, imprime el progreso.

        Returns:
            bool: True si convergió, False en caso contrario.
        """
        self.errors_history = []
        
        if verbose:
            print(f"Entrenando perceptrón no lineal ({self.activation_type})...")
            
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += error**2
                
                # Actualización usando gradiente descendente
                delta = error * self.activation_derivative(np.dot(inputs, self.weights) + self.bias)
                self.weights += self.learning_rate * delta * inputs
                self.bias += self.learning_rate * delta
                
            mse = total_error / len(training_inputs)
            self.errors_history.append(mse)
            
            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}/{epochs} - MSE: {mse:.6f}")
                
            if mse < 0.0001:
                if verbose:
                    print("Convergencia alcanzada.")
                return True
                
        if verbose:
            print("No se alcanzó la convergencia deseada.")
        return False