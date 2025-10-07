import numpy as np

class StepPerceptron:
    """
    Implementación de un Perceptrón simple con activación escalón.
    """
    def __init__(self, input_size, learning_rate=0.1):
        """
        Inicializa el Perceptrón.

        Args:
            input_size (int): El número de características de entrada.
            learning_rate (float): La tasa de aprendizaje.
        """
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.errors_history = []
        self.history = []  

    def activation_function(self, x):
        """
        Función de activación escalón.
        Devuelve 1 si x >= 0, de lo contrario -1.
        """
        return np.where(x >= 0, 1, -1)  # Changed to handle numpy arrays

    def predict(self, inputs):
        """
        Predice la salida para un conjunto de entradas.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels, epochs=100, verbose=True):
        """
        Entrena el perceptrón.

        Args:
            training_inputs (np.array): Array de numpy con los datos de entrenamiento.
            labels (np.array): Array de numpy con las etiquetas esperadas.
            epochs (int): Número máximo de iteraciones sobre los datos.
            verbose (bool): Si es True, imprime el progreso del entrenamiento.

        Returns:
            bool: True si el perceptrón convergió, False en caso contrario.
        """
        self.errors_history = []
        self.history = []  
        
        if verbose:
            print("Entrenando el perceptrón...")
        
        self.history.append((self.weights.copy(), float(self.bias)))
            
        for epoch in range(epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)[0]  # Get scalar prediction
                if prediction != label:
                    errors += 1
                    update = self.learning_rate * (label - prediction)
                    self.weights += update * inputs
                    self.bias += update
                    
            self.errors_history.append(errors)
            self.history.append((self.weights.copy(), float(self.bias)))
            
            if verbose:
                print(f"Época {epoch + 1}/{epochs} - Errores: {errors}")
                
            if errors == 0:
                if verbose:
                    print("El perceptrón ha convergido.")
                return True
                
        if verbose:
            print("El perceptrón no convergió en el número de épocas dado.")
        return False
