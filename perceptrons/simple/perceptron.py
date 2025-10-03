import numpy as np

class Perceptron:
    """
    Implementación de un Perceptrón simple.
    """
    def __init__(self, input_size, learning_rate=0.1):
        """
        Inicializa el Perceptrón.

        Args:
            input_size (int): El número de características de entrada.
            learning_rate (float): La tasa de aprendizaje.
        """
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation_function(self, x):
        """
        Función de activación escalón.
        Devuelve 1 si x >= 0, de lo contrario -1.
        """
        return 1 if x >= 0 else -1

    def predict(self, inputs):
        """
        Predice la salida para un conjunto de entradas.
        """
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels, epochs=100, verbose=True):
        """
        Entrena el perceptrón.

        Args:
            training_inputs (np.array): Un array de numpy con los datos de entrenamiento.
            labels (np.array): Un array de numpy con las etiquetas de salida esperadas.
            epochs (int): El número máximo de iteraciones sobre los datos de entrenamiento.
            verbose (bool): Si es True, imprime el progreso del entrenamiento.
        """
        if verbose:
            print("Entrenando el perceptrón...")
        for epoch in range(epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction != label:
                    errors += 1
                    update = self.learning_rate * (label - prediction)
                    self.weights += update * inputs
                    self.bias += update
            if verbose:
                print(f"Época {epoch + 1}/{epochs} - Errores: {errors}")
            if errors == 0:
                if verbose:
                    print("El perceptrón ha convergido.")
                return True
        if verbose:
            print("El perceptrón no convergió en el número de épocas dado.")
        return False
