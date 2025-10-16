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
        self.errors_history_scaled = []
        self.errors_history_real = []
        self.weights_history = []
        self._yscaler = None

    def _transform_y(self, y, range_to, y_min=None, y_max=None):
        """Transforma y al rango especificado usando MinMax manual."""
        if y_min is None:
            y_min = y.min()
        if y_max is None:
            y_max = y.max()
            
        a, b = range_to
        
        if np.isclose(y_min, y_max):
            return np.full_like(y, a, dtype=float), {"a": a, "b": b, "y_min": y_min, "y_max": y_max}
        else:
            y_scaled = a + (y - y_min) * (b - a) / (y_max - y_min)
            return y_scaled, {"a": a, "b": b, "y_min": y_min, "y_max": y_max}

    def _inverse_transform_y(self, y_scaled, scaler_dict):
        """Transforma inversa de y escalado a valores originales."""
        a, b, y_min, y_max = scaler_dict["a"], scaler_dict["b"], scaler_dict["y_min"], scaler_dict["y_max"]
        
        if np.isclose(y_min, y_max):
            return np.full_like(y_scaled, y_min, dtype=float)
        else:
            return y_min + (y_scaled - a) * (y_max - y_min) / (b - a)
        
    def get_testmse_history(self, X_test, y_test):
        y_true = np.asarray(y_test, dtype=float).ravel()
        mses = []

        for W, b in self.weights_history:
            # forward con pesos/bias de ese "epoch"
            s = np.dot(X_test, W) + b
            y_pred_scaled = self.activation_function(s)

            # volver a escala real si hubo escalado
            if self._yscaler is not None:
                y_pred = self._inverse_transform_y(y_pred_scaled, self._yscaler)
            else:
                y_pred = y_pred_scaled

            err = y_true - y_pred
            mse = float(np.mean(err ** 2))
            mses.append(mse)

        return mses

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
        return 1

    def predict(self, inputs):
        """
        Realiza una predicción para las entradas dadas.
        """
        summation = np.dot(inputs, self.weights) + self.bias
        out = self.activation_function(summation)
        if self._yscaler is not None:
            out = self._inverse_transform_y(out, self._yscaler)
        return out
    
    def _range_for_act(self):
        if self.activation_type == "sigmoid":
            return (0, 1)
        elif self.activation_type == "tanh":
            return (-1, 1)
        return None  # lineal -> no escalo

    def train(self, training_inputs, y, epochs=1000, verbose=True, min_mse=0.0001, y_min=None, y_max=None):
        """
        Entrena el perceptrón no lineal.
        """
        self.errors_history_scaled = []
        self.errors_history_real = []
        self.weights_history = []
        
        if verbose:
            print(f"Entrenando perceptrón no lineal ({self.activation_type})...")

        y_orig = np.asarray(y).ravel()

        # Escalado de y si corresponde
        rng = self._range_for_act()
        if rng is not None:
            y_scaled, self._yscaler = self._transform_y(y_orig, rng, y_min=y_min, y_max=y_max)
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
                delta = error * self.activation_derivative(summation)
                self.weights += self.learning_rate * delta * inputs
                self.bias += self.learning_rate * delta

            self.weights_history.append((self.weights.copy(), self.bias))
            
            # FASE 2: Cálculo del error DESPUÉS de actualizar todos los pesos
            total_error = 0
            total_error_real = 0
            
            # Recalcular predicciones con los pesos FINALES de esta época
            for inputs, label, label_real in zip(training_inputs, y_scaled, y_orig):
                summation = np.dot(inputs, self.weights) + self.bias
                prediction = self.activation_function(summation)
                error = label - prediction
                total_error += error**2
                
                if rng is not None:
                    prediction_real = self._inverse_transform_y(prediction, self._yscaler)
                    error_real = label_real - prediction_real
                    total_error_real += error_real**2
            
            mse = float(total_error / len(training_inputs))
            self.errors_history_scaled.append(mse)
            
            if rng is not None:
                mse_real = float(total_error_real / len(training_inputs))
            else:
                mse_real = mse

            self.errors_history_real.append(mse_real)

            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}/{epochs} - MSE: {mse_real:.6f}")
                
            if mse_real < min_mse:
                if verbose:
                    print("Convergencia alcanzada.")
                return True
                
        if verbose:
            print("No se alcanzó la convergencia deseada.")
        return False