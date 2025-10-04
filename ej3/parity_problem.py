# /home/pipemind/sia/SIA-TP3/ej3/parity_problem.py
import numpy as np
import sys
import os

# Añadir el directorio raíz del proyecto a la ruta de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Sigmoid
from perceptrons.multicapa.loss import bce, bce_prime
from perceptrons.multicapa.optimizers import Adam
from ej3.data_utils import load_digits_data, get_parity_labels

def solve_parity_problem():
    """
    Resuelve el problema de discriminación de paridad para los dígitos 7x5.
    """
    print("\n--- Resolviendo el Problema de Discriminación de Paridad ---")

    # 1. Cargar los datos y generar las etiquetas de paridad
    file_path = os.path.join(os.path.dirname(__file__), 'TP3-ej3-digitos.txt')
    X_digits, y_digits = load_digits_data(file_path)
    y_parity = get_parity_labels(y_digits)

    # 2. Definir la arquitectura de la red
    # 35 neuronas de entrada -> 10 en capa oculta -> 1 de salida
    layers = [
        Dense(input_size=35, output_size=10),
        Tanh(),
        Dense(input_size=10, output_size=1),
        Sigmoid()
    ]

    # 3. Configurar el MLP
    optimizer = Adam(learning_rate=0.01)
    mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)

    # 4. Entrenar la red
    print("Entrenando la red para el problema de paridad...")
    # El dataset es pequeño, podemos usar todas las muestras en cada época (batch_size = 10)
    mlp.train(X_digits, y_parity, epochs=2000, batch_size=10, verbose=True)

    # 5. Evaluar la red entrenada
    print("\nResultados después del entrenamiento para Paridad:")
    predictions = mlp.predict(X_digits)
    
    correct_predictions = 0
    for i in range(len(X_digits)):
        digit = y_digits[i].item()
        expected_parity = y_parity[i].item()
        pred_value = predictions[i].item()
        
        # El umbral es 0.5 para decidir entre 0 (par) y 1 (impar)
        predicted_parity = 1 if pred_value > 0.5 else 0
        
        if predicted_parity == expected_parity:
            correct_predictions += 1
            result = "Correcto"
        else:
            result = "Incorrecto"
            
        print(f"Dígito: {digit}, Paridad Esperada: {expected_parity}, Predicción: {pred_value:.4f} ({predicted_parity}) -> {result}")

    accuracy = (correct_predictions / len(X_digits)) * 100
    print(f"\nPrecisión final: {accuracy:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    solve_parity_problem()