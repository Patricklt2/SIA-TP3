# /home/pipemind/sia/SIA-TP3/ej3/digit_classification_problem.py
import numpy as np
import sys
import os

# Añadir el directorio raíz del proyecto a la ruta de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Softmax
from perceptrons.multicapa.loss import cce, cce_prime
from perceptrons.multicapa.optimizers import Adam
from ej3.data_utils import load_digits_data, get_one_hot_labels

def add_noise(X, noise_level=0.15):
    """
    Añade ruido a los datos de entrada invirtiendo un porcentaje de los píxeles.
    
    Args:
        X (np.array): Los datos de entrada originales.
        noise_level (float): El porcentaje de píxeles a invertir (0.0 a 1.0).
        
    Returns:
        np.array: Los datos de entrada con ruido.
    """
    X_noisy = np.copy(X)
    num_pixels_to_flip = int(noise_level * X.shape[1])
    
    for i in range(len(X_noisy)):
        # Seleccionar índices de píxeles aleatorios para invertir
        flip_indices = np.random.choice(X.shape[1], num_pixels_to_flip, replace=False)
        for idx in flip_indices:
            # Invertir el píxel (0 a 1, 1 a 0)
            X_noisy[i][idx][0] = 1 - X_noisy[i][idx][0]
            
    return X_noisy

def solve_digit_classification_problem():
    """
    Resuelve el problema de clasificación de dígitos (0-9) y evalúa con ruido.
    """
    print("\n--- Resolviendo el Problema de Clasificación de Dígitos (0-9) ---")

    # 1. Cargar datos y generar etiquetas one-hot
    file_path = os.path.join(os.path.dirname(__file__), 'TP3-ej3-digitos.txt')
    X_digits, y_digits = load_digits_data(file_path)
    y_one_hot = get_one_hot_labels(y_digits)

    # 2. Definir la arquitectura de la red
    # 35 entradas -> 20 en capa oculta -> 10 en capa de salida
    layers = [
        Dense(input_size=35, output_size=20),
        Tanh(),
        Dense(input_size=20, output_size=10),
        Softmax()
    ]

    # 3. Configurar y entrenar el MLP
    optimizer = Adam(learning_rate=0.01)
    mlp = MLP(layers, loss=cce, loss_prime=cce_prime, optimizer=optimizer)
    
    print("Entrenando la red para clasificación de dígitos...")
    mlp.train(X_digits, y_one_hot, epochs=2000, batch_size=10, verbose=True)

    # 4. Evaluar con datos sin ruido
    print("\n--- Evaluación con Datos Originales (sin ruido) ---")
    predictions = mlp.predict(X_digits)
    predicted_digits = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_digits.flatten() == y_digits.flatten()) * 100
    
    for i in range(len(X_digits)):
        print(f"Dígito: {y_digits[i].item()}, Predicción: {predicted_digits[i]} -> {'Correcto' if predicted_digits[i] == y_digits[i].item() else 'Incorrecto'}")
    print(f"\nPrecisión con datos sin ruido: {accuracy:.2f}%")

    # 5. Evaluar con datos con ruido
    print("\n--- Evaluación con Datos con Ruido (15%) ---")
    X_noisy = add_noise(X_digits, noise_level=0.15)
    predictions_noisy = mlp.predict(X_noisy)
    predicted_digits_noisy = np.argmax(predictions_noisy, axis=1)
    accuracy_noisy = np.mean(predicted_digits_noisy.flatten() == y_digits.flatten()) * 100

    for i in range(len(X_noisy)):
        print(f"Dígito: {y_digits[i].item()}, Predicción con ruido: {predicted_digits_noisy[i]} -> {'Correcto' if predicted_digits_noisy[i] == y_digits[i].item() else 'Incorrecto'}")
    print(f"\nPrecisión con datos con ruido: {accuracy_noisy:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    solve_digit_classification_problem()