# /home/pipemind/sia/SIA-TP3/ej3/xor_problem_mlp.py
import numpy as np
import sys
import os

# Añadir el directorio raíz del proyecto a la ruta de Python
# para que podamos importar nuestros módulos de perceptrones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Sigmoid # Importar Sigmoid
from perceptrons.multicapa.loss import bce, bce_prime # Importar bce
from perceptrons.multicapa.optimizers import Adam # Usaremos Adam, es más robusto

def solve_xor_with_mlp():
    """
    Resuelve el problema de la función lógica XOR con un Perceptrón Multicapa
    configurado para clasificación binaria (0/1).
    """
    print("\n--- Resolviendo la función lógica XOR con MLP (Clasificación Binaria) ---")

    # Datos de entrenamiento para XOR con formato 0/1
    X_train = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])
    y_train = np.array([
        [[0]],
        [[1]],
        [[1]],
        [[0]]
    ])

    # Arquitectura: Capa oculta con Tanh, capa de salida con Sigmoid
    layers = [
        Dense(input_size=2, output_size=3),
        Tanh(),
        Dense(input_size=3, output_size=1),
        Sigmoid() # Sigmoid para la salida 0/1
    ]

    # Usar Adam y la pérdida BCE
    optimizer = Adam(learning_rate=0.01)
    mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)

    # Entrenar la red (converge más rápido)
    print("Entrenando la red para XOR...")
    mlp.train(X_train, y_train, epochs=1000, batch_size=4, verbose=True)

    # Probar la red entrenada
    print("\nResultados después del entrenamiento para XOR:")
    predictions = mlp.predict(X_train)
    
    for i, (expected, pred) in enumerate(zip(y_train, predictions)):
        # El umbral ahora es 0.5
        pred_rounded = 1 if pred.item() > 0.5 else 0
        is_correct = "Correcto" if pred_rounded == expected.item() else "Incorrecto"
        print(f"Entrada: {X_train[i].flatten()}, Salida Esperada: {expected.item()}, Predicción: {pred.item():.4f} ({pred_rounded}) -> {is_correct}")

    print("-" * 60)

if __name__ == "__main__":
    solve_xor_with_mlp()
