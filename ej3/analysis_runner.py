# /home/pipemind/sia/SIA-TP3/ej3/analysis_runner.py
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Añadir el directorio raíz del proyecto a la ruta de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Sigmoid
from perceptrons.multicapa.loss import bce, bce_prime
from perceptrons.multicapa.optimizers import SGD, Momentum, Adam
from ej3.data_utils import load_digits_data, get_parity_labels

def run_analysis():
    """
    Ejecuta un análisis comparativo de hiperparámetros para el problema de paridad.
    """
    print("--- Iniciando Análisis de Hiperparámetros para el Problema de Paridad ---")

    # Cargar y preparar los datos
    file_path = os.path.join(os.path.dirname(__file__), 'TP3-ej3-digitos.txt')
    X_data, y_digits = load_digits_data(file_path)
    y_data = get_parity_labels(y_digits)
    
    epochs = 1000
    
    # --- 1. Análisis de Optimizadores ---
    print("\n[Análisis 1/2] Comparando Optimizadores...")
    
    optimizers_to_test = {
        "SGD": SGD(learning_rate=0.1),
        "Momentum": Momentum(learning_rate=0.1, momentum=0.9),
        "Adam": Adam(learning_rate=0.01) # Adam suele necesitar un learning rate más bajo
    }
    
    loss_history_optimizers = {}

    for name, optimizer in optimizers_to_test.items():
        print(f"  Entrenando con: {name}...")
        # Se define la misma arquitectura para una comparación justa
        layers = [
            Dense(input_size=35, output_size=10), Tanh(),
            Dense(input_size=10, output_size=1), Sigmoid()
        ]
        mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)
        history = mlp.train(X_data, y_data, epochs=epochs, batch_size=10, verbose=False)
        loss_history_optimizers[name] = history

    # Graficar resultados de optimizadores
    plt.figure(figsize=(10, 6))
    for name, history in loss_history_optimizers.items():
        plt.plot(history, label=name)
    plt.title("Análisis de Optimizadores: Pérdida a través de las Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (BCE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("optimizer_analysis.png")
    print("  Gráfico 'optimizer_analysis.png' guardado.")

    # --- 2. Análisis de Tasa de Aprendizaje (usando Adam) ---
    print("\n[Análisis 2/2] Comparando Tasas de Aprendizaje con Adam...")
    
    learning_rates_to_test = [0.1, 0.01, 0.001]
    loss_history_lr = {}

    for lr in learning_rates_to_test:
        print(f"  Entrenando con learning_rate: {lr}...")
        layers = [
            Dense(input_size=35, output_size=10), Tanh(),
            Dense(input_size=10, output_size=1), Sigmoid()
        ]
        optimizer = Adam(learning_rate=lr)
        mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)
        history = mlp.train(X_data, y_data, epochs=epochs, batch_size=10, verbose=False)
        loss_history_lr[f"LR={lr}"] = history

    # Graficar resultados de tasa de aprendizaje
    plt.figure(figsize=(10, 6))
    for name, history in loss_history_lr.items():
        plt.plot(history, label=name)
    plt.title("Análisis de Tasa de Aprendizaje (Adam): Pérdida a través de las Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (BCE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_rate_analysis.png")
    print("  Gráfico 'learning_rate_analysis.png' guardado.")
    
    print("\n--- Análisis Completado ---")

if __name__ == "__main__":
    run_analysis()