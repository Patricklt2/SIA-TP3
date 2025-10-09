import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Sigmoid, Softmax
from perceptrons.multicapa.loss import bce, bce_prime, cce, cce_prime
from perceptrons.multicapa.optimizers import SGD, Momentum, Adam, AdamW
from ej3.data_utils import load_digits_data, get_parity_labels, get_one_hot_labels
from ej3.extra_graphics import plot_confusion_matrix, plot_digits_separately

def add_noise(X, noise_level=0.15):
    X_noisy = np.copy(X)
    num_pixels_to_flip = int(noise_level * X.shape[1])
    for i in range(len(X_noisy)):
        flip_indices = np.random.choice(X.shape[1], num_pixels_to_flip, replace=False)
        for idx in flip_indices:
            X_noisy[i][idx][0] = 1 - X_noisy[i][idx][0]
    return X_noisy

def run_analysis():
    file_path = os.path.join(os.path.dirname(__file__), 'TP3-ej3-digitos.txt')
    X_data, y_digits = load_digits_data(file_path)
    
    y_parity = get_parity_labels(y_digits)
    epochs_parity = 1000
    
    optimizers_to_test = {
        "SGD": SGD(learning_rate=0.1),
        "Momentum": Momentum(learning_rate=0.1, momentum=0.9),
        "Adam": Adam(learning_rate=0.01),
        "AdamW": AdamW(learning_rate=0.01, weight_decay=0.01)
    }
    loss_history_optimizers = {}
    for name, optimizer in optimizers_to_test.items():
        layers = [Dense(35, 10), Tanh(), Dense(10, 1), Sigmoid()]
        mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)
        history = mlp.train(X_data, y_parity, epochs=epochs_parity, batch_size=10, verbose=False)
        loss_history_optimizers[name] = history

    plt.figure(figsize=(10, 6))
    for name, history in loss_history_optimizers.items():
        plt.plot(history, label=name)
    plt.title("Análisis de Optimizadores (Paridad): Pérdida vs. Épocas")
    plt.xlabel("Épocas"); plt.ylabel("Pérdida (BCE)"); plt.legend(); plt.grid(True)
    plt.savefig("optimizer_analysis_parity.png")

    learning_rates_to_test = [0.1, 0.01, 0.001]
    loss_history_lr = {}
    for lr in learning_rates_to_test:
        layers = [Dense(35, 10), Tanh(), Dense(10, 1), Sigmoid()]
        optimizer = Adam(learning_rate=lr)
        mlp = MLP(layers, loss=bce, loss_prime=bce_prime, optimizer=optimizer)
        history = mlp.train(X_data, y_parity, epochs=epochs_parity, batch_size=10, verbose=False)
        loss_history_lr[f"LR={lr}"] = history

    plt.figure(figsize=(10, 6))
    for name, history in loss_history_lr.items():
        plt.plot(history, label=name)
    plt.title("Análisis de Tasa de Aprendizaje (Paridad): Pérdida vs. Épocas")
    plt.xlabel("Épocas"); plt.ylabel("Pérdida (BCE)"); plt.legend(); plt.grid(True)
    plt.savefig("learning_rate_analysis_parity.png")
    y_one_hot = get_one_hot_labels(y_digits)
    
    layers_digits = [Dense(35, 20), Tanh(), Dense(20, 10), Softmax()]
    optimizer_digits = Adam(learning_rate=0.01)
    mlp_digits = MLP(layers_digits, loss=cce, loss_prime=cce_prime, optimizer=optimizer_digits)
    
    mlp_digits.train(X_data, y_one_hot, epochs=2000, batch_size=10, verbose=False)
    model_path = os.path.join(os.path.dirname(__file__), 'ej3_digit_model.npz')
    mlp_digits.save_weights(model_path)

    predictions = mlp_digits.predict(X_data)
    predicted_digits = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_digits.flatten() == y_digits.flatten()) * 100
    plot_confusion_matrix(y_digits.flatten(), predicted_digits.flatten(), [str(i) for i in range(10)],
                          "Matriz de Confusión - Datos Originales", "confusion_matrix_clean.png")


    X_noisy = add_noise(X_data, noise_level=0.15)
    predictions_noisy = mlp_digits.predict(X_noisy)
    predicted_digits_noisy = np.argmax(predictions_noisy, axis=1)
    accuracy_noisy = np.mean(predicted_digits_noisy.flatten() == y_digits.flatten()) * 100
    plot_confusion_matrix(y_digits.flatten(), predicted_digits_noisy.flatten(), [str(i) for i in range(10)],
                          "Matriz de Confusión - Datos con Ruido (15%)", "confusion_matrix_noisy.png")
    plot_digits_separately(X_data, X_noisy, y_digits.flatten(),
                        predicted_digits_noisy.flatten(),
                        base_filename="digit_prediction")

    labels = ['Datos Originales', 'Datos con Ruido (15%)']
    accuracies = [accuracy, accuracy_noisy]
    colors = ['#1f77b4', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, accuracies, color=colors)

    ax.set_ylabel('Precisión (%)')
    ax.set_ylim(0, 110)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    filename = 'accuracy_comparison_noise.png'
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    run_analysis()