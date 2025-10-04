# /home/pipemind/sia/SIA-TP3/perceptrons/multicapa/loss.py
import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def bce(y_true, y_pred):
    """Binary Cross-Entropy"""
    # Evitar log(0) con un valor pequeño (epsilon)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_prime(y_true, y_pred):
    """Derivada de Binary Cross-Entropy"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def cce(y_true, y_pred):
    """Categorical Cross-Entropy"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def cce_prime(y_true, y_pred):
    """
    Derivada de CCE con Softmax.
    La derivada combinada es simplemente (y_pred - y_true).
    """
    return y_pred - y_true