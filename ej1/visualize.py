import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, perceptron, title):
    # Graficar puntos
    for inputs, label in zip(X, y):
        if label == 1:
            plt.scatter(inputs[0], inputs[1], color="blue", marker="o", label="Clase 1" if "Clase 1" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(inputs[0], inputs[1], color="red", marker="x", label="Clase -1" if "Clase -1" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Calcular frontera de decisión: w1*x1 + w2*x2 + b = 0
    x_vals = np.linspace(-2, 2, 100)
    if perceptron.weights[1] != 0:
        y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
        plt.plot(x_vals, y_vals, "k--", label="Frontera de decisión")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()
