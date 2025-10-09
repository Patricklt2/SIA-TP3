# ej1/plot_all_boundaries.py
import numpy as np
import matplotlib.pyplot as plt

from perceptrons.step.perceptron import StepPerceptron  # tu clase

EPS = 1e-9  # tolerancia numérica para detectar "casi cero"


def plot_points(X, y, ax):
    """Pinta los puntos de ambas clases."""
    mask_pos = (y == 1)
    mask_neg = (y == -1)
    ax.scatter(X[mask_pos, 0], X[mask_pos, 1], marker='o', s=90, label='Clase 1')
    ax.scatter(X[mask_neg, 0], X[mask_neg, 1], marker='x', s=90, label='Clase -1')


def draw_boundary(ax, w, b, xlim, ylim, color, alpha=0.25, lw=1.5, label=None):
    """
    Dibuja SIEMPRE la frontera w1*x + w2*y + b = 0:
      - |w2|>=EPS -> y = -(w1*x + b)/w2
      - |w2|<EPS y |w1|>=EPS -> línea vertical x = -b/w1
      - ambos ~0 -> no dibuja (estado degenerado)
    """
    w1, w2 = float(w[0]), float(w[1])

    if abs(w2) >= EPS:
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, color=color, alpha=alpha, lw=lw, label=label)
    elif abs(w1) >= EPS:
        x0 = -b / w1
        ax.axvline(x0, color=color, alpha=alpha, lw=lw, linestyle='--', label=label)
    else:
        # w ~ [0,0] y b ~ 0: no hay dirección definible (inicio puro)
        pass


def plot_all_epoch_boundaries(X, y, history, title="Todas las épocas (fronteras)", save_path=None):
    """
    Superpone la frontera de decisión de TODAS las épocas en un solo eje:
      - Todas las épocas menos la última con degradé (suaves).
      - La última época resaltada en negro y más gruesa.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle=':', alpha=0.6)

    # Rango de ejes cómodo
    pad = 0.5
    xmin, xmax = X[:, 0].min() - pad, X[:, 0].max() + pad
    ymin, ymax = X[:, 1].min() - pad, X[:, 1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Puntos
    plot_points(X, y, ax)

    # Degradé para las primeras N-1 épocas
    cmap = plt.cm.plasma
    n = len(history)

    if n == 0:
        ax.legend(loc="upper right")
        plt.show()
        return

    # Dibujar todas las fronteras menos la última con degradé
    for i, (w, b) in enumerate(history[:-1]):
        label = "Épocas anteriores" if i == 0 else None
        draw_boundary(ax, w, b, (xmin, xmax), (ymin, ymax),
                      color="#6baed6", alpha=0.35, lw=1.5, label=label)

    # Última frontera resaltada en negro
    w_last, b_last = history[-1]
    draw_boundary(ax, w_last, b_last, (xmin, xmax), (ymin, ymax),
                  color="black", alpha=0.9, lw=2.5, label=f"Época {n-1} final")

    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {save_path}")

    plt.show()

def print_history(history):
    """
    Imprime el historial de pesos y bias guardados durante el entrenamiento.
    
    Args:
        history (list): Lista de tuplas (weights, bias) guardadas en cada época o actualización.
    """
    print("\nHistorial de pesos y bias:")
    for i, (w, b) in enumerate(history):
        print(f"Paso {i}: Pesos = {w}, Bias = {b}")

# ---------------------------
# Runners: AND y XOR
# ---------------------------
def run_and_all_lines():
     # Datos de entrenamiento para AND
    x_and = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    # Crear y entrenar el perceptrón
    input_dim = x_and.shape[1]
    perceptron = StepPerceptron(input_size=input_dim, learning_rate=0.1)
    perceptron.train(x_and, y_and, epochs=10)
    print_history(perceptron.history)
    plot_all_epoch_boundaries(x_and, y_and, perceptron.history,
                              title="Perceptrón simple – AND (todas las épocas)",
                              save_path="all_lines_AND.png")


def run_xor_all_lines():
    # Datos de entrenamiento para XOR
    x_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_xor = np.array([1, 1, -1, -1])

    # Crear y entrenar el perceptrón
    input_dim = x_xor.shape[1]
    perceptron = StepPerceptron(input_size=input_dim, learning_rate=0.1)
    perceptron.train(x_xor, y_xor, epochs=10)
    print_history(perceptron.history)
    plot_all_epoch_boundaries(x_xor, y_xor, perceptron.history,
                              title="Perceptrón simple – XOR (todas las épocas)",
                              save_path="all_lines_XOR.png")


if __name__ == "__main__":
    run_and_all_lines()
    run_xor_all_lines()
