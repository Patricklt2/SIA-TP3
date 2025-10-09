import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        matrix[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='Etiqueta Verdadera',
           xlabel='Etiqueta Predicha')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_digits_separately(X_original, X_noisy, y_true, y_pred_noisy, base_filename="prediction_digit"):
    num_digits = len(X_original)

    for i in range(num_digits):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
        fig.suptitle(f"Análisis del Dígito: {y_true[i]}", fontsize=16)

        axes[0].imshow(X_original[i].reshape(7, 5), cmap='gray_r')
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(X_noisy[i].reshape(7, 5), cmap='gray_r')
        axes[1].set_title("Con Ruido")
        axes[1].axis('off')
        
        pred_label = y_pred_noisy[i]
        is_correct = (pred_label == y_true[i])
        color = 'green' if is_correct else 'red'
        
        axes[2].text(0.5, 0.5, str(pred_label), fontsize=40, ha='center', va='center', color=color)
        axes[2].set_title("Predicción")
        axes[2].axis('off')
        
        filename = f"{base_filename}_{y_true[i]}.png"
        plt.savefig(filename)
        plt.close(fig)
