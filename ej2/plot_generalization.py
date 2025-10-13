
import argparse, json, os
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_k(summary, outpath):
    ks = sorted(int(k) for k in summary["per_k"].keys())

    train_means = np.array([summary["per_k"][str(k)]["mean_train_over_folds"] for k in ks])
    test_means  = np.array([summary["per_k"][str(k)]["mean_test_over_folds"] for k in ks])
    train_stds  = np.array([summary["per_k"][str(k)]["std_train_over_folds"] for k in ks])
    test_stds   = np.array([summary["per_k"][str(k)]["std_test_over_folds"] for k in ks])

    # Configurar errores para evitar valores negativos
    train_err_low = np.minimum(train_means, train_stds)
    train_err_high = train_stds
    test_err_low = np.minimum(test_means, test_stds)
    test_err_high = test_stds

    x = np.arange(len(ks))
    width = 0.35

    plt.figure(figsize=(10, 6))  # Un poco más grande para mejor visualización
    bars_tr = plt.bar(x - width/2, train_means, width, label="Train MSE",
                      yerr=[train_err_low, train_err_high], capsize=5, 
                      color="#1f77b4", alpha=0.85, error_kw={'elinewidth': 1, 'markeredgewidth': 1})
    bars_te = plt.bar(x + width/2, test_means, width, label="Test MSE",
                      yerr=[test_err_low, test_err_high], capsize=5, 
                      color="#ff7f0e", alpha=0.85, error_kw={'elinewidth': 1, 'markeredgewidth': 1})

    plt.xticks(x, [str(k) for k in ks])
    plt.xlabel("Cantidad de folds (K)")
    plt.ylabel("MSE (final)")
    plt.title("MSE final promedio vs cantidad de folds")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Mejorar las etiquetas de valores
    def _add_labels(bars):
        for b in bars:
            h = b.get_height()
            # Calcular posición vertical óptima para la etiqueta
            y_pos = h + 0.1 * h  # 10% más arriba que la barra
            plt.text(b.get_x() + b.get_width()/2, y_pos,
                     f"{h:.3f}", ha="center", va="bottom", 
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    _add_labels(bars_tr)
    _add_labels(bars_te)

    # Ajustar límites del eje Y para dar espacio a las etiquetas
    y_max = max(np.max(train_means + train_stds), np.max(test_means + test_stds))
    plt.ylim(0, y_max * 1.15)  # 15% más de espacio arriba

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)  # Mayor resolución
    plt.close()



def plot_bar_folds_of_bestk(summary, outpath):
    best_k = summary["best_k"]
    folds = summary["per_k"][str(best_k)]["folds"]
    folds_idx = [f["fold"] for f in folds]
    train_means = [f["mse_train_mean"] for f in folds]
    test_means  = [f["mse_test_mean"] for f in folds]

    x = np.arange(len(folds_idx))
    width = 0.35

    plt.figure(figsize=(10, 6))  # Un poco más grande
    bars_tr = plt.bar(x - width/2, train_means, width, label="Train MSE", color="#1f77b4", alpha=0.85)
    bars_te = plt.bar(x + width/2, test_means, width, label="Test MSE", color="#ff7f0e", alpha=0.85)
    
    plt.xticks(x, [str(i) for i in folds_idx])
    plt.xlabel(f"Folds (K={best_k})")
    plt.ylabel("MSE (final)")
    plt.title("MSE final por fold dentro del mejor K")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Mejorar las etiquetas de valores
    def _add_labels(bars):
        for b in bars:
            h = b.get_height()
            # Calcular posición vertical óptima
            y_pos = h + 0.02 * max(max(train_means), max(test_means))  # Offset proporcional al máximo
            plt.text(b.get_x() + b.get_width()/2, y_pos,
                     f"{h:.3f}", ha="center", va="bottom", 
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    _add_labels(bars_tr)
    _add_labels(bars_te)

    # Ajustar límites del eje Y para dar espacio a las etiquetas
    y_max = max(max(train_means), max(test_means))
    plt.ylim(0, y_max * 1.15)  # 15% más de espacio arriba

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)  # Mayor resolución
    plt.close()


def plot_learning_curves(curves_json, outdir):
    """
    Genera dos gráficos:
      - learning_curves_best.png: todas las épocas
      - learning_curves_from60.png: zoom desde la época 60
    """
    with open(curves_json, "r") as f:
        curves = json.load(f)
    train = np.asarray(curves.get("train_mse_per_epoch", []), dtype=float)
    test  = np.asarray(curves.get("test_mse_per_epoch", []), dtype=float)
    epochs = np.arange(len(train))

    # --- Gráfico completo ---
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train, label="Train MSE", linewidth=2)
    plt.plot(np.arange(len(test)), test, label="Test MSE", linewidth=2)
    plt.title("Evolución del MSE por época (completo)")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (escala real)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curves_best.png"))
    plt.close()

    # --- Gráfico desde época 60 ---
    start = 80
    if len(train) > start:
        plt.figure(figsize=(8,6))
        plt.plot(epochs[start:], train[start:], label="Train MSE", linewidth=2)
        plt.plot(np.arange(start, len(test)), test[start:], label="Test MSE", linewidth=2)
        plt.title(f"Evolución del MSE desde la época {start}")
        plt.xlabel("Épocas")
        plt.ylabel("MSE (escala real)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"learning_curves_from{start}.png"))
        plt.close()


def plot_train_scatter(curves_json, outpath):
    with open(curves_json, "r") as f:
        curves = json.load(f)
    tr = curves.get("train_scatter", {})
    y_true = np.asarray(tr.get("y_true", []), dtype=float)
    y_pred = np.asarray(tr.get("y_pred", []), dtype=float)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=14, alpha=0.7, edgecolors="none")
    # recta ideal y=x
    if y_true.size > 0:
        minv = float(min(y_true.min(), y_pred.min()))
        maxv = float(max(y_true.max(), y_pred.max()))
        plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
    plt.xlabel("y (real)")
    plt.ylabel("ŷ (predicho) - train")
    plt.title("Dispersión y vs ŷ en TRAIN (mejor caso)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graficador para el estudio de K-folds y curvas por época.")
    ap.add_argument("--study", required=True, help="Ruta al cv_study.json generado (para barras)")
    ap.add_argument("--curves", help="Ruta al generalization_curves.json generado con --save_curves")
    ap.add_argument("--outdir", default="plots", help="Directorio de salida para PNGs")
    args = ap.parse_args()

    with open(args.study, "r") as f:
        summary = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    # Barras
    plot_bar_k(summary, os.path.join(args.outdir, "bar_mse_vs_k.png"))
    plot_bar_folds_of_bestk(summary, os.path.join(args.outdir, "bar_folds_in_bestk.png"))

    # Curvas/Scatter si se provee el archivo
    # Curvas/Scatter si se provee el archivo
    if args.curves and os.path.exists(args.curves):
        plot_learning_curves(args.curves, args.outdir)  # 👈 esto cambia
        plot_train_scatter(args.curves, os.path.join(args.outdir, "train_scatter_y_vs_yhat.png"))


    print("✅ Plots generados en", args.outdir)
