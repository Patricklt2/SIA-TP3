import argparse, json, os
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_k(summary, outpath):
    ks = sorted(int(k) for k in summary["per_k"].keys() )

    train_means = []
    test_means  = []
    train_err_low = []
    train_err_up  = []
    test_err_low  = []
    test_err_up   = []

    for k in ks:
        pk = summary["per_k"][str(k)]

        # alturas de barra (ya las tenés)
        m_tr = float(pk["mean_train_over_folds"])
        m_te = float(pk["mean_test_over_folds"])
        train_means.append(m_tr)
        test_means.append(m_te)

        # distribución para barras asimétricas: medias por fold
        if "fold_train_means" in pk and "fold_test_means" in pk:
            ftm = np.asarray(pk["fold_train_means"], dtype=float)
            fem = np.asarray(pk["fold_test_means"], dtype=float)
        else:
            # fallback si viniera un json viejo
            ftm = np.asarray([f["mse_train_mean"] for f in pk["folds"]], dtype=float)
            fem = np.asarray([f["mse_test_mean"]  for f in pk["folds"]], dtype=float)

        # percentiles 25/75 (IQR) -> barras asimétricas alrededor del mean
        p25_tr, p75_tr = np.percentile(ftm, [25, 75])
        p25_te, p75_te = np.percentile(fem, [25, 75])

        train_err_low.append(max(0.0, m_tr - p25_tr))
        train_err_up.append(max(0.0, p75_tr - m_tr))
        test_err_low.append(max(0.0, m_te - p25_te))
        test_err_up.append(max(0.0, p75_te - m_te))

    train_means = np.asarray(train_means)
    test_means  = np.asarray(test_means)
    train_err_low = np.asarray(train_err_low)
    train_err_up  = np.asarray(train_err_up)
    test_err_low  = np.asarray(test_err_low)
    test_err_up   = np.asarray(test_err_up)

    x = np.arange(len(ks))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars_tr = plt.bar(x - width/2, train_means, width, label="Train MSE",
                      yerr=[train_err_low, train_err_up], capsize=5,
                      color="#1f77b4", alpha=0.85, error_kw={'elinewidth': 1, 'markeredgewidth': 1})
    bars_te = plt.bar(x + width/2, test_means, width, label="Test MSE",
                      yerr=[test_err_low, test_err_up], capsize=5,
                      color="#ff7f0e", alpha=0.85, error_kw={'elinewidth': 1, 'markeredgewidth': 1})

    plt.xticks(x, [str(k) for k in ks])
    plt.xlabel("Cantidad de folds (K)")
    plt.ylabel("MSE (final)")
    plt.title("MSE final promedio vs cantidad de folds (barras = IQR)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # etiquetas
    def _add_labels(bars):
        ymax = max( (train_means + train_err_up).max(), (test_means + test_err_up).max() )
        for b in bars:
            h = b.get_height()
            y_pos = h + 0.02 * ymax
            plt.text(b.get_x() + b.get_width()/2, y_pos, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    _add_labels(bars_tr); _add_labels(bars_te)

    y_max = max( (train_means + train_err_up).max(), (test_means + test_err_up).max() )
    plt.ylim(0, y_max * 1.15)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_bar_folds_of_bestk(summary, outpath):
    best_k = summary["best_k"]
    folds = summary["per_k"][str(best_k)]["folds"]

    folds_idx = [f["fold"] for f in folds]
    train_means = np.array([f["mse_train_mean"] for f in folds], dtype=float)
    test_means  = np.array([f["mse_test_mean"] for f in folds], dtype=float)

    # errores asimétricos por fold usando los valores individuales guardados
    train_err_low, train_err_up = [], []
    test_err_low,  test_err_up  = [], []

    for f, m_tr, m_te in zip(folds, train_means, test_means):
        rtr = np.asarray(f.get("rep_train", []), dtype=float)
        rte = np.asarray(f.get("rep_test",  []), dtype=float)
        if rtr.size >= 2:
            p25, p75 = np.percentile(rtr, [25, 75])
            train_err_low.append(max(0.0, m_tr - p25))
            train_err_up.append(max(0.0, p75 - m_tr))
        else:
            # fallback: std simétrica si no hay reps
            std = float(f.get("mse_train_std", 0.0))
            train_err_low.append(std); train_err_up.append(std)

        if rte.size >= 2:
            p25, p75 = np.percentile(rte, [25, 75])
            test_err_low.append(max(0.0, m_te - p25))
            test_err_up.append(max(0.0, p75 - m_te))
        else:
            std = float(f.get("mse_test_std", 0.0))
            test_err_low.append(std); test_err_up.append(std)

    train_err_low = np.asarray(train_err_low)
    train_err_up  = np.asarray(train_err_up)
    test_err_low  = np.asarray(test_err_low)
    test_err_up   = np.asarray(test_err_up)

    x = np.arange(len(folds_idx))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars_tr = plt.bar(x - width/2, train_means, width, label="Train MSE", color="#1f77b4", alpha=0.85,
                      yerr=[train_err_low, train_err_up], capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})
    bars_te = plt.bar(x + width/2, test_means,  width, label="Test MSE",  color="#ff7f0e", alpha=0.85,
                      yerr=[test_err_low,  test_err_up],  capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})
    
    plt.xticks(x, [str(i) for i in folds_idx])
    plt.xlabel(f"Folds (K={best_k})")
    plt.ylabel("MSE (final)")
    plt.title("MSE final por fold dentro del mejor K (barras = IQR)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    def _add_labels(bars):
        ymax = max( (train_means + train_err_up).max(), (test_means + test_err_up).max() )
        for b in bars:
            h = b.get_height()
            y_pos = h + 0.02 * ymax
            plt.text(b.get_x() + b.get_width()/2, y_pos, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    _add_labels(bars_tr); _add_labels(bars_te)

    y_max = max( (train_means + train_err_up).max(), (test_means + test_err_up).max() )
    plt.ylim(0, y_max * 1.15)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()



def plot_learning_curves_single_fold(curves_json, outdir, fold_name="best"):
    """
    Genera gráficos train vs test para un solo fold
    """
    with open(curves_json, "r") as f:
        curves = json.load(f)
    train = np.asarray(curves.get("train_mse_per_epoch", []), dtype=float)
    test  = np.asarray(curves.get("test_mse_per_epoch", []), dtype=float)
    epochs = np.arange(len(train))

    # Gráfico completo - escala lineal
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train, label="Train MSE", linewidth=2)
    plt.plot(np.arange(len(test)), test, label="Test MSE", linewidth=2)
    plt.title(f"Evolución del MSE por época - {fold_name}")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (escala lineal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}.png"))
    plt.close()

    # Gráfico completo - escala logarítmica desde época 0
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train, label="Train MSE", linewidth=2)
    plt.plot(np.arange(len(test)), test, label="Test MSE", linewidth=2)
    plt.yscale("log")
    plt.title(f"Evolución del MSE por época (escala log) - {fold_name}")
    plt.xlabel("Épocas")
    plt.ylabel("MSE (escala logarítmica)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_log.png"))
    plt.close()

    # Gráfico desde época 80 - escala lineal
    start = 80
    if len(train) > start:
        plt.figure(figsize=(8,6))
        plt.plot(epochs[start:], train[start:], label="Train MSE", linewidth=2)
        plt.plot(np.arange(start, len(test)), test[start:], label="Test MSE", linewidth=2)
        plt.title(f"Evolución del MSE desde la época {start} - {fold_name}")
        plt.xlabel("Épocas")
        plt.ylabel("MSE (escala lineal)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_from{start}.png"))
        plt.close()

        # Gráfico desde época 80 - escala logarítmica
        plt.figure(figsize=(8,6))
        plt.plot(epochs[start:], train[start:], label="Train MSE", linewidth=2)
        plt.plot(np.arange(start, len(test)), test[start:], label="Test MSE", linewidth=2)
        plt.yscale("log")
        plt.title(f"Evolución del MSE desde la época {start} (escala log) - {fold_name}")
        plt.xlabel("Épocas")
        plt.ylabel("MSE (escala logarítmica)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_from{start}_log.png"))
        plt.close()

def plot_test_scatter_single_fold(curves_json, outpath, fold_name="best"):
    """
    Genera scatter plot para un solo fold
    """
    with open(curves_json, "r") as f:
        curves = json.load(f)
    te = curves.get("test_scatter", {})
    y_true = np.asarray(te.get("y_true", []), dtype=float)
    y_pred = np.asarray(te.get("y_pred", []), dtype=float)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=14, alpha=0.7, edgecolors="none")
    if y_true.size > 0:
        minv = float(min(y_true.min(), y_pred.min()))
        maxv = float(max(y_true.max(), y_pred.max()))
        plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
    plt.xlabel("y (real)")
    plt.ylabel("ŷ (predicho) - test")
    plt.title(f"Dispersión y vs ŷ en TEST - {fold_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_learning_curves_all_folds(all_folds_json, outdir):
    """
    Genera gráficos train vs test para CADA fold individualmente
    """
    with open(all_folds_json, "r") as f:
        all_folds_data = json.load(f)
    
    # Crear gráficos individuales para cada fold
    for fold_name, fold_data in all_folds_data.items():
        train = np.asarray(fold_data.get("train_mse_per_epoch", []), dtype=float)
        test = np.asarray(fold_data.get("test_mse_per_epoch", []), dtype=float)
        epochs = np.arange(len(train))

        # Gráfico completo - escala lineal
        plt.figure(figsize=(8,6))
        plt.plot(epochs, train, label="Train MSE", linewidth=2)
        plt.plot(np.arange(len(test)), test, label="Test MSE", linewidth=2)
        plt.title(f"Evolución del MSE por época - {fold_name}")
        plt.xlabel("Épocas")
        plt.ylabel("MSE (escala lineal)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}.png"))
        plt.close()

        # Gráfico completo - escala logarítmica desde época 0
        plt.figure(figsize=(8,6))
        plt.plot(epochs, train, label="Train MSE", linewidth=2)
        plt.plot(np.arange(len(test)), test, label="Test MSE", linewidth=2)
        plt.yscale("log")
        plt.title(f"Evolución del MSE por época (escala log) - {fold_name}")
        plt.xlabel("Épocas")
        plt.ylabel("MSE (escala logarítmica)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_log.png"))
        plt.close()

        # Gráfico desde época 80 - escala lineal
        start = 80
        if len(train) > start:
            plt.figure(figsize=(8,6))
            plt.plot(epochs[start:], train[start:], label="Train MSE", linewidth=2)
            plt.plot(np.arange(start, len(test)), test[start:], label="Test MSE", linewidth=2)
            plt.title(f"Evolución del MSE desde la época {start} - {fold_name}")
            plt.xlabel("Épocas")
            plt.ylabel("MSE (escala lineal)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_from{start}.png"))
            plt.close()

            # Gráfico desde época 80 - escala logarítmica
            plt.figure(figsize=(8,6))
            plt.plot(epochs[start:], train[start:], label="Train MSE", linewidth=2)
            plt.plot(np.arange(start, len(test)), test[start:], label="Test MSE", linewidth=2)
            plt.yscale("log")
            plt.title(f"Evolución del MSE desde la época {start} (escala log) - {fold_name}")
            plt.xlabel("Épocas")
            plt.ylabel("MSE (escala logarítmica)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"learning_curves_{fold_name}_from{start}_log.png"))
            plt.close()

        # Scatter plot para este fold
        te_scatter = fold_data.get("test_scatter", {})
        y_true = np.asarray(te_scatter.get("y_true", []), dtype=float)
        y_pred = np.asarray(te_scatter.get("y_pred", []), dtype=float)

        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, s=14, alpha=0.7, edgecolors="none")
        if y_true.size > 0:
            minv = float(min(y_true.min(), y_pred.min()))
            maxv = float(max(y_true.max(), y_pred.max()))
            plt.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1)
        plt.xlabel("y (real)")
        plt.ylabel("ŷ (predicho) - test")
        plt.title(f"Dispersión y vs ŷ en TEST - {fold_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"test_scatter_{fold_name}.png"))
        plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graficador para el estudio de K-folds y curvas por época.")
    ap.add_argument("--study", required=True, help="Ruta al cv_study.json generado (para barras)")
    ap.add_argument("--curves", help="Ruta al curves_best.json generado con --save_curves")
    ap.add_argument("--all_folds_curves", help="Ruta al curves_all_folds_kX.json generado con --save_all_folds_curves")
    ap.add_argument("--outdir", default="plots", help="Directorio de salida para PNGs")
    args = ap.parse_args()

    with open(args.study, "r") as f:
        summary = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    
    # Barras
    plot_bar_k(summary, os.path.join(args.outdir, "bar_mse_vs_k.png"))
    plot_bar_folds_of_bestk(summary, os.path.join(args.outdir, "bar_folds_in_bestk.png"))

    # Curvas del mejor fold si se provee
    if args.curves and os.path.exists(args.curves):
        plot_learning_curves_single_fold(args.curves, args.outdir, "best")
        plot_test_scatter_single_fold(args.curves, os.path.join(args.outdir, "test_scatter_best.png"), "best")

    # Curvas de TODOS los folds si se provee
    if args.all_folds_curves and os.path.exists(args.all_folds_curves):
        plot_learning_curves_all_folds(args.all_folds_curves, args.outdir)

    print("✅ Plots generados en", args.outdir)