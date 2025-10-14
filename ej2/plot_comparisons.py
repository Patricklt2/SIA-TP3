# ej2/plot_comparisons_train.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation

# ---------------- utils básicos ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_trials(results_dir):
    all_path = os.path.join(results_dir, "all_trials.csv")
    bests_path = os.path.join(results_dir, "bests.csv")
    if not os.path.isfile(all_path):
        raise FileNotFoundError(f"No encontré all_trials.csv en {all_path}")
    if not os.path.isfile(bests_path):
        raise FileNotFoundError(f"No encontré bests.csv en {bests_path}")
    all_df = pd.read_csv(all_path)
    bests_df = pd.read_csv(bests_path)
    return all_df, bests_df

# ---------------- GRÁFICO DE BARRAS COMPARATIVO ----------------
def plot_comparative_bar_chart(bests_df, out_dir):
    """
    Gráfico de barras comparando los mejores modelos de cada función de activación
    con sus barras de error.
    """
    if bests_df.empty:
        print("[WARN] No hay datos para el gráfico de barras comparativo")
        return
    
    # Colores para cada función de activación
    colors = {"linear": "#7C3AED", "tanh": "#16A34A", "sigmoid": "#F59E0B"}
    
    # Preparar datos
    activations = []
    means = []
    stds = []
    colors_plot = []
    
    for _, row in bests_df.iterrows():
        act = str(row["activation"]).upper()
        activations.append(act)
        means.append(row["mse_train_mean"])
        stds.append(row["mse_train_std"])
        colors_plot.append(colors.get(str(row["activation"]).lower(), "#666666"))
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gráfico de barras con errores
    bars = ax.bar(activations, means, yerr=stds, capsize=8, 
                  color=colors_plot, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Añadir valores encima de las barras
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01 * max(means),
                f'{mean:.4f}\n±{std:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Configuraciones del gráfico
    ax.set_title('Comparación de Mejores Modelos por Función de Activación', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('MSE Train Promedio', fontsize=12)
    ax.set_xlabel('Función de Activación', fontsize=12)
    
    # Mejorar el grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Ajustar límites del eje Y para mejor visualización
    y_max = max(means) + max(stds) + 0.1 * max(means)
    ax.set_ylim(0, y_max)
    
    # Añadir información adicional en el gráfico
    best_idx = np.argmin(means)
    best_act = activations[best_idx]
    best_value = means[best_idx]
    
    ax.text(0.02, 0.98, f'Mejor modelo: {best_act} (MSE = {best_value:.4f})', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Guardar
    path = os.path.join(out_dir, "comparative_best_models_bar.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[saved]", path)

# ---------------- curvas de comparación (TRAIN) ----------------
def plot_betas_train(all_df, out_dir, act):
    """Grafica MSE_train vs beta para una activación específica"""
    sub = all_df[(all_df["activation"] == act) & (all_df["phase"] == "beta")].copy()
    if sub.empty:
        print(f"[WARN] No hay datos de beta para {act}")
        return

    betas = sub["beta"].to_numpy(float)
    means = sub["mse_train_mean"].to_numpy(float)
    stds  = sub["mse_train_std"].fillna(0.0).to_numpy(float)

    order = np.argsort(betas)
    betas, means, stds = betas[order], means[order], stds[order]

    color = "#2563EB"
    fig, ax = plt.subplots(figsize=(7,5))

    # Verificar si necesitamos escala logarítmica en Y
    ymin, ymax = np.nanmin(means - stds), np.nanmax(means + stds)
    use_log_y = ymax / max(ymin, 1e-12) > 50
    
    if use_log_y:
        # En escala logarítmica, las barras de error deben ser asimétricas
        yerr_lower = np.where(means - stds > 0, means - (means - stds), means * 0.1)
        yerr_upper = means + stds - means
        yerr = [yerr_lower, yerr_upper]
    else:
        yerr = stds

    # Barras de error
    ax.errorbar(
        betas, means, yerr=yerr,
        fmt='o-', color=color, ecolor=color,
        elinewidth=2.2, capsize=6, capthick=2,
        linewidth=2.4, markersize=7,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6
    )

    # marcar mínimo
    i = int(np.argmin(means))
    ax.scatter([betas[i]], [means[i]], marker='*', s=200, color=color, zorder=5)
    
    # Calcular posición vertical de la etiqueta
    if use_log_y:
        label_y_pos = means[i] * 1.8  # Multiplicador mayor para escala log
    else:
        data_range = np.nanmax(means) - np.nanmin(means)
        label_y_pos = means[i] + 0.25 * data_range  # Offset más grande
    
    ax.text(betas[i], label_y_pos, f"{means[i]:.4f}", 
            color=color, ha="center", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if use_log_y:
        ax.set_yscale("log")

    ax.set_title(f"{act.upper()}: MSE_train vs β")
    ax.set_xlabel("β")
    ax.set_ylabel("MSE_train promedio (±1σ)")
    ax.grid(True, alpha=0.35, linestyle=":")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{act}_mse_train_vs_beta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[saved]", path)

def plot_lrs_train(all_df, out_dir, act):
    """Grafica MSE_train vs learning rate para una activación específica"""
    if act in ("tanh","sigmoid"):
        sub = all_df[(all_df["activation"] == act) & (all_df["phase"] == "lr")].copy()
    else:
        sub = all_df[(all_df["activation"] == act)].copy()
    if sub.empty:
        print(f"[WARN] No hay datos de learning rate para {act}")
        return

    lrs   = sub["learning_rate"].to_numpy(float)
    means = sub["mse_train_mean"].to_numpy(float)
    stds  = sub["mse_train_std"].fillna(0.0).to_numpy(float)

    order = np.argsort(lrs)
    lrs, means, stds = lrs[order], means[order], stds[order]

    color_map = {"linear": "#7C3AED", "tanh": "#16A34A", "sigmoid": "#F59E0B"}
    color = color_map.get(act, "#111827")

    fig, ax = plt.subplots(figsize=(7,5))

    # Verificar si necesitamos escala logarítmica en Y
    ymin, ymax = np.nanmin(means - stds), np.nanmax(means + stds)
    use_log_y = ymax / max(ymin, 1e-12) > 50
    
    if use_log_y:
        # En escala logarítmica, las barras de error deben ser asimétricas
        yerr_lower = np.where(means - stds > 0, means - (means - stds), means * 0.1)
        yerr_upper = means + stds - means
        yerr = [yerr_lower, yerr_upper]
    else:
        yerr = stds

    # Barras de error
    ax.errorbar(
        lrs, means, yerr=yerr,
        fmt='o-', color=color, ecolor=color,
        elinewidth=2.2, capsize=6, capthick=2,
        linewidth=2.4, markersize=7,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6
    )

    # marcar mínimo
    i = int(np.argmin(means))
    ax.scatter([lrs[i]], [means[i]], marker='*', s=200, color=color, zorder=5)
    
    # Calcular posición vertical de la etiqueta
    if use_log_y:
        label_y_pos = means[i] * 2.0  # Multiplicador mayor para escala log
    else:
        data_range = np.nanmax(means) - np.nanmin(means)
        label_y_pos = means[i] + 0.3 * data_range  # Offset más grande
    
    ax.text(lrs[i], label_y_pos, f"{means[i]:.4f}", 
            color=color, ha="center", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # escala log en X con notación científica
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    
    if use_log_y:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.35, linestyle=":", which="both")

    ttl = f"{act.upper()}: MSE_train vs learning rate"
    if act in ("tanh","sigmoid"):
        ttl += " (β*)"
    ax.set_title(ttl)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("MSE_train promedio (±1σ)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"{act}_mse_train_vs_lr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[saved]", path)

def plot_best_histories(bests_df, out_dir):
    """Grafica las historias de entrenamiento de los mejores modelos"""
    plt.figure(figsize=(8,6))
    colors = {"linear": "purple", "tanh": "green", "sigmoid": "orange"}
    
    for _, row in bests_df.iterrows():
        act = str(row["activation"])
        history_json = row.get("history_train_json", "")
        
        if not history_json:
            print(f"[WARN] No hay historial de entrenamiento para {act}")
            continue
            
        try:
            hist = json.loads(history_json)
            if not hist:
                continue
                
            epochs = np.arange(1, len(hist)+1)
            plt.plot(epochs, hist, "-", linewidth=2, label=act.upper(),
                    color=colors.get(act, None), alpha=0.9)

            # marcar último valor
            y_pos = hist[-1]
            # Mover la etiqueta de sigmoid hacia arriba para evitar superposición
            if act == "sigmoid":
                y_pos = hist[-1] * 1.5  # Aumenté el multiplicador
            
            plt.text(epochs[-1], y_pos, f"{hist[-1]:.3f}",
                    fontsize=9, color=colors.get(act, "black"),
                    va="center", ha="left")
                    
        except Exception as e:
            print(f"[WARN] Error cargando historial para {act}: {e}")
            continue

    plt.yscale("log")
    plt.title("Evolución del MSE (real) — Mejores modelos por función")
    plt.xlabel("Época")
    plt.ylabel("MSE (escala real)")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    out_hist_all = os.path.join(out_dir, "all_best_train_histories.png")
    plt.tight_layout()
    plt.savefig(out_hist_all, dpi=150, bbox_inches="tight")
    plt.close()
    print("[saved]", out_hist_all)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Graficar comparaciones (TRAIN) y mejores modelos.")
    ap.add_argument("--results_dir", required=True, help="Directorio con los CSV de resultados (all_trials.csv y bests.csv)")
    ap.add_argument("--out_dir", default="ej2/results/plots", help="Directorio de salida para gráficos")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # leer datos
    all_df, bests_df = read_trials(args.results_dir)

    print(f"📊 {len(all_df)} configuraciones en all_trials.csv")
    print(f"🏆 {len(bests_df)} mejores configuraciones en bests.csv")

    # 1) MSE_train vs β (uno por función)
    for act in ["tanh","sigmoid"]:
        plot_betas_train(all_df, args.out_dir, act)

    # 2) MSE_train vs LR (uno por función)
    for act in ["tanh","sigmoid","linear"]:
        plot_lrs_train(all_df, args.out_dir, act)

    # 3) Gráfico de barras comparativo
    plot_comparative_bar_chart(bests_df, args.out_dir)

    # 4) Graficar todas las historias de MSE(real) juntas
    plot_best_histories(bests_df, args.out_dir)

    print("✅ Gráficos guardados en:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()