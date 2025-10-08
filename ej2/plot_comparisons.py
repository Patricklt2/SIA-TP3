# ej2/plot_comparisons_train.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from perceptrons.simple.perceptron import SimplePerceptron

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

def resolve_relative_paths(df, base_dir):
    """Normaliza run_csv: absoluto, relativo a base_dir, y subcarpetas (tanh/sigmoid/linear)."""
    if "run_csv" not in df.columns:
        return df
    def _fix(p):
        if not isinstance(p, str):
            return p
        s = str(p).strip().strip('"').strip("'")
        s = os.path.normpath(s)
        # absoluto
        if os.path.isabs(s) and os.path.isfile(s):
            return s
        # relativo a base_dir
        cand = os.path.normpath(os.path.join(base_dir, s))
        if os.path.isfile(cand):
            return cand
        # basename en base_dir
        base = os.path.basename(s)
        cand2 = os.path.normpath(os.path.join(base_dir, base))
        if os.path.isfile(cand2):
            return cand2
        # subcarpetas
        for sub in ("tanh", "sigmoid", "linear"):
            cand3 = os.path.normpath(os.path.join(base_dir, sub, base))
            if os.path.isfile(cand3):
                return cand3
        return s  # lo devolvemos igual; se filtrará después si no existe
    df = df.copy()
    df["run_csv"] = df["run_csv"].apply(_fix)
    return df

# ---------------- helpers para TRAIN ----------------
def _avg_std_mse_train_from_run(run_csv_path):
    """Lee el CSV de un run (5 repeticiones) y devuelve (mean, std) de mse_train final."""
    if not os.path.isfile(run_csv_path):
        return None, None
    df = pd.read_csv(run_csv_path)
    if "mse_train" not in df.columns:
        return None, None
    vals = df["mse_train"].astype(float).to_numpy()
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return mean, std

def build_train_summary(all_df, base_dir):
    """
    Recorre all_trials.csv, re-calcula promedio/desvío de mse_train leyendo cada run_csv.
    Devuelve un DF con columnas:
      activation, phase, beta, learning_rate, mse_train_mean, mse_train_std, run_csv, ...
    """
    rows = []
    for _, r in all_df.iterrows():
        run_csv = r.get("run_csv", "")
        # normalizamos ruta
        if isinstance(run_csv, str):
            if not os.path.isabs(run_csv):
                # ya viene normalizada por resolve_relative_paths, pero reforzamos:
                run_csv = os.path.normpath(os.path.join(base_dir, run_csv)) if not os.path.isfile(run_csv) else run_csv
        mean_tr, std_tr = _avg_std_mse_train_from_run(run_csv)
        if mean_tr is None:
            # intentamos subcarpetas con basename
            base = os.path.basename(str(r.get("run_csv", "")))
            for sub in ("tanh", "sigmoid", "linear"):
                cand = os.path.normpath(os.path.join(base_dir, sub, base))
                mean_tr, std_tr = _avg_std_mse_train_from_run(cand)
                if mean_tr is not None:
                    run_csv = cand
                    break
        if mean_tr is None:
            print(f"[WARN] No encuentro run_csv o 'mse_train' en: {r.get('run_csv')}")
            continue

        rows.append({
            "activation": r.get("activation"),
            "phase": r.get("phase", np.nan),
            "beta": r.get("beta", np.nan),
            "learning_rate": r.get("learning_rate", np.nan),
            "kfolds": r.get("kfolds"),
            "test_fold": r.get("test_fold"),
            "shuffle": r.get("shuffle"),
            "seed_base": r.get("seed_base"),
            "mse_train_mean": mean_tr,
            "mse_train_std": std_tr,
            "run_csv": run_csv
        })
    return pd.DataFrame(rows)

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
def plot_betas_train(train_df, out_dir, act):
    sub = train_df[(train_df["activation"] == act) & (train_df["phase"] == "beta")].copy()
    if sub.empty:
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

    # marcar mínimo - SUBIR MÁS LA ETIQUETA
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

def plot_lrs_train(train_df, out_dir, act):
    if act in ("tanh","sigmoid"):
        sub = train_df[(train_df["activation"] == act) & (train_df["phase"] == "lr")].copy()
    else:
        sub = train_df[(train_df["activation"] == act)].copy()
    if sub.empty:
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

    # marcar mínimo - SUBIR MÁS LA ETIQUETA
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

def _extract_history_real(row):
    for key in row.index:
        if "mse_history_real_json" in str(key).lower():
            raw = row[key]
            try:
                seq = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                seq = raw
            out=[]
            for v in seq:
                if isinstance(v,(list,tuple)): v=v[0]
                if isinstance(v,np.ndarray):   v=np.asarray(v).reshape(-1)[0]
                out.append(float(v))
            return out
    return None


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Graficar comparaciones (TRAIN) y mejores modelos.")
    ap.add_argument("-c","--config", required=True, help="JSON base (dataset, target)")
    ap.add_argument("--results_dir", default="ej2/results/compare", help="Directorio con los CSV de resultados")
    ap.add_argument("--out_dir", default="ej2/results/plots", help="Directorio de salida para gráficos")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # config base (dataset/target)
    with open(args.config, "r", encoding="utf-8") as f:
        base = json.load(f)
    dataset = base["dataset"]; target = base.get("target", "y")

    # leer y normalizar paths
    all_df, bests_df = read_trials(args.results_dir)
    all_df = resolve_relative_paths(all_df, args.results_dir)
    bests_df = resolve_relative_paths(bests_df, args.results_dir)

    # construir resumen TRAIN leyendo cada run_csv
    train_df = build_train_summary(all_df, args.results_dir)
    if train_df.empty:
        raise RuntimeError("No pude leer ningún run_csv con mse_train. Verificá los paths en all_trials.csv.")

    print(f"📊 {len(train_df)} configuraciones con mse_train agregadas.")

    # 1) MSE_train vs β (uno por función)
    for act in ["tanh","sigmoid"]:
        plot_betas_train(train_df, args.out_dir, act)

    # 2) MSE_train vs LR (uno por función)
    for act in ["tanh","sigmoid","linear"]:
        plot_lrs_train(train_df, args.out_dir, act)

    # 3) Mejor de cada función (según MSE_train promedio) → scatter + historia
    bests = []
    for act in ["tanh","sigmoid","linear"]:
        sub = train_df[train_df["activation"] == act]
        if sub.empty: 
            continue
        idx = sub["mse_train_mean"].idxmin()
        bests.append(train_df.loc[idx])
    bests = pd.DataFrame(bests)

    # 4) NUEVO: Gráfico de barras comparativo
    plot_comparative_bar_chart(bests, args.out_dir)

    # 5) Graficar todas las historias de MSE(real) juntas
    plt.figure(figsize=(8,6))
    colors = {"linear": "purple", "tanh": "green", "sigmoid": "orange"}
    legends = []

    for _, brow in bests.iterrows():
        act = str(brow["activation"])
        run_csv = brow["run_csv"]
        if not os.path.isfile(run_csv):
            print(f"[WARN] run_csv no encontrado para {act}: {run_csv}")
            continue

        df_run = pd.read_csv(run_csv)
        if "mse_train" not in df_run.columns:
            print(f"[WARN] mse_train no está en {run_csv}")
            continue

        rep = df_run.loc[df_run["mse_train"].idxmin()]
        hist = _extract_history_real(rep)
        if not hist:
            print(f"[WARN] {act}: no hay historial real en {run_csv}")
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

    plt.yscale("log")
    plt.title("Evolución del MSE (real) — Mejores modelos por función")
    plt.xlabel("Época")
    plt.ylabel("MSE (escala real)")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    out_hist_all = os.path.join(args.out_dir, "all_best_train_histories.png")
    plt.tight_layout()
    plt.savefig(out_hist_all, dpi=150, bbox_inches="tight")
    plt.close()
    print("[saved]", out_hist_all)

    print("✅ Gráficos guardados en:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()