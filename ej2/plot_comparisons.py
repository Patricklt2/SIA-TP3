# ej2/plot_comparisons_train.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ---------------- curvas de comparación (TRAIN) ----------------
def plot_betas_train(train_df, out_dir, act):
    sub_beta = train_df[(train_df["activation"] == act) & (train_df["phase"] == "beta")]
    if sub_beta.empty:
        return
    g = sub_beta.groupby("beta", as_index=False)[["mse_train_mean"]].agg(["mean","std"]).reset_index()
    betas = g["beta"].values
    means = g[("mse_train_mean","mean")].values
    stds  = g[("mse_train_mean","std")].fillna(0.0).values
    order = np.argsort(betas)
    betas, means, stds = betas[order], means[order], stds[order]

    plt.figure(figsize=(7,5))
    plt.errorbar(betas, means, yerr=stds, fmt='-o', capsize=5)
    i = int(np.argmin(means))
    plt.scatter([betas[i]], [means[i]], marker='*', s=180, zorder=5)
    plt.title(f"{act.upper()}: MSE_train vs β")
    plt.xlabel("β"); plt.ylabel("MSE_train promedio (±1σ)")
    plt.grid(True, alpha=0.4); plt.tight_layout()
    path = os.path.join(out_dir, f"{act}_mse_train_vs_beta.png")
    plt.savefig(path, dpi=150); plt.close()
    print("[saved]", path)

def plot_lrs_train(train_df, out_dir, act):
    if act in ("tanh","sigmoid"):
        sub_lr = train_df[(train_df["activation"] == act) & (train_df["phase"] == "lr")]
    else:
        sub_lr = train_df[(train_df["activation"] == act)]
    if sub_lr.empty:
        return
    g = sub_lr.groupby("learning_rate", as_index=False)[["mse_train_mean"]].agg(["mean","std"]).reset_index()
    lrs   = g["learning_rate"].values
    order = np.argsort(lrs)
    means = g[("mse_train_mean","mean")].values[order]
    stds  = g[("mse_train_mean","std")].fillna(0.0).values[order]

    plt.figure(figsize=(7,5))
    plt.errorbar(lrs[order], means, yerr=stds, fmt='-o', capsize=5)
    i = int(np.argmin(means))
    plt.scatter([lrs[order][i]], [means[i]], marker='*', s=180, zorder=5)
    ttl = f"{act.upper()}: MSE_train vs learning rate"
    if act in ("tanh","sigmoid"):
        ttl += " (β*)"
    plt.title(ttl); plt.xlabel("learning rate"); plt.xscale("log")
    plt.ylabel("MSE_train promedio (±1σ)")
    plt.grid(True, alpha=0.4, which="both"); plt.tight_layout()
    path = os.path.join(out_dir, f"{act}_mse_train_vs_lr.png")
    plt.savefig(path, dpi=150); plt.close()
    print("[saved]", path)

# ---------------- reconstrucción para scatter / historia ----------------
def _range_for_act(act):
    if act == "sigmoid": return (0,1)
    if act == "tanh":    return (-1,1)
    return None

def _rebuild_split_and_scalers(dataset, target, kfolds, test_fold, shuffle, seed_base):
    df = pd.read_csv(dataset)
    y_raw = df[target].values.astype(float)
    X_raw = df[[c for c in df.columns if c != target]].values.astype(float)
    kf = KFold(n_splits=int(kfolds), shuffle=bool(shuffle), random_state=int(seed_base))
    tr_idx, te_idx = list(kf.split(X_raw))[int(test_fold)-1]
    Xtr_raw, Xte_raw = X_raw[tr_idx], X_raw[te_idx]
    ytr_raw, yte_raw = y_raw[tr_idx], y_raw[te_idx]
    xsc = MinMaxScaler(feature_range=(-1,1))
    Xtr = xsc.fit_transform(Xtr_raw)
    Xte = xsc.transform(Xte_raw)
    return Xtr, Xte, ytr_raw, yte_raw

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

def _predict_from_rep_row(rep_row, dataset, target):
    act   = str(rep_row["activation"]).lower()
    beta  = float(rep_row.get("beta", 1.0)) if not pd.isna(rep_row.get("beta", np.nan)) else 1.0
    lr    = float(rep_row.get("learning_rate", 0.01))
    kf    = int(rep_row["kfolds"]); tf = int(rep_row["test_fold"])
    shuf  = bool(rep_row["shuffle"]); seed = int(rep_row["seed_base"])

    Xtr, Xte, ytr_raw, yte_raw = _rebuild_split_and_scalers(dataset, target, kf, tf, shuf, seed)

    rng = _range_for_act(act)
    ysc = None
    if rng is not None:
        ysc = MinMaxScaler(feature_range=rng)
        ysc.fit(ytr_raw.reshape(-1,1))

    # reconstruir pesos
    w_cols = sorted([c for c in rep_row.index if str(c).startswith("w_")],
                    key=lambda c: int(str(c).split("_")[1]))
    weights = np.array([float(rep_row[c]) for c in w_cols], dtype=float)
    bias    = float(rep_row["bias"])

    model = SimplePerceptron(input_size=Xtr.shape[1], learning_rate=lr, activation=act, beta=beta)
    model.weights = weights.copy()
    model.bias    = bias
    model._yscaler = ysc

    y_pred = model.predict(Xte)
    return yte_raw, np.asarray(y_pred).ravel()

def plot_ytrue_vs_ypred(y_true, y_pred, title, out_path):
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.9, label=f"Predicciones (R²={r2:.3f})")
    y_min, y_max = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    xs = np.linspace(y_min, y_max, 100)
    plt.plot(xs, xs, "k--", lw=2, label="y = x")
    plt.xlabel("y real"); plt.ylabel("y predicha"); plt.title(title)
    plt.legend(); plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print("[saved]", out_path)

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

        # --- Graficar todas las historias de MSE(real) juntas ---
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
        plt.text(epochs[-1], hist[-1], f"{hist[-1]:.3f}",
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
