# ej2/generalization_study.py
import os, json, argparse
import numpy as np
import pandas as pd

# usa tu runner existente
from ej2.runner import run

DEF_KS = [3, 4, 5, 6, 8, 10]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_cfg_and_read(cfg):
    """Ejecuta runner.run(cfg) y devuelve el DataFrame de ese run (5 repeticiones)."""
    out_csv = cfg["output_csv"]
    ensure_dir(os.path.dirname(out_csv) or ".")
    df = run(cfg)
    if df is None:
        df = pd.read_csv(out_csv)
    return df

def summarize_run(cfg, df_run, phase, extra=None):
    mse_te_mean = float(df_run["mse_test"].mean()) if "mse_test" in df_run.columns else np.nan
    mse_te_std  = float(df_run["mse_test"].std(ddof=1)) if "mse_test" in df_run.columns and len(df_run) > 1 else np.nan
    mse_tr_mean = float(df_run["mse_train"].mean())
    mse_tr_std  = float(df_run["mse_train"].std(ddof=1)) if len(df_run) > 1 else 0.0
    row = {
        "phase": phase,
        "activation": cfg["activation"],
        "beta": cfg.get("beta", np.nan),
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "repetitions": cfg["repetitions"],
        "seed_base": cfg["seed"],
        "kfolds": cfg["kfolds"],
        "test_fold": cfg["test_fold"],
        "shuffle": cfg["shuffle"],
        "mse_test_mean": mse_te_mean,
        "mse_test_std": mse_te_std,
        "mse_train_mean": mse_tr_mean,
        "mse_train_std": mse_tr_std,
        "run_csv": cfg["output_csv"],
    }
    if extra:
        row.update(extra)
    return row

def k_sweep(base, results_dir, ks, reps):
    """Fase 1: barrido de K. Devuelve (k_best, rows). Criterio: menor mse_test_mean."""
    rows = []
    best_k = None
    best_score = np.inf

    for k in ks:
        tmp = f"k{k}.csv"
        cfg = {
            "dataset":       base["dataset"],
            "target":        base.get("target", "y"),
            "kfolds":        int(k),
            "test_fold":     1,  # en esta fase no importa cuál; el runner ya usa el fold elegido
            "shuffle":       bool(base.get("shuffle", True)),
            "activation":    base["activation"],
            "beta":          float(base.get("beta", 1.0)),
            "learning_rate": float(base.get("learning_rate", 0.01)),
            "epochs":        int(base["epochs"]),
            "repetitions":   reps,
            "seed":          int(base.get("seed", 1234)),
            "output_csv":    os.path.join(results_dir, f"tmp_k_sweep_{tmp}"),
        }
        df_run = run_cfg_and_read(cfg)
        row = summarize_run(cfg, df_run, phase="k_sweep")
        rows.append(row)

        score = row["mse_test_mean"]
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_k = k

    return best_k, rows

def fold_sweep(base, results_dir, k_best, reps):
    """Fase 2: con K fijo, probar cada fold como test (1..K). Devuelve (fold_best, rows)."""
    rows = []
    best_fold = None
    best_score = np.inf

    for tf in range(1, int(k_best) + 1):
        tmp = f"k{k_best}_fold{tf}.csv"
        cfg = {
            "dataset":       base["dataset"],
            "target":        base.get("target", "y"),
            "kfolds":        int(k_best),
            "test_fold":     int(tf),
            "shuffle":       bool(base.get("shuffle", True)),
            "activation":    base["activation"],
            "beta":          float(base.get("beta", 1.0)),
            "learning_rate": float(base.get("learning_rate", 0.01)),
            "epochs":        int(base["epochs"]),
            "repetitions":   reps,
            "seed":          int(base.get("seed", 1234)),
            "output_csv":    os.path.join(results_dir, f"tmp_fold_sweep_{tmp}"),
        }
        df_run = run_cfg_and_read(cfg)
        row = summarize_run(cfg, df_run, phase="fold_sweep")
        rows.append(row)

        score = row["mse_test_mean"]
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_fold = tf

    return best_fold, rows

def main():
    ap = argparse.ArgumentParser(
        description="Estudio de generalización para una activación con β y LR fijos: (1) elegir K óptimo, (2) elegir fold de test óptimo."
    )
    ap.add_argument("-c","--config", required=True,
                    help="JSON base con dataset, target, activation, beta, learning_rate, epochs, seed, shuffle")
    ap.add_argument("--ks", help=f"Lista de K a evaluar (default {DEF_KS})")
    ap.add_argument("--reps", type=int, default=5, help="Repeticiones por configuración (default 5)")
    ap.add_argument("--results_dir", default="ej2/results/generalization", help="Directorio de salida")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base = json.load(f)

    ks = [int(x) for x in (args.ks.split(",") if args.ks else DEF_KS)]
    reps = int(args.reps)
    ensure_dir(args.results_dir)

    # --- Fase 1: barrer K ---
    k_best, rows_k = k_sweep(base, args.results_dir, ks, reps)
    df_k = pd.DataFrame(rows_k)
    path_k = os.path.join(args.results_dir, "k_sweep.csv")
    df_k.to_csv(path_k, index=False)

    if k_best is None:
        print("❌ No se pudo determinar K óptimo (métricas no válidas). Abortando.")
        return

    print(f"➡️ K óptimo por generalización (mse_test_mean): {k_best}")

    # --- Fase 2: con K fijo, elegir fold de test ---
    fold_best, rows_f = fold_sweep(base, args.results_dir, k_best, reps)
    df_f = pd.DataFrame(rows_f)
    path_f = os.path.join(args.results_dir, f"fold_sweep_k{k_best}.csv")
    df_f.to_csv(path_f, index=False)

    if fold_best is None:
        print("❌ No se pudo determinar fold óptimo. Revisá los CSV.")
        return

    print(f"➡️ Fold óptimo para testeo (con K={k_best}): {fold_best}")

    # --- Guardar resumen final ---
    summary = {
        "activation": base["activation"],
        "beta": float(base.get("beta", 1.0)),
        "learning_rate": float(base.get("learning_rate", 0.01)),
        "epochs": int(base["epochs"]),
        "shuffle": bool(base.get("shuffle", True)),
        "seed_base": int(base.get("seed", 1234)),
        "repetitions": reps,
        "k_candidates": ks,
        "k_best": int(k_best),
        "fold_best": int(fold_best),
        "k_sweep_csv": path_k,
        "fold_sweep_csv": path_f,
        "dataset": base["dataset"]
    }
    with open(os.path.join(args.results_dir, "generalization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Pretty print ---
    print("\n== Barrido de K (mse_test_mean ± std) ==")
    if not df_k.empty:
        show = df_k[["kfolds","mse_test_mean","mse_test_std"]].sort_values("kfolds")
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n== Barrido de folds con K={k_best} ==")
    if not df_f.empty:
        show = df_f[["test_fold","mse_test_mean","mse_test_std"]].sort_values("test_fold")
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n✅ Listo. Archivos:\n- {path_k}\n- {path_f}\n- {os.path.join(args.results_dir, 'generalization_summary.json')}")

if __name__ == "__main__":
    main()
