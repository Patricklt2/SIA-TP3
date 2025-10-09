# ej2/compare_activations.py
import os, json, argparse
import numpy as np
import pandas as pd
from ej2.runner import run  # tu runner con run(cfg)

# barrido
DEF_BETAS = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DEF_LRS   = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]      # para tanh/sigmoid (fase LR)
DEF_LRS_LINEAL = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # para linear

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_list(s, cast=float):
    if not s: return None
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def run_cfg_and_read(cfg):
    """Ejecuta runner.run(cfg) y devuelve el DataFrame de ese run (5 repeticiones)."""
    out_csv = cfg["output_csv"]
    ensure_dir(os.path.dirname(out_csv) or ".")
    df = run(cfg)  # si tu runner no devuelve df, lo leemos
    if df is None:
        df = pd.read_csv(out_csv)
    return df

def _row_from_run(cfg, df_run, phase):
    """
    Devuelve un dict resumen de un run:
      - mse_train_mean/std (SE USA PARA ELEGIR MEJORES)
      - (opcional) mse_test_mean/std (solo referencia)
    """
    mse_tr_mean = float(df_run["mse_train"].mean())
    mse_tr_std  = float(df_run["mse_train"].std(ddof=1)) if len(df_run) > 1 else 0.0
    mse_te_mean = float(df_run["mse_test"].mean()) if "mse_test" in df_run.columns else np.nan
    mse_te_std  = float(df_run["mse_test"].std(ddof=1)) if "mse_test" in df_run.columns and len(df_run) > 1 else np.nan

    return {
        "phase": phase,
        "activation": cfg["activation"],
        "beta": cfg.get("beta", np.nan),
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "kfolds": cfg["kfolds"],
        "test_fold": cfg["test_fold"],
        "shuffle": cfg["shuffle"],
        "repetitions": cfg["repetitions"],
        "seed_base": cfg["seed"],
        "mse_train_mean": mse_tr_mean,   # << clave
        "mse_train_std":  mse_tr_std,    # << clave
        "avg_mse_test":   mse_te_mean,   # referencia
        "std_mse_test":   mse_te_std,    # referencia
        "run_csv": cfg["output_csv"]
    }

def phase_beta_sweep(base, results_dir, betas, reps):
    """
    Barre beta para tanh y sigmoid con el LR base del JSON.
    Elige β* por menor mse_train_mean (PROMEDIO de las 5 reps de train).
    """
    trials = []
    best_beta = {}  # activation -> beta

    for act in ["tanh", "sigmoid"]:
        best_score = np.inf
        best_b = None

        for beta in betas:
            tmp_name = f"{act}_beta{beta}.csv"
            cfg = {
                "dataset":       base["dataset"],
                "target":        base.get("target", "y"),
                "kfolds":        int(base["kfolds"]),
                "test_fold":     int(base["test_fold"]),
                "shuffle":       bool(base.get("shuffle", True)),
                "activation":    act,
                "beta":          float(beta),
                "learning_rate": float(base.get("learning_rate", 0.01)),  # LR fijo en fase β
                "epochs":        int(base["epochs"]),
                "repetitions":   reps,
                "seed":          int(base.get("seed", 1234)),
                "output_csv":    os.path.join(results_dir, f"tmp_{tmp_name}")
            }
            df_run = run_cfg_and_read(cfg)
            row = _row_from_run(cfg, df_run, phase="beta")
            trials.append(row)

            score = row["mse_train_mean"]   # << usamos TRAIN
            if score < best_score:
                best_score = score
                best_b = beta

        best_beta[act] = best_b
        print(f"➡️ Mejor β para {act}: {best_b} (MSE_train promedio={best_score:.6f})")

    return best_beta, trials

def phase_lr_sweep(base, results_dir, lrs, reps, best_beta):
    """
    Barre LR para tanh/sigmoid usando su mejor beta (por train);
    y para linear barre LR sin beta.
    """
    trials = []

    # tanh & sigmoid con β óptimo
    for act in ["tanh", "sigmoid"]:
        beta_star = best_beta.get(act, None)
        if beta_star is None:
            continue
        for lr in lrs:
            tmp_name = f"{act}_beta{beta_star}_lr{lr}.csv"
            cfg = {
                "dataset":       base["dataset"],
                "target":        base.get("target", "y"),
                "kfolds":        int(base["kfolds"]),
                "test_fold":     int(base["test_fold"]),
                "shuffle":       bool(base.get("shuffle", True)),
                "activation":    act,
                "beta":          float(beta_star),
                "learning_rate": float(lr),
                "epochs":        int(base["epochs"]),
                "repetitions":   reps,
                "seed":          int(base.get("seed", 1234)),
                "output_csv":    os.path.join(results_dir, f"tmp_{tmp_name}")
            }
            df_run = run_cfg_and_read(cfg)
            row = _row_from_run(cfg, df_run, phase="lr")
            trials.append(row)

    # linear solo LR
    for lr in DEF_LRS_LINEAL:
        tmp_name = f"linear_lr{lr}.csv"
        cfg = {
            "dataset":       base["dataset"],
            "target":        base.get("target", "y"),
            "kfolds":        int(base["kfolds"]),
            "test_fold":     int(base["test_fold"]),
            "shuffle":       bool(base.get("shuffle", True)),
            "activation":    "linear",
            "beta":          float(base.get("beta", 1.0)),  # ignorado por linear
            "learning_rate": float(lr),
            "epochs":        int(base["epochs"]),
            "repetitions":   reps,
            "seed":          int(base.get("seed", 1234)),
            "output_csv":    os.path.join(results_dir, f"tmp_{tmp_name}")
        }
        df_run = run_cfg_and_read(cfg)
        row = _row_from_run(cfg, df_run, phase="lr")
        trials.append(row)

    return trials

def main():
    ap = argparse.ArgumentParser(
        description="Comparar: 1) mejor β para tanh/sigmoid por MSE_TRAIN, 2) mejor LR con ese β; y linear barrido por LR. 5 reps c/u."
    )
    ap.add_argument("-c","--config", required=True,
                    help="JSON base (dataset, target, kfolds, test_fold, epochs, seed, etc.)")
    ap.add_argument("--betas", help=f"Lista de betas (default {DEF_BETAS})")
    ap.add_argument("--lrs", help=f"Lista de learning_rates para tanh/sigmoid (default {DEF_LRS})")
    ap.add_argument("--results_dir", default="ej2/results/compare", help="Dónde escribir CSVs")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base = json.load(f)

    betas = parse_list(args.betas) or DEF_BETAS
    lrs   = parse_list(args.lrs)   or DEF_LRS
    reps  = 5

    ensure_dir(args.results_dir)

    # --- Fase 1: elegir mejor beta por activación (por MSE_TRAIN) ---
    best_beta, trials_beta = phase_beta_sweep(base, args.results_dir, betas, reps)

    # --- Fase 2: barrer LR con β* (tanh/sigmoid) + linear por LR ---
    trials_lr = phase_lr_sweep(base, args.results_dir, lrs, reps, best_beta)

    # --- Guardar ALL y BESTS ---
    all_trials = trials_beta + trials_lr
    all_df = pd.DataFrame(all_trials)
    all_path = os.path.join(args.results_dir, "all_trials.csv")
    all_df.to_csv(all_path, index=False)

    # best por activación (en todo el set) usando MSE_TRAIN
    best_rows = []
    for act in ["tanh", "sigmoid", "linear"]:
        sub = all_df[all_df["activation"] == act]
        if not len(sub):
            continue
        best_idx = sub["mse_train_mean"].idxmin()   # << clave: criterio por TRAIN
        best_rows.append(sub.loc[best_idx])

    bests_df = pd.DataFrame(best_rows)
    bests_path = os.path.join(args.results_dir, "bests.csv")
    bests_df.to_csv(bests_path, index=False)

    print("\n✅ Listo.")
    print(f"- ALL:   {all_path}")
    print(f"- BESTS: {bests_path}")
    print("\nβ óptimos detectados (por MSE_train):")
    for act in ["tanh", "sigmoid"]:
        print(f"  {act}: β* = {best_beta.get(act)}")
    if not bests_df.empty:
        cols = ["activation","beta","learning_rate","mse_train_mean","mse_train_std","run_csv"]
        print("\nMejores combinaciones finales (por MSE_train):")
        print(bests_df[cols].to_string(index=False))
    else:
        print("\nNo hubo resultados.")

if __name__ == "__main__":
    main()
