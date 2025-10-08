# ej2/compare_activations.py
import os, json, argparse
import numpy as np
import pandas as pd
from ej2.runner import run  # tu runner con run(cfg)

DEF_BETAS = [0.2, 0.5, 1.0, 1.5, 2.0]
DEF_LRS   = [1e-5, 1e-4, 1e-3, 1e-2]  # ajustá si querés

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_list(s, cast=float):
    if not s: return None
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def run_cfg_and_read(cfg):
    out_csv = cfg["output_csv"]
    ensure_dir(os.path.dirname(out_csv) or ".")
    df = run(cfg)  # si tu runner no devuelve df, lo leemos
    if df is None:
        df = pd.read_csv(out_csv)
    return df

def phase_beta_sweep(base, results_dir, betas, reps):
    """Barre beta para tanh y sigmoid con el LR base del JSON. Devuelve dict con best_beta por activación y las filas de trials."""
    trials = []
    best_beta = {}  # activation -> beta

    for act in ["tanh", "sigmoid"]:
        best_avg = np.inf
        best_b   = None
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
            df = run_cfg_and_read(cfg)
            avg = float(df["mse_test"].mean())
            std = float(df["mse_test"].std(ddof=1)) if len(df) > 1 else 0.0
            trials.append({
                "phase": "beta",
                "activation": act, "beta": beta, "learning_rate": cfg["learning_rate"],
                "epochs": cfg["epochs"], "kfolds": cfg["kfolds"], "test_fold": cfg["test_fold"],
                "shuffle": cfg["shuffle"], "repetitions": reps, "seed_base": cfg["seed"],
                "avg_mse_test": avg, "std_mse_test": std, "run_csv": cfg["output_csv"]
            })
            if avg < best_avg:
                best_avg = avg
                best_b = beta
        best_beta[act] = best_b
        print(f"➡️ Mejor β para {act}: {best_b} (MSE_test promedio={best_avg:.6f})")
    return best_beta, trials

def phase_lr_sweep(base, results_dir, lrs, reps, best_beta):
    """Barre LR para tanh/sigmoid usando su mejor beta; y para linear sin beta."""
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
            df = run_cfg_and_read(cfg)
            avg = float(df["mse_test"].mean())
            std = float(df["mse_test"].std(ddof=1)) if len(df) > 1 else 0.0
            trials.append({
                "phase": "lr",
                "activation": act, "beta": beta_star, "learning_rate": lr,
                "epochs": cfg["epochs"], "kfolds": cfg["kfolds"], "test_fold": cfg["test_fold"],
                "shuffle": cfg["shuffle"], "repetitions": reps, "seed_base": cfg["seed"],
                "avg_mse_test": avg, "std_mse_test": std, "run_csv": cfg["output_csv"]
            })

    # linear solo LR
    for lr in lrs:
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
        df = run_cfg_and_read(cfg)
        avg = float(df["mse_test"].mean())
        std = float(df["mse_test"].std(ddof=1)) if len(df) > 1 else 0.0
        trials.append({
            "phase": "lr",
            "activation": "linear", "beta": None, "learning_rate": lr,
            "epochs": cfg["epochs"], "kfolds": cfg["kfolds"], "test_fold": cfg["test_fold"],
            "shuffle": cfg["shuffle"], "repetitions": reps, "seed_base": cfg["seed"],
            "avg_mse_test": avg, "std_mse_test": std, "run_csv": cfg["output_csv"]
        })

    return trials

def main():
    ap = argparse.ArgumentParser(
        description="Comparar: 1) mejor β para tanh/sigmoid, 2) mejor LR con ese β; y linear barrido por LR. 5 reps c/u."
    )
    ap.add_argument("-c","--config", required=True,
                    help="JSON base (dataset, target, kfolds, test_fold, epochs, seed, etc.)")
    ap.add_argument("--betas", help=f"Lista de betas (default {DEF_BETAS})")
    ap.add_argument("--lrs", help=f"Lista de learning_rates (default {DEF_LRS})")
    ap.add_argument("--results_dir", default="ej2/results/compare", help="Dónde escribir CSVs")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base = json.load(f)

    betas = parse_list(args.betas) or DEF_BETAS
    lrs   = parse_list(args.lrs)   or DEF_LRS
    reps  = 5

    ensure_dir(args.results_dir)

    # --- Fase 1: elegir mejor beta por activación ---
    best_beta, trials_beta = phase_beta_sweep(base, args.results_dir, betas, reps)

    # --- Fase 2: barrer LR con beta óptimo (tanh/sigmoid) + linear por LR ---
    trials_lr = phase_lr_sweep(base, args.results_dir, lrs, reps, best_beta)

    # --- Guardar ALL y BESTS ---
    all_trials = trials_beta + trials_lr
    all_df = pd.DataFrame(all_trials)
    all_path = os.path.join(args.results_dir, "all_trials.csv")
    all_df.to_csv(all_path, index=False)

    # best por activación (en todo el set)
    best_rows = []
    for act in ["tanh", "sigmoid", "linear"]:
        sub = all_df[all_df["activation"] == act]
        if not len(sub):
            continue
        best_idx = sub["avg_mse_test"].idxmin()
        best_rows.append(sub.loc[best_idx])
    bests_df = pd.DataFrame(best_rows)
    bests_path = os.path.join(args.results_dir, "bests.csv")
    bests_df.to_csv(bests_path, index=False)

    print("\n✅ Listo.")
    print(f"- ALL:   {all_path}")
    print(f"- BESTS: {bests_path}")
    print("\nβ óptimos detectados:")
    for act in ["tanh", "sigmoid"]:
        print(f"  {act}: β* = {best_beta.get(act)}")
    if not bests_df.empty:
        cols = ["activation","beta","learning_rate","avg_mse_test","std_mse_test","run_csv"]
        print("\nMejores combinaciones finales:")
        print(bests_df[cols].to_string(index=False))
    else:
        print("\nNo hubo resultados.")

if __name__ == "__main__":
    main()
