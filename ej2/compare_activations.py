# ej2/compare_activations.py
import os, json, argparse
import numpy as np
import pandas as pd


from ej2.utils import make_kfold_indices, ensure_dir, load_config, load_data, train_once, normalize_X

# Barridos por defecto
DEF_BETAS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
DEF_LRS   = [1e-4, 1e-3, 1e-2, 1e-1, 1]      # tanh/sigmoid
DEF_LRS_LINEAR = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # linear

def parse_list(s, cast=float):
    if not s:
        return None
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def load_split(dataset, target, kfolds, test_fold):
    """Carga dataset y devuelve Xtr, ytr, Xte, yte con el fold indicado (1-indexed)."""
    y, X = load_data(dataset, target)
    X = normalize_X(X)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    folds = make_kfold_indices(X.shape[0], int(kfolds))
    tr_idx, te_idx = folds[int(test_fold) - 1]
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], y_min, y_max


def aggregate_over_reps(Xtr, ytr, epochs, lr, activation, beta, reps, y_min=None, y_max=None):
    finals, hists = [], []
    for r in range(1, reps + 1):
        model = train_once(Xtr, ytr, epochs, lr, activation, beta, y_min=y_min, y_max=y_max)
        hist = [float(v) for v in list(model.errors_history_real)]
        mse_final = float(hist[-1]) if hist else float("inf")
        finals.append(mse_final)
        hists.append(hist)
    finals = np.asarray(finals, dtype=float)
    mean = float(np.mean(finals))
    std  = float(np.std(finals, ddof=1)) if finals.size > 1 else 0.0
    # el "mejor" hist (menor final)
    best_idx = int(np.argmin(finals))
    best_hist = hists[best_idx] if hists else []
    return mean, std, best_hist

def phase_beta_sweep(config, results_dir, betas, reps):
    """Barre beta para tanh y sigmoid (LR fijo del JSON). Devuelve mejor beta por activación (por TRAIN)."""
    trials = []
    best_beta = {}

    # pre-split (fix fold)
    Xtr, ytr, _, _, y_min, y_max = load_split(config["dataset"], config["target"],
                                config["kfolds"], config["test_fold"])

    for act in ["tanh"]:
        best_score = float("inf")
        best_b = None
        for beta in betas:
            mean_tr, std_tr, _ = aggregate_over_reps(
                Xtr, ytr,
                epochs=config["epochs"],
                lr=config["lr"],
                activation=act,
                beta=beta,
                reps=reps,
                y_min=y_min, y_max=y_max
            )
            row = {
                "phase": "beta",
                "activation": act,
                "beta": float(beta),
                "learning_rate": float(config["lr"]),
                "epochs": int(config["epochs"]),
                "kfolds": int(config["kfolds"]),
                "test_fold": int(config["test_fold"]),
                "repetitions": int(reps),
                "mse_train_mean": mean_tr,
                "mse_train_std":  std_tr
            }
            trials.append(row)
            if mean_tr < best_score:
                best_score = mean_tr
                best_b = beta
        best_beta[act] = best_b
        print(f"➡️ Mejor β para {act}: {best_b} (MSE_train promedio={best_score:.6f})")

    # guardar ALL
    all_df = pd.DataFrame(trials)
    ensure_dir(results_dir)
    all_path = os.path.join(results_dir, "all_trials.csv")
    all_df.to_csv(all_path, index=False)

    return best_beta, trials

def phase_lr_sweep(config, results_dir, lrs, reps, best_beta):
    """Barre LR para tanh/sigmoid con β* y para linear por su lista de LR."""
    trials = []

    Xtr, ytr, _, _, y_min, y_max = load_split(config["dataset"], config["target"],
                                config["kfolds"], config["test_fold"])

    # tanh/sigmoid con β*
    for act in ["tanh"]:
        bstar = best_beta.get(act, None)
        if bstar is None:
            continue
        for lr in lrs:
            mean_tr, std_tr, _ = aggregate_over_reps(
                Xtr, ytr,
                epochs=config["epochs"],
                lr=lr,
                activation=act,
                beta=bstar,
                reps=reps,
                y_min=y_min, y_max=y_max
            )
            trials.append({
                "phase": "lr",
                "activation": act,
                "beta": float(bstar),
                "learning_rate": float(lr),
                "epochs": int(config["epochs"]),
                "kfolds": int(config["kfolds"]),
                "test_fold": int(config["test_fold"]),
                "repetitions": int(reps),
                "mse_train_mean": mean_tr,
                "mse_train_std":  std_tr
            })

    # linear: solo LR
    for lr in DEF_LRS_LINEAR:
        mean_tr, std_tr, _ = aggregate_over_reps(
            Xtr, ytr,
            epochs=config["epochs"],
            lr=lr,
            activation="linear",
            beta=config["beta"],
            reps=reps,
            y_min=y_min, y_max=y_max
        )
        trials.append({
            "phase": "lr",
            "activation": "linear",
            "beta": float(config["beta"]),  # ignorado
            "learning_rate": float(lr),
            "epochs": int(config["epochs"]),
            "kfolds": int(config["kfolds"]),
            "test_fold": int(config["test_fold"]),
            "repetitions": int(reps),
            "mse_train_mean": mean_tr,
            "mse_train_std":  std_tr
        })

    # anexar a ALL
    all_path = os.path.join(results_dir, "all_trials.csv")
    if os.path.exists(all_path):
        prev = pd.read_csv(all_path)
        all_df = pd.concat([prev, pd.DataFrame(trials)], ignore_index=True)
    else:
        all_df = pd.DataFrame(trials)
    all_df.to_csv(all_path, index=False)

    return trials

def compute_bests(all_trials, config, reps):
    """Selecciona el mejor por activación (por MSE_train) y guarda bests.csv con historia del mejor rep."""
    df = pd.DataFrame(all_trials)
    best_rows = []
    Xtr, ytr, _, _, y_min, y_max = load_split(config["dataset"], config["target"],
                                config["kfolds"], config["test_fold"])

    for act in ["tanh", "linear"]:
        sub = df[df["activation"] == act]
        if sub.empty: 
            continue
        # criterio: mínimo mse_train_mean
        idx = sub["mse_train_mean"].idxmin()
        best = sub.loc[idx].to_dict()

        # re-entrenar 'reps' repeticiones con esa config y quedarnos con la historia del mejor final
        lr = best["learning_rate"]
        beta = best.get("beta", 1.0)
        finals, hists = [], []
        for r in range(1, reps+1):
            model = train_once(Xtr, ytr, config["epochs"], lr, act, beta, y_min=y_min, y_max=y_max)
            hist = [float(v) for v in list(model.errors_history_real)]
            mse_final = float(hist[-1]) if hist else float("inf")
            finals.append(mse_final); hists.append(hist)
        best_i = int(np.argmin(finals))
        best_hist = hists[best_i] if hists else []

        best["history_train_json"] = json.dumps(best_hist)
        best_rows.append(best)

    bests_df = pd.DataFrame(best_rows)
    return bests_df

def main():
    ap = argparse.ArgumentParser(description="Comparar activaciones usando config.json")
    ap.add_argument("--config", required=True, help="Ruta al archivo config.json")
    ap.add_argument("--betas", help=f"Lista de betas (default {DEF_BETAS})")
    ap.add_argument("--lrs", help=f"Lista de learning_rates para tanh/sigmoid (default {DEF_LRS})")
    ap.add_argument("--results_dir", default="ej2/results/compare", help="Dónde escribir resultados")
    args = ap.parse_args()

    # Cargar configuración
    config = load_config(args.config)
    
    # Agregar parámetros específicos para compare_activations
    config["kfolds"] = config.get("kfolds", 5)  # Para el split fijo
    config["test_fold"] = config.get("test_fold", 1)  # Fold fijo para comparación
    
    betas = parse_list(args.betas) or DEF_BETAS
    lrs   = parse_list(args.lrs)   or DEF_LRS
    reps  = int(config["reps"])

    ensure_dir(args.results_dir)

    # fase 1
    best_beta, trials_beta = phase_beta_sweep(config, args.results_dir, betas, reps)

    # fase 2
    trials_lr = phase_lr_sweep(config, args.results_dir, lrs, reps, best_beta)

    # all_trials ya guardado; armamos en memoria para bests
    all_trials = trials_beta + trials_lr
    bests_df = compute_bests(all_trials, config, reps)
    bests_path = os.path.join(args.results_dir, "bests.csv")
    bests_df.to_csv(bests_path, index=False)

    print("\n✅ Listo.")
    print(f"- ALL:   {os.path.join(args.results_dir, 'all_trials.csv')}")
    print(f"- BESTS: {bests_path}")
    for act in ["tanh"]:
        print(f"  β* {act}: {best_beta.get(act)}")

if __name__ == "__main__":
    main()