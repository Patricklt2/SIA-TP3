import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptrons.simple.perceptron import SimplePerceptron
from ej2.utils import evaluate_real, make_kfold_indices, load_config, load_data, train_once

def collect_all_folds_curves(dataset, target, k, reps, epochs, lr, activation, beta):
    """Entrena para todos los folds y guarda curvas de todos ellos"""
    y, X = load_data(dataset, target)
    folds = make_kfold_indices(X.shape[0], k)
    
    all_folds_data = {}
    
    for fold_idx, (tr_idx, te_idx) in enumerate(folds, start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        fold_best_te_final = float("inf")
        fold_best_train_curve = None
        fold_best_test_curve = None
        fold_best_test_true = None
        fold_best_test_pred = None

        for r in range(1, reps + 1):
            model = train_once(Xtr, ytr, epochs, lr, activation, beta)

            train_curve = list(map(float, model.errors_history_real))
            test_curve = list(map(float, model.get_testmse_history(Xte, yte)))

            te_final = test_curve[-1] if len(test_curve) else float("inf")
            if te_final < fold_best_te_final:
                fold_best_te_final = te_final
                fold_best_train_curve = train_curve
                fold_best_test_curve = test_curve
                yhat_te_final = model.predict(Xte)
                fold_best_test_true = yte.astype(float).ravel().tolist()
                fold_best_test_pred = np.asarray(yhat_te_final, dtype=float).ravel().tolist()

        all_folds_data[f"fold_{fold_idx}"] = {
            "train_mse_per_epoch": fold_best_train_curve or [],
            "test_mse_per_epoch": fold_best_test_curve or [],
            "test_scatter": {"y_true": fold_best_test_true or [], "y_pred": fold_best_test_pred or []},
            "final_test_mse": fold_best_te_final
        }
    
    return all_folds_data

def run_study(
    dataset, target, k_list, reps, epochs, lr, activation, beta, verbose=False
):
    data = pd.read_csv(dataset)
    y = data[target].values.astype(float)
    X = data[[c for c in data.columns if c != target]].values.astype(float)
    n = X.shape[0]

    study = {
        "meta": {
            "dataset": dataset,
            "target": target,
            "reps": reps,
            "epochs": epochs,
            "lr": lr,
            "activation": activation,
            "beta": beta
        },
        "per_k": {}
    }

    best_k = None
    best_k_mean_test = float("inf")

    for k in k_list:
        folds = make_kfold_indices(n, k)

        folds_summary = []
        k_train_means = []
        k_test_means = []

        best_fold_idx = None
        best_fold_test_mean = float("inf")

        for fold_idx, (tr_idx, te_idx) in enumerate(folds, start=1):
            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]

            rep_train = []
            rep_test = []

            for r in range(1, reps + 1):

                model = train_once(X_train, y_train, epochs, lr, activation, beta)

                mse_tr = float(model.errors_history_real[-1])
                y_hat_te = model.predict(X_test)
                mse_te = evaluate_real(y_hat_te, y_test)

                rep_train.append(mse_tr)
                rep_test.append(mse_te)

            fold_train_mean = float(np.mean(rep_train))
            fold_test_mean = float(np.mean(rep_test))

            fold_summary = {
                "fold": fold_idx,
                "mse_train_mean": fold_train_mean,
                "mse_test_mean": fold_test_mean,
                "mse_train_std": float(np.std(rep_train)),
                "mse_test_std": float(np.std(rep_test)),
            }
            folds_summary.append(fold_summary)

            k_train_means.append(fold_train_mean)
            k_test_means.append(fold_test_mean)

            if fold_test_mean < best_fold_test_mean:
                best_fold_test_mean = fold_test_mean
                best_fold_idx = fold_idx

        k_summary = {
            "k": k,
            "folds": folds_summary,
            "mean_train_over_folds": float(np.mean(k_train_means)),
            "mean_test_over_folds": float(np.mean(k_test_means)),
            "std_train_over_folds": float(np.std(k_train_means)),
            "std_test_over_folds": float(np.std(k_test_means)),
            "best_fold_for_k": int(best_fold_idx),
        }

        study["per_k"][str(k)] = k_summary

        if k_summary["mean_test_over_folds"] < best_k_mean_test:
            best_k_mean_test = k_summary["mean_test_over_folds"]
            best_k = k

    study["best_k"] = int(best_k)
    study["best_k_mean_test"] = float(best_k_mean_test)
    study["best_fold_in_best_k"] = int(study["per_k"][str(best_k)]["best_fold_for_k"])

    return study

def parse_k_list(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted({int(p) for p in parts})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Estudio de K-Folds y selección de fold para test (Ej2) usando config.json.")
    ap.add_argument("--config", required=True, help="Ruta al archivo config.json")
    args = ap.parse_args()

    # Cargar configuración
    config = load_config(args.config)
    
    # Extraer parámetros de la configuración
    target = config["target"]
    k_list = parse_k_list(config["klist"])
    reps = config["reps"]
    epochs = config["epochs"]
    lr = config["lr"]
    activation = config["activation"]
    beta = config["beta"]
    out = config["out"]
    save_all_folds_curves = config["save_all_folds_curves"]
    all_folds_curves_out = config["all_folds_curves_out"]
    dataset = config["dataset"]

    study = run_study(
        dataset=dataset,
        target=target,
        k_list=k_list,
        reps=reps,
        epochs=epochs,
        lr=lr,
        activation=activation,
        beta=beta,
        verbose=False
    )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(study, f, indent=2)
    print(f"✅ Guardado estudio en {out}")
    print(f"Mejor k: {study['best_k']}  | Mejor fold dentro de k: {study['best_fold_in_best_k']}")

    # guardar curvas de todos los folds si se pidió
    if save_all_folds_curves:
        best_k = int(study["best_k"])
        all_folds_curves = collect_all_folds_curves(
            dataset=dataset,
            target=target,
            k=best_k,
            reps=reps,
            epochs=epochs,
            lr=lr,
            activation=activation,
            beta=beta
        )
        
        if all_folds_curves_out:
            all_folds_path = all_folds_curves_out
        else:
            base_dir = os.path.dirname(out) or "."
            all_folds_path = os.path.join(base_dir, f"curves_all_folds_k{best_k}.json")
        
        with open(all_folds_path, "w") as f:
            json.dump(all_folds_curves, f, indent=2)
        print(f"🟢 Curvas de TODOS los folds guardadas en {all_folds_path}")