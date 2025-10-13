import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptrons.simple.perceptron import SimplePerceptron
from ej2.utils import evaluate_real, make_kfold_indices

def collect_all_folds_curves(dataset, target, k, reps, epochs, lr, activation, beta, seed_base):
    """Entrena para todos los folds y guarda curvas de todos ellos"""
    data = pd.read_csv(dataset)
    y = data[target].values.astype(float)
    X = data[[c for c in data.columns if c != target]].values.astype(float)
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
            np.random.seed(seed_base + fold_idx * 1000 + r)
            model = SimplePerceptron(
                input_size=Xtr.shape[1],
                learning_rate=lr,
                activation=activation,
                beta=beta
            )
            model.train(Xtr, ytr, epochs=epochs, verbose=False)

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

def collect_bestcase_curves(dataset, target, k, fold, reps, epochs, lr, activation, beta, seed_base):
    """Vuelve a entrenar en el mejor K/fold para encontrar la mejor repetición
    (por MSE de test final) y devuelve:
      - train_mse_per_epoch
      - test_mse_per_epoch
      - test_scatter: y_true, y_pred finales
    """
    data = pd.read_csv(dataset)
    y = data[target].values.astype(float)
    X = data[[c for c in data.columns if c != target]].values.astype(float)
    folds = make_kfold_indices(X.shape[0], k)
    tr_idx, te_idx = folds[fold - 1]
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    best_te_final = float("inf")
    best_train_curve = None
    best_test_curve = None
    best_test_true = None
    best_test_pred = None

    for r in range(1, reps + 1):
        np.random.seed(seed_base + r)
        model = SimplePerceptron(
            input_size=Xtr.shape[1],
            learning_rate=lr,
            activation=activation,
            beta=beta
        )
        model.train(Xtr, ytr, epochs=epochs, verbose=False)

        train_curve = list(map(float, model.errors_history_real))
        test_curve = list(map(float, model.get_testmse_history(Xte, yte)))

        te_final = test_curve[-1] if len(test_curve) else float("inf")
        if te_final < best_te_final:
            best_te_final = te_final
            best_train_curve = train_curve
            best_test_curve = test_curve
            yhat_te_final = model.predict(Xte)
            best_test_true = yte.astype(float).ravel().tolist()
            best_test_pred = np.asarray(yhat_te_final, dtype=float).ravel().tolist()

    return {
        "train_mse_per_epoch": best_train_curve or [],
        "test_mse_per_epoch": best_test_curve or [],
        "test_scatter": {"y_true": best_test_true or [], "y_pred": best_test_pred or []}
    }

# ---------------- existing study (no per-epoch) ----------------

def run_study(
    dataset, target, k_list, reps, epochs, lr, activation, beta, seed_base, verbose=False
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
            "beta": beta,
            "seed_base": seed_base,
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
                seed = seed_base + fold_idx * 1000 + r
                np.random.seed(seed)

                model = SimplePerceptron(
                    input_size=X_train.shape[1],
                    learning_rate=lr,
                    activation=activation,
                    beta=beta
                )
                model.train(X_train, y_train, epochs=epochs, verbose=False)

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
    ap = argparse.ArgumentParser(description="Estudio de K-Folds y selección de fold para test (Ej2) + curvas/plots opcionales.")
    ap.add_argument("--dataset", required=True, help="Ruta al CSV de datos")
    ap.add_argument("--target", default="y", help="Nombre de la columna target")
    ap.add_argument("--klist", default="3,4,5,6,8,10", help="Lista separada por comas de cantidades de folds a evaluar")
    ap.add_argument("--reps", type=int, default=5, help="Repeticiones por fold")
    ap.add_argument("--epochs", type=int, default=1200, help="Épocas de entrenamiento")
    ap.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    ap.add_argument("--activation", default="tanh", choices=["linear", "tanh", "sigmoid", "relu"], help="Activación")
    ap.add_argument("--beta", type=float, default=1.0, help="Beta para activaciones")
    ap.add_argument("--seed", type=int, default=1234, help="Seed base")
    ap.add_argument("--out", default="cv_study.json", help="Archivo de salida (.json)")
    # nuevos flags
    ap.add_argument("--save_curves", action="store_true", help="Guarda epoch vs MSE (train/test) y scatter (test) del mejor K/fold.")
    ap.add_argument("--save_all_folds_curves", action="store_true", help="Guarda curvas para TODOS los folds del mejor K.")
    ap.add_argument("--curves_out", default=None, help="Ruta del JSON de curvas (default: mismo dir de --out, nombre curves_best.json)")
    ap.add_argument("--all_folds_curves_out", default=None, help="Ruta del JSON de curvas de todos los folds")
    ap.add_argument("--plots_outdir", default=None, help="Si se setea, guarda learning_curves.png y test_scatter.png aquí.")

    args = ap.parse_args()
    k_list = parse_k_list(args.klist)

    study = run_study(
        dataset=args.dataset,
        target=args.target,
        k_list=k_list,
        reps=args.reps,
        epochs=args.epochs,
        lr=args.lr,
        activation=args.activation,
        beta=args.beta,
        seed_base=args.seed,
        verbose=False
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(study, f, indent=2)
    print(f"✅ Guardado estudio en {args.out}")
    print(f"Mejor k: {study['best_k']}  | Mejor fold dentro de k: {study['best_fold_in_best_k']}")

    # guardar curvas del mejor fold si se pidió
    if args.save_curves or args.plots_outdir:
        best_k = int(study["best_k"])
        best_fold = int(study["best_fold_in_best_k"])
        curves = collect_bestcase_curves(
            dataset=args.dataset,
            target=args.target,
            k=best_k,
            fold=best_fold,
            reps=args.reps,
            epochs=args.epochs,
            lr=args.lr,
            activation=args.activation,
            beta=args.beta,
            seed_base=args.seed
        )
        # curvas json
        if args.save_curves:
            if args.curves_out:
                curves_path = args.curves_out
            else:
                base_dir = os.path.dirname(args.out) or "."
                curves_path = os.path.join(base_dir, "curves_best.json")
            with open(curves_path, "w") as f:
                json.dump(curves, f, indent=2)
            print(f"🟢 Curvas del mejor fold guardadas en {curves_path}")
        else:
            # si no guardamos archivo, crear temporal para plots
            base_dir = os.path.dirname(args.out) or "."
            curves_path = os.path.join(base_dir, "_curves_temp.json")
            with open(curves_path, "w") as f:
                json.dump(curves, f, indent=2)

        # plots
        if args.plots_outdir:
            os.makedirs(args.plots_outdir, exist_ok=True)
            plot_learning_curves(curves_path, os.path.join(args.plots_outdir, "learning_curves_best.png"))
            plot_test_scatter(curves_path, os.path.join(args.plots_outdir, "test_scatter_y_vs_yhat.png"))
            print(f"📈 Plots guardados en {args.plots_outdir}")

    # guardar curvas de todos los folds si se pidió
    if args.save_all_folds_curves:
        best_k = int(study["best_k"])
        all_folds_curves = collect_all_folds_curves(
            dataset=args.dataset,
            target=args.target,
            k=best_k,
            reps=args.reps,
            epochs=args.epochs,
            lr=args.lr,
            activation=args.activation,
            beta=args.beta,
            seed_base=args.seed
        )
        
        if args.all_folds_curves_out:
            all_folds_path = args.all_folds_curves_out
        else:
            base_dir = os.path.dirname(args.out) or "."
            all_folds_path = os.path.join(base_dir, f"curves_all_folds_k{best_k}.json")
        
        with open(all_folds_path, "w") as f:
            json.dump(all_folds_curves, f, indent=2)
        print(f"🟢 Curvas de TODOS los folds guardadas en {all_folds_path}")