# ej2/fold_runner_sp.py
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Importa TU perceptrón (perceptron.py)
from perceptrons.simple.perceptron import SimplePerceptron


def evaluate_real(y_pred, y_true):
    """Métricas en escala REAL (y_pred ya viene desescalado por el modelo)."""
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2  = float(r2_score(y_true, y_pred))
    return mse, mae, r2

def run(cfg):
    # --- datos ---
    data = pd.read_csv(cfg["dataset"])
    target = cfg.get("target", "y")

    y_raw = data[target].values.astype(float)
    X_raw = data[[c for c in data.columns if c != target]].values.astype(float)

    # --- hiperparámetros ---
    kfolds      = int(cfg["kfolds"])
    test_fold   = int(cfg["test_fold"])      # 1..k
    activation  = str(cfg["activation"]).lower()
    beta        = float(cfg.get("beta", 1.0))
    lr          = float(cfg["learning_rate"])
    epochs      = int(cfg["epochs"])
    reps        = int(cfg["repetitions"])
    seed_base   = int(cfg.get("seed", 1234))
    shuffle     = bool(cfg.get("shuffle", True))
    out_csv     = cfg.get("output_csv", "results/fold_run_sp.csv")

    # --- folds ---
    if(shuffle):
        kf = KFold(n_splits=kfolds, shuffle=shuffle, random_state=seed_base)
    else:
        kf = KFold(n_splits=kfolds, shuffle=shuffle)
    folds = list(kf.split(X_raw))

    if not (1 <= test_fold <= kfolds):
        raise ValueError(f"test_fold debe estar entre 1 y {kfolds}")

    tr_idx, te_idx = folds[test_fold - 1]
    X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]
    y_train, y_test = y_raw[tr_idx], y_raw[te_idx]

    # --- escalado de X (solo X) ---
    xsc = MinMaxScaler(feature_range=(-1, 1))
    X_train = xsc.fit_transform(X_train_raw)
    X_test  = xsc.transform(X_test_raw)

    results = []

    for r in range(1, reps + 1):
        seed = seed_base + r * 97
        np.random.seed(seed)  # para la init de pesos/bias

        # --- modelo ---
        model = SimplePerceptron(
            input_size=X_train.shape[1],
            learning_rate=lr,
            activation=activation,   # 'tanh' | 'sigmoid' | 'relu' | 'linear'
            beta=beta
        )

        # entrenar (el modelo maneja internamente el escalado de y si aplica)
        model.train(X_train, y_train, epochs=epochs, verbose=False)

        # predicciones en ESCALA REAL (el modelo ya desescala si corresponde)
        y_hat_test  = model.predict(X_test)

        # métricas en escala real
        mse_tr= model.errors_history_real[-1]
        mse_te, mae_te, r2_te = evaluate_real(y_hat_test,  y_test)


        results.append({
            "activation": activation,
            "beta": beta,
            "learning_rate": lr,
            "epochs": epochs,
            "repetitions": reps,
            "rep": r,
            "kfolds": kfolds,
            "shuffle": shuffle,
            "seed_base": seed_base,
            "seed_used": seed,
            "test_fold": test_fold,
            "mse_train": mse_tr,
            "mse_test": mse_te,
            "mae_test": mae_te,
            "r2_test": r2_te,
            "bias": float(model.bias),
            **{f"w_{i}": float(wi) for i, wi in enumerate(model.weights)},
            # historiales del modelo
            "mse_history_real_json":   json.dumps([float(v) for v in getattr(model, "errors_history_real", [])])
        })

        print(f"Rep {r}/{reps} -> test MSE={mse_te:.6f}")

    # --- salida ---
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Guardado en {out_csv} ({len(df)} filas)")
    print(f"Test fold: {test_fold}/{kfolds}")
    print(f"Promedio MSE_test = {df['mse_test'].mean():.6f}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runner de perceptrón usando SimplePerceptron (K-Fold fijo)")
    ap.add_argument("--config", "-c", required=True, help="Ruta al archivo JSON de configuración")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run(cfg)
