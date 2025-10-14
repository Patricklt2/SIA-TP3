import os, json, argparse
import numpy as np
import pandas as pd

from perceptrons.simple.perceptron import SimplePerceptron

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def load_config(config_path):
    """Carga la configuración desde un archivo JSON"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Valores por defecto si no están en el config
    default_config = {
        "target": "y",
        "klist": "3,4,5,6,8,10",
        "reps": 5,
        "epochs": 1200,
        "lr": 0.01,
        "activation": "tanh",
        "beta": 1.0,
        "out": "cv_study.json",
        "save_all_folds_curves": False,
        "all_folds_curves_out": None
    }
    
    # Combinar configuraciones (los valores del archivo tienen prioridad)
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    return config

def evaluate_real(y_pred, y_true):
    """Métricas en escala REAL (y_pred ya viene desescalado por el modelo)."""
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse

def make_kfold_indices(n_samples: int, k: int):
    """
    Genera una lista de pares (train_idx, test_idx) para K-Fold.
    Reparte lo más parejo posible; los primeros (n_samples % k) folds tienen 1 muestra extra.

    Returns: list[tuple[np.ndarray, np.ndarray]]
    """
    if k < 2 or k > n_samples:
        raise ValueError(f"k debe estar en [2, {n_samples}]")

    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1

    indices = np.arange(n_samples)
    folds = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        start = stop
    return folds

def load_data(dataset, target):
    """Carga dataset y devuelve Xtr, ytr, Xte, yte con el fold indicado (1-indexed)."""
    data = pd.read_csv(dataset)
    y = data[target].values.astype(float)
    X = data[[c for c in data.columns if c != target]].values.astype(float)
    return y, X

def train_once(Xtr, ytr, epochs, lr, activation, beta):
    model = SimplePerceptron(
        input_size=Xtr.shape[1],
        learning_rate=float(lr),
        activation=str(activation),
        beta=float(beta) if beta is not None else 1.0
    )
    model.train(Xtr, ytr, epochs=int(epochs), verbose=False)
    # historial de MSE en escala real (train)
    return model

def run(kfolds, test_fold, activation, beta, lr, epochs, reps, verbose, out_csv):
    # --- datos ---
    data = pd.read_csv(cfg["dataset"])
    target = cfg.get("target", "y")

    y_raw = data[target].values.astype(float)
    X_raw = data[[c for c in data.columns if c != target]].values.astype(float)

    n_samples = X_raw.shape[0]
    folds = make_kfold_indices(n_samples, kfolds)

    if not (1 <= test_fold <= kfolds):
        raise ValueError(f"test_fold debe estar entre 1 y {kfolds}")

    tr_idx, te_idx = folds[test_fold - 1]

    # Datos ya normalizados
    X_train = X_raw[tr_idx]
    X_test  = X_raw[te_idx]
    y_train = y_raw[tr_idx]
    y_test  = y_raw[te_idx]

    results = []

    for r in range(1, reps + 1):

        # --- modelo ---
        model = SimplePerceptron(
            input_size=X_train.shape[1],
            learning_rate=lr,
            activation=activation,   # 'tanh' | 'sigmoid' | 'linear'
            beta=beta
        )

        # entrenar (el modelo maneja internamente el escalado de y si aplica)
        model.train(X_train, y_train, epochs=epochs, verbose=verbose)

        # predicciones en ESCALA REAL (el modelo ya desescala si corresponde)
        y_hat_test = model.predict(X_test)

        if(verbose):
            for y_pred, y_real in zip(y_hat_test, y_test):
                print(f"Real: {y_real:.4f}. Predicted: {y_pred:.4f}")

        # métricas en escala real
        mse_tr = model.errors_history_real[-1]
        mse_te = evaluate_real(y_hat_test, y_test)

        results.append({
            "activation": activation,
            "beta": beta,
            "learning_rate": lr,
            "epochs": epochs,
            "repetitions": reps,
            "rep": r,
            "kfolds": kfolds,
            "test_fold": test_fold,
            "mse_train": mse_tr,
            "mse_test": mse_te,
            "bias": float(model.bias),
            **{f"w_{i}": float(wi) for i, wi in enumerate(model.weights)},
            # historiales del modelo
            "mse_history_real_json": json.dumps([float(v) for v in getattr(model, "errors_history_real", [])])
        })

        print(f"Rep {r}/{reps} -> test MSE={mse_te:.6f}")

    # --- salida ---
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Guardado en {out_csv} ({len(df)} filas)")
    print(f"Test fold: {test_fold}/{kfolds}")
    print(f"MSE_train = {df['mse_train'].mean():.6f}")
    print(f"MSE_test = {df['mse_test'].mean():.6f}")
    
    # Mostrar estadísticas de la partición
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"y_test range:  [{y_test.min():.3f}, {y_test.max():.3f}]")
    return df
 

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runner de perceptrón usando SimplePerceptron (K-Fold fijo)")
    ap.add_argument("--config", "-c", required=True, help="Ruta al archivo JSON de configuración")        
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Parámetros del JSON
    kfolds = int(cfg.get("kfolds", 5))
    test_fold = int(cfg.get("test_fold", 1))
    activation = cfg.get("activation", "tanh")
    beta = float(cfg.get("beta", 1.0))
    lr = float(cfg.get("learning_rate", 0.01))
    epochs = int(cfg.get("epochs", 1000))
    reps = int(cfg.get("repetitions", 5))
    verbose = bool(cfg.get("verbose", False))
    out_csv = cfg.get("output_csv", "results.csv")

    run(kfolds, test_fold, activation, beta, lr, epochs, reps, verbose, out_csv)