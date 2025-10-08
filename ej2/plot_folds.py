# ej2/plot_all_folds_strip_analysis.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_all_folds_data(summary):
    """Obtiene los datos de train y test para TODOS los folds"""
    dataset = summary['dataset']
    target = summary.get('target', 'y')
    kfolds = summary['k_best']
    
    df = pd.read_csv(dataset)
    feature_names = [col for col in df.columns if col != target]
    X_raw = df[feature_names].values.astype(float)
    y_raw = df[target].values.astype(float)
    
    kf = KFold(n_splits=int(kfolds), shuffle=False)
    
    fold_data = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw), 1):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]
        
        fold_data[fold_idx] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_names': feature_names
        }
    
    return fold_data

def create_strip_plot_data(fold_data, fold_num, feature_names):
    """Prepara los datos para strip plot"""
    X_train = fold_data[fold_num]['X_train']
    X_test = fold_data[fold_num]['X_test']
    y_train = fold_data[fold_num]['y_train']
    y_test = fold_data[fold_num]['y_test']
    
    # Preparar datos para features
    feature_data = []
    for i, feature_name in enumerate(feature_names):
        # Datos de train
        for val in X_train[:, i]:
            feature_data.append({
                'fold': fold_num,
                'feature': feature_name,
                'value': val,
                'set': 'Train',
                'type': 'Feature'
            })
        # Datos de test
        for val in X_test[:, i]:
            feature_data.append({
                'fold': fold_num,
                'feature': feature_name,
                'value': val,
                'set': 'Test',
                'type': 'Feature'
            })
    
    # Preparar datos para target
    target_data = []
    for val in y_train:
        target_data.append({
            'fold': fold_num,
            'feature': 'Target (y)',
            'value': val,
            'set': 'Train',
            'type': 'Target'
        })
    for val in y_test:
        target_data.append({
            'fold': fold_num,
            'feature': 'Target (y)',
            'value': val,
            'set': 'Test',
            'type': 'Target'
        })
    
    return pd.DataFrame(feature_data), pd.DataFrame(target_data)

def plot_all_folds_features_strip(fold_data, fold_sweep_csv, out_dir, activation, k_best):
    """Crea strip plots de features para TODOS los folds"""
    
    # Cargar performance
    df_perf = pd.read_csv(fold_sweep_csv)
    performance_by_fold = {}
    for fold_num in fold_data.keys():
        perf_data = df_perf[df_perf['test_fold'] == fold_num]
        if not perf_data.empty:
            performance_by_fold[fold_num] = perf_data['mse_test_mean'].iloc[0]
    
    feature_names = list(fold_data.values())[0]['feature_names']
    all_folds = sorted(fold_data.keys())
    
    # Preparar datos para todos los folds
    all_features_dfs = []
    all_target_dfs = []
    
    for fold_num in all_folds:
        features_df, target_df = create_strip_plot_data(fold_data, fold_num, feature_names)
        all_features_dfs.append(features_df)
        all_target_dfs.append(target_df)
    
    all_features_df = pd.concat(all_features_dfs)
    all_target_df = pd.concat(all_target_dfs)
    
    # Crear figura grande para todos los folds
    n_folds = len(all_folds)
    fig, axes = plt.subplots(n_folds, 2, figsize=(20, 4 * n_folds))
    
    # Si solo hay un fold, convertir axes en 2D
    if n_folds == 1:
        axes = axes.reshape(1, -1)
    
    for row, fold_num in enumerate(all_folds):
        fold_perf = performance_by_fold.get(fold_num, np.nan)
        
        # Features - Fold actual
        fold_features = all_features_df[all_features_df['fold'] == fold_num]
        
        ax1 = axes[row, 0]
        for i, feature in enumerate(feature_names):
            feature_train = fold_features[(fold_features['feature'] == feature) & (fold_features['set'] == 'Train')]
            feature_test = fold_features[(fold_features['feature'] == feature) & (fold_features['set'] == 'Test')]
            
            # Añadir jitter en el eje X
            jitter_train = i + np.random.normal(0, 0.05, len(feature_train))
            jitter_test = i + np.random.normal(0, 0.05, len(feature_test))
            
            ax1.scatter(jitter_train, feature_train['value'], alpha=0.6, s=20, 
                       color='blue', label='Train' if i == 0 and row == 0 else "")
            ax1.scatter(jitter_test, feature_test['value'], alpha=0.8, s=30, 
                       color='red', label='Test' if i == 0 and row == 0 else "", marker='s', edgecolors='black')
        
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Valores')
        ax1.set_title(f'Fold {fold_num} - MSE: {fold_perf:.4f}\nDistribución de Features (Strip Plot)')
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45)
        if row == 0:
            ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Target - Fold actual
        fold_target = all_target_df[all_target_df['fold'] == fold_num]
        
        ax2 = axes[row, 1]
        target_train = fold_target[fold_target['set'] == 'Train']
        target_test = fold_target[fold_target['set'] == 'Test']
        
        jitter_train = np.random.normal(-0.2, 0.05, len(target_train))
        jitter_test = np.random.normal(0.2, 0.05, len(target_test))
        
        ax2.scatter(jitter_train, target_train['value'], alpha=0.6, s=30, 
                   color='green', label='Train' if row == 0 else "")
        ax2.scatter(jitter_test, target_test['value'], alpha=0.8, s=40, 
                   color='orange', label='Test' if row == 0 else "", marker='s', edgecolors='black')
        
        ax2.set_xlabel('Target (y)')
        ax2.set_ylabel('Valores')
        ax2.set_title(f'Fold {fold_num} - MSE: {fold_perf:.4f}\nDistribución del Target (Strip Plot)')
        ax2.set_xticks([-0.2, 0.2])
        ax2.set_xticklabels(['Train', 'Test'])
        if row == 0:
            ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    filename = f"all_folds_strip_features_target_k{k_best}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

def plot_all_folds_features_boxplot(fold_data, fold_sweep_csv, out_dir, activation, k_best):
    """Crea boxplots de features para TODOS los folds"""
    
    # Cargar performance
    df_perf = pd.read_csv(fold_sweep_csv)
    performance_by_fold = {}
    for fold_num in fold_data.keys():
        perf_data = df_perf[df_perf['test_fold'] == fold_num]
        if not perf_data.empty:
            performance_by_fold[fold_num] = perf_data['mse_test_mean'].iloc[0]
    
    feature_names = list(fold_data.values())[0]['feature_names']
    all_folds = sorted(fold_data.keys())
    
    # Crear figura grande para todos los folds
    n_folds = len(all_folds)
    fig, axes = plt.subplots(n_folds, 2, figsize=(20, 4 * n_folds))
    
    # Si solo hay un fold, convertir axes en 2D
    if n_folds == 1:
        axes = axes.reshape(1, -1)
    
    for row, fold_num in enumerate(all_folds):
        fold_perf = performance_by_fold.get(fold_num, np.nan)
        
        # Features - Fold actual
        ax1 = axes[row, 0]
        
        # Preparar datos para boxplot de features
        train_data_features = []
        test_data_features = []
        
        for i, feature_name in enumerate(feature_names):
            train_data_features.append(fold_data[fold_num]['X_train'][:, i])
            test_data_features.append(fold_data[fold_num]['X_test'][:, i])
        
        # Crear boxplots para train y test
        positions_train = np.arange(len(feature_names)) - 0.2
        positions_test = np.arange(len(feature_names)) + 0.2
        
        box1 = ax1.boxplot(train_data_features, positions=positions_train, widths=0.3, 
                          patch_artist=True, labels=feature_names)
        box2 = ax1.boxplot(test_data_features, positions=positions_test, widths=0.3, 
                          patch_artist=True, labels=feature_names)
        
        # Colorear los boxplots
        for box in box1['boxes']:
            box.set_facecolor('lightblue')
            box.set_alpha(0.7)
        for box in box2['boxes']:
            box.set_facecolor('lightcoral')
            box.set_alpha(0.7)
        
        # Añadir líneas de mediana
        for median in box1['medians']:
            median.set_color('blue')
            median.set_linewidth(2)
        for median in box2['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Valores')
        ax1.set_title(f'Fold {fold_num} - MSE: {fold_perf:.4f}\nDistribución de Features (Boxplot)')
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45)
        
        # Añadir leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Train'),
            Patch(facecolor='lightcoral', alpha=0.7, label='Test')
        ]
        if row == 0:
            ax1.legend(handles=legend_elements)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Target - Fold actual
        ax2 = axes[row, 1]
        
        # Preparar datos para boxplot del target
        train_target = fold_data[fold_num]['y_train']
        test_target = fold_data[fold_num]['y_test']
        
        target_data = [train_target, test_target]
        positions_target = [-0.2, 0.2]
        
        box_target = ax2.boxplot(target_data, positions=positions_target, widths=0.3, 
                                patch_artist=True, labels=['Train', 'Test'])
        
        # Colorear los boxplots del target
        colors_target = ['lightgreen', 'orange']
        for box, color in zip(box_target['boxes'], colors_target):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Añadir líneas de mediana
        for median in box_target['medians']:
            median.set_color('darkgreen' if median == box_target['medians'][0] else 'darkorange')
            median.set_linewidth(2)
        
        ax2.set_xlabel('Target (y)')
        ax2.set_ylabel('Valores')
        ax2.set_title(f'Fold {fold_num} - MSE: {fold_perf:.4f}\nDistribución del Target (Boxplot)')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    filename = f"all_folds_boxplot_features_target_k{k_best}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

def main():
    ap = argparse.ArgumentParser(description="Strip plots y boxplots para TODOS los folds")
    ap.add_argument("--summary", required=True, help="Archivo JSON de resumen de generalización")
    ap.add_argument("--out_dir", default="ej2/results/plots/generalization", help="Directorio de salida")
    args = ap.parse_args()
    
    ensure_dir(args.out_dir)
    
    # Cargar resumen de generalización
    with open(args.summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    activation = summary['activation']
    k_best = summary['k_best']
    fold_sweep_csv = summary.get('fold_sweep_csv')
    
    if not fold_sweep_csv or not os.path.exists(fold_sweep_csv):
        print(f"[ERROR] No se encuentra fold_sweep_csv: {fold_sweep_csv}")
        return
    
    # Obtener datos de TODOS los folds
    fold_data = get_all_folds_data(summary)
    
    print(f"📊 Analizando TODOS los {len(fold_data)} folds para K={k_best}")
    
    # Generar ambos gráficos
    print("📈 Generando strip plot completo para todos los folds...")
    plot_all_folds_features_strip(fold_data, fold_sweep_csv, args.out_dir, activation, k_best)
    
    print("📊 Generando boxplot completo para todos los folds...")
    plot_all_folds_features_boxplot(fold_data, fold_sweep_csv, args.out_dir, activation, k_best)
    
    print(f"\n✅ Ambos gráficos guardados en: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()