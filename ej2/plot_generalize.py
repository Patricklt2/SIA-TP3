# ej2/plot_generalization.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_k_sweep(k_sweep_csv, out_dir, activation, beta, lr):
    """Grafica MSE vs K (número de folds)"""
    df = pd.read_csv(k_sweep_csv)
    
    if df.empty:
        print(f"[WARN] No hay datos en {k_sweep_csv}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico 1: MSE Test vs K
    k_values = df["kfolds"].values
    mse_test_means = df["mse_test_mean"].values
    mse_test_stds = df["mse_test_std"].fillna(0).values
    
    # Encontrar el mejor K (menor MSE test)
    best_idx = np.nanargmin(mse_test_means)
    best_k = k_values[best_idx]
    best_mse = mse_test_means[best_idx]
    
    # Gráfico de barras para MSE Test
    bars = ax1.bar(k_values, mse_test_means, yerr=mse_test_stds, capsize=5, 
                   color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
    
    # Resaltar el mejor K
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(0.9)
    
    ax1.set_xlabel('Número de Folds (K)')
    ax1.set_ylabel('MSE Test (promedio ± σ)')
    ax1.set_title(f'Generalización vs K - {activation.upper()}\nβ={beta}, LR={lr}')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores encima de las barras
    for i, (bar, mean, std) in enumerate(zip(bars, mse_test_means, mse_test_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01 * max(mse_test_means),
                f'{mean:.4f}\n±{std:.4f}', 
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Gráfico 2: Comparación Train vs Test
    mse_train_means = df["mse_train_mean"].values
    mse_train_stds = df["mse_train_std"].fillna(0).values
    
    ax2.errorbar(k_values, mse_train_means, yerr=mse_train_stds, 
                 fmt='o-', color='green', label='Train', capsize=5, linewidth=2)
    ax2.errorbar(k_values, mse_test_means, yerr=mse_test_stds, 
                 fmt='s-', color='red', label='Test', capsize=5, linewidth=2)
    
    ax2.set_xlabel('Número de Folds (K)')
    ax2.set_ylabel('MSE')
    ax2.set_title('Comparación MSE Train vs Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Escala logarítmica si el rango es amplio
    all_vals = np.concatenate([mse_train_means, mse_test_means])
    if np.max(all_vals) / np.min(all_vals[all_vals > 0]) > 100:
        ax2.set_yscale('log')
    
    # Añadir anotación del mejor K
    ax1.text(0.02, 0.98, f'Mejor K: {best_k} (MSE = {best_mse:.4f})', 
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Guardar
    filename = f"generalization_k_sweep_{activation}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[saved]", path)

def plot_fold_sweep(fold_sweep_csv, out_dir, activation, beta, lr, k_best):
    """Grafica MSE vs Fold de Test para un K fijo"""
    df = pd.read_csv(fold_sweep_csv)
    
    if df.empty:
        print(f"[WARN] No hay datos en {fold_sweep_csv}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Preparar datos
    folds = df["test_fold"].values
    mse_test_means = df["mse_test_mean"].values
    mse_test_stds = df["mse_test_std"].fillna(0).values
    mse_train_means = df["mse_train_mean"].values
    mse_train_stds = df["mse_train_std"].fillna(0).values
    
    # Encontrar el mejor fold
    best_idx = np.nanargmin(mse_test_means)
    best_fold = folds[best_idx]
    best_mse = mse_test_means[best_idx]
    
    # Gráfico 1: MSE por fold (barras)
    x_pos = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, mse_train_means, width, yerr=mse_train_stds,
                   capsize=4, label='Train', color='lightgreen', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, mse_test_means, width, yerr=mse_test_stds,
                   capsize=4, label='Test', color='lightcoral', alpha=0.8)
    
    # Resaltar el mejor fold
    bars2[best_idx].set_color('red')
    bars2[best_idx].set_alpha(1.0)
    
    ax1.set_xlabel('Fold de Test')
    ax1.set_ylabel('MSE (promedio ± σ)')
    ax1.set_title(f'MSE por Fold de Test (K={k_best})\n{activation.upper()} - β={beta}, LR={lr}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Fold {f}' for f in folds])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras de test
    for i, (bar, mean, std) in enumerate(zip(bars2, mse_test_means, mse_test_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01 * max(mse_test_means),
                f'{mean:.4f}', ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Gráfico 2: Dispersión Train vs Test
    ax2.scatter(mse_train_means, mse_test_means, s=80, alpha=0.7, color='blue')
    
    # Añadir etiquetas de folds
    for i, (tr, te, fold) in enumerate(zip(mse_train_means, mse_test_means, folds)):
        ax2.annotate(f'F{fold}', (tr, te), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    # Línea y=x para referencia
    min_val = min(np.min(mse_train_means), np.min(mse_test_means))
    max_val = max(np.max(mse_train_means), np.max(mse_test_means))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x')
    
    ax2.set_xlabel('MSE Train')
    ax2.set_ylabel('MSE Test')
    ax2.set_title('Dispersión: Train vs Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Escala logarítmica si es necesario
    if max_val / min_val > 100:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    
    # Añadir anotación del mejor fold
    ax1.text(0.02, 0.98, f'Mejor fold: {best_fold}\nMSE = {best_mse:.4f}', 
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Guardar
    filename = f"generalization_fold_sweep_{activation}_k{k_best}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[saved]", path)

def plot_generalization_comparison(summary_files, out_dir):
    """Compara los resultados de generalización entre diferentes configuraciones"""
    all_data = []
    
    for summary_file in summary_files:
        if not os.path.exists(summary_file):
            print(f"[WARN] No se encuentra {summary_file}")
            continue
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Leer el CSV de k_sweep para obtener todos los puntos
        k_sweep_csv = summary.get('k_sweep_csv', '')
        if os.path.exists(k_sweep_csv):
            df_k = pd.read_csv(k_sweep_csv)
            best_k_data = df_k[df_k['kfolds'] == summary['k_best']].iloc[0]
            
            all_data.append({
                'activation': summary['activation'],
                'beta': summary.get('beta', 1.0),
                'learning_rate': summary.get('learning_rate', 0.01),
                'k_best': summary['k_best'],
                'fold_best': summary['fold_best'],
                'mse_test_best': best_k_data['mse_test_mean'],
                'mse_test_std': best_k_data['mse_test_std'],
                'mse_train_best': best_k_data['mse_train_mean'],
                'color': {'linear': '#7C3AED', 'tanh': '#16A34A', 'sigmoid': '#F59E0B'}.get(
                    summary['activation'].lower(), '#666666')
            })
    
    if not all_data:
        print("[WARN] No hay datos para comparación")
        return
    
    # Crear gráfico comparativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    activations = [f"{d['activation'].upper()}\nβ={d['beta']}, LR={d['learning_rate']:.0e}" 
                   for d in all_data]
    mse_test_vals = [d['mse_test_best'] for d in all_data]
    mse_test_errs = [d['mse_test_std'] for d in all_data]
    colors = [d['color'] for d in all_data]
    k_vals = [d['k_best'] for d in all_data]
    
    # Gráfico 1: Barras comparativas de MSE Test
    bars = ax1.bar(activations, mse_test_vals, yerr=mse_test_errs, capsize=8,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('MSE Test (mejor K)')
    ax1.set_title('Comparación de Generalización entre Configuraciones')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores y K óptimo
    for i, (bar, mse, k, act_data) in enumerate(zip(bars, mse_test_vals, k_vals, all_data)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + mse_test_errs[i] + 0.01 * max(mse_test_vals),
                f'{mse:.4f}\nK={k}', ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Gráfico 2: K óptimo por configuración
    ax2.bar(activations, k_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('K Óptimo')
    ax2.set_title('K Óptimo por Configuración')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores de K
    for i, (act, k) in enumerate(zip(activations, k_vals)):
        ax2.text(i, k + 0.1, f'K={k}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    path = os.path.join(out_dir, "generalization_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[saved]", path)

def main():
    ap = argparse.ArgumentParser(description="Visualizar resultados del estudio de generalización")
    ap.add_argument("--summary", required=True, help="Archivo JSON de resumen de generalización o directorio con múltiples summary files")
    ap.add_argument("--out_dir", default="ej2/results/plots/generalization", help="Directorio de salida para gráficos")
    args = ap.parse_args()
    
    ensure_dir(args.out_dir)
    
    summary_files = []
    
    if os.path.isdir(args.summary):
        # Buscar todos los archivos JSON en el directorio
        for file in os.listdir(args.summary):
            if file.endswith('generalization_summary.json'):
                summary_files.append(os.path.join(args.summary, file))
    else:
        # Archivo individual
        summary_files = [args.summary]
    
    # Procesar cada archivo de resumen
    for summary_file in summary_files:
        if not os.path.exists(summary_file):
            print(f"[WARN] No se encuentra {summary_file}")
            continue
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        activation = summary['activation']
        beta = summary.get('beta', 1.0)
        lr = summary.get('learning_rate', 0.01)
        k_best = summary['k_best']
        
        print(f"📊 Procesando: {activation.upper()} (β={beta}, LR={lr})")
        
        # Graficar barrido de K
        k_sweep_csv = summary.get('k_sweep_csv')
        if k_sweep_csv and os.path.exists(k_sweep_csv):
            plot_k_sweep(k_sweep_csv, args.out_dir, activation, beta, lr)
        
        # Graficar barrido de folds
        fold_sweep_csv = summary.get('fold_sweep_csv')
        if fold_sweep_csv and os.path.exists(fold_sweep_csv):
            plot_fold_sweep(fold_sweep_csv, args.out_dir, activation, beta, lr, k_best)
    
    # Gráfico comparativo si hay múltiples configuraciones
    if len(summary_files) > 1:
        plot_generalization_comparison(summary_files, args.out_dir)
    
    print(f"\n✅ Gráficos de generalización guardados en: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()