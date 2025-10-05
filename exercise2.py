import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from perceptrons.simple.perceptron import SimplePerceptron
from perceptrons.nonlinear.perceptron import NonLinearPerceptron

def plot_learning_curves(models_history, title):
    plt.figure(figsize=(12, 6))
    for name, history in models_history.items():
        plt.plot(history, label=name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{title.replace(" ", "_")}.png')

def plot_metrics_comparison(results):
    metrics = ['MSE', 'MAE', 'R2']
    models = list(results.keys())
    
    # Normalize values to 0-1 range for MSE and MAE (lower is better)
    mse_values = [results[m]['mse_mean'] for m in models]
    mae_values = [results[m]['mae_mean'] for m in models]
    r2_values = [results[m]['r2_mean'] for m in models]  # R2 is already 0-1
    
    # Convert MSE and MAE to "goodness" (1 - normalized_value)
    mse_norm = 1 - (mse_values - np.min(mse_values)) / (np.max(mse_values) - np.min(mse_values))
    mae_norm = 1 - (mae_values - np.min(mae_values)) / (np.max(mae_values) - np.min(mae_values))
    
    values = {
        'MSE': mse_norm,
        'MAE': mae_norm,
        'R2': r2_values
    }
    
    plt.figure(figsize=(15, 6))
    plt.title('Performance Comparison: Multiple Non-Linear Perceptrons (Normalized)')
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        plt.bar(x + i*width, [values[m][i] for m in metrics], width, label=model)
    
    plt.xticks(x + width*len(models)/2, metrics)
    plt.ylim(0, 1)  # Force y-axis to be 0-1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png')

def plot_predictions_vs_true(X, y, model, scaler_y, title):
    y_pred = model.predict(X)
    y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
    for i, (ax, x_label) in enumerate(zip(axes, ['x1', 'x2', 'x3'])):
        ax.scatter(X[:, i], y_true, c='blue', label='True Values')
        ax.scatter(X[:, i], y_pred, c='red', label='Perceptron Prediction')
        ax.set_xlabel(x_label)
        ax.set_ylabel('y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{title.replace(" ", "_")}.png')

def main():
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    plt.style.use('bmh')

    # Load and prepare data
    data = pd.read_csv('TP3-ej2-conjunto.csv')
    X = data[['x1', 'x2', 'x3']].values
    y = data['y'].values

    # Normalize data
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Define model configurations
    configurations = {
        'Linear': {'betas': [1]},
        'Sigmoid': {'betas': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        'Tanh': {'betas': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Training loop
    for act_name, config in configurations.items():
        for beta in config['betas']:
            model_name = f"{act_name}-Beta-{beta}"
            model_results = []
            histories = []
            maes = []
            r2s = []

            for train_idx, test_idx in kf.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                # Create appropriate perceptron based on activation type
                if act_name == 'Linear':
                    model = SimplePerceptron(input_size=X.shape[1], learning_rate=0.001)
                else:
                    model = NonLinearPerceptron(
                        input_size=X.shape[1],
                        learning_rate=0.001,
                        activation=act_name.lower(),
                        beta=beta
                    )

                # Train the model
                model.train(X_train, y_train, epochs=1000, verbose=False)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                test_mse = np.mean((y_test - test_pred)**2)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                model_results.append(test_mse)
                maes.append(test_mae)
                r2s.append(test_r2)
                histories.append(model.errors_history)

            results[model_name] = {
                'mse_mean': np.mean(model_results),
                'mse_std': np.std(model_results),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
                'r2_mean': np.mean(r2s),
                'r2_std': np.std(r2s),
                'history': np.mean(histories, axis=0)
            }

            # Plot predictions vs true values for best model
            if act_name == 'Tanh' and beta == 2.0:
                plot_predictions_vs_true(
                    X_scaled, y_scaled, model, scaler_y,
                    f'Predictions vs True Values - {model_name}'
                )

    # Generate plots and save results
    plot_metrics_comparison(results)
    
    for act_name in configurations.keys():
        histories = {k: v['history'] for k, v in results.items() if act_name in k}
        plot_learning_curves(histories, f'Learning Curves - {act_name}')

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE Mean': [v['mse_mean'] for v in results.values()],
        'MSE Std': [v['mse_std'] for v in results.values()],
        'MAE Mean': [v['mae_mean'] for v in results.values()],
        'R2 Mean': [v['r2_mean'] for v in results.values()]
    })
    results_df.to_csv('results/performance_metrics.csv', index=False)

    plt.show()

if __name__ == "__main__":
    main()