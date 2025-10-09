# SIA-TP3

## Requisitos
- Python 3.8+  
- Librerías necesarias:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `tensorflow` 
  - `pillow` 

Instalación rápida:  
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

---

## Ejercicio 1

### Descripción
En este ejercicio se implementa un **perceptrón simple con activación escalón** para resolver los problemas lógicos clásicos **AND** y **XOR**.  
Además, se incluye una herramienta de visualización que muestra la evolución de la **frontera de decisión** a lo largo de las épocas de entrenamiento.

### Ejecución

Para correr la simulación del Ejercicio 1, asegúrate de estar en la carpeta raíz del proyecto y ejecuta el siguiente comando en tu terminal:

Desde la carpeta raíz del proyecto:

```bash
python -m ej1.main
```

Esto ejecutará ambos experimentos:
- **AND** → problema linealmente separable (convergencia exitosa)  
- **XOR** → problema no linealmente separable (el perceptrón no converge)

Cada experimento imprime el progreso del entrenamiento y el estado final de los pesos.

Para observar cómo cambia la frontera del perceptrón en cada época, correr:

```bash
python -m ej1.visualize_epocs
```

Esto generará y mostrará las gráficas:
- `all_lines_AND.png` → evolución de las fronteras para el problema AND  
- `all_lines_XOR.png` → evolución de las fronteras para el problema XOR  

Cada línea representa la frontera de decisión en una época distinta, con la **última resaltada en negro**.

## Ejercicio 2

### Descripción

En este ejercicio se implementa y evalúa un **Perceptrón Simple** con distintas funciones de activación:
- **Lineal**
- **Tanh**
- **Sigmoide**

El objetivo es estudiar cómo los **hiperparámetros β (beta)** y la **tasa de aprendizaje (learning rate)** afectan el desempeño,  
y luego analizar la **capacidad de generalización** mediante validación cruzada (*k-fold*).


### Ejecución
Desde la carpeta raíz del proyecto, correr:

#### Comparación de activaciones y búsqueda de hiperparámetros
Ejecuta el barrido de **β** y **learning rate** para las activaciones `tanh`, `sigmoid` y `linear`:

```bash
python -m ej2.compare_activations -c ej2/base.json
```

Esto genera en `ej2/results/compare/`:
- `all_trials.csv` → todos los runs
- `bests.csv` → mejor combinación por activación

#### Visualización de comparaciones
Graficá los resultados anteriores (MSE vs β, MSE vs LR, comparación entre activaciones):

```bash
python -m ej2.plot_comparisons -c ej2/base.json --results_dir ej2/results/compare --out_dir ej2/results/plots
```

Salidas:
- `*_mse_train_vs_beta.png` → evolución del MSE según β  
- `*_mse_train_vs_lr.png` → evolución del MSE según LR  
- `comparative_best_models_bar.png` → comparación de mejores modelos  
- `all_best_train_histories.png` → curvas de entrenamiento de los mejores modelos

#### Estudio de generalización (barrido de K y folds)
Evalúa cómo cambia la generalización con distintos K y folds:

```bash
python -m ej2.generalization_study -c ej2/generalization_config.json --ks 3,4,5,6,8,10 --reps 5 --results_dir ej2/results/generalization
```

Guarda:
- `k_sweep.csv` → resultados promediados por K  
- `fold_sweep_kX.csv` → resultados por fold de test  
- `generalization_summary.json` → resumen con `k_best` y `fold_best`

#### Visualización del estudio de generalización
Genera gráficos resumen del barrido de K y de folds:

```bash
python -m ej2.plot_generalize --summary ej2/results/generalization/generalization_summary.json
```

Salidas:
- `generalization_k_sweep_<act>.png` → MSE vs K (train/test)
- `generalization_fold_sweep_<act>_k<best>.png` → MSE por fold + dispersión train/test


#### Análisis de datos por fold
Muestra cómo se distribuyen las features y el target en cada fold:

```bash
python -m ej2.plot_folds --summary ej2/results/generalization/generalization_summary.json
```

Genera en `ej2/results/plots/generalization/`:
- `all_folds_strip_features_target_kX.png`  
- `all_folds_boxplot_features_target_kX.png`


## Ejercicio 3

### Descripción
En este ejercicio se implementa un **Perceptrón Multicapa (MLP)** desde cero, sin librerías de deep learning, para resolver distintos problemas de clasificación:

1. **Función lógica XOR** con un MLP (`xor_problem_mlp.py`).  
2. **Discriminación de paridad**: determinar si un dígito (0–9) es par o impar (`parity_problem.py`).  
3. **Clasificación de dígitos (0–9)**: identificar el dígito correcto y evaluar la red con ruido en los datos (`digit_classification_problem.py`).  
4. **Análisis de hiperparámetros**: comparación de optimizadores (SGD, Momentum, Adam) y tasas de aprendizaje (`analysis_runner.py`).  

Los datos se cargan desde `TP3-ej3-digitos.txt` usando utilidades en `data_utils.py`.


### Ejecución

#### 1. Función XOR con MLP
```bash
python -m ej3.xor_problem_mlp
```

#### 2. Discriminación de Paridad
```bash
python -m ej3.parity_problem
```

#### 3. Clasificación de Dígitos (0–9)
```bash
python -m ej3.digit_classification_problem
```

#### 4. Análisis de Hiperparámetros
```bash
python -m ej3.analysis_runner
```

Este script genera gráficos comparativos:
- `optimizer_analysis.png` → rendimiento de distintos optimizadores.  
- `learning_rate_analysis.png` → efecto de la tasa de aprendizaje con Adam.  

## Ejercicio 4

### Descripción
En este ejercicio se usa el **Perceptrón Multicapa (MLP)** para clasificar dígitos del **dataset MNIST**.


### Ejecución (desde la raíz del repo)

```bash
python -m ej4.main
```
### Luego se puede correr una pizarra interactiva para probar al perceptron

```bash
python -m ej4.interactive_predictor
```