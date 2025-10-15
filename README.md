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

#### Ejecución directa de entrenamiento simple
Permite correr una sola instancia del perceptrón con los parámetros definidos en el config.json

```bash
python -m ej2.utils --config ej2/config.json
```

Ejemplo (`ej2/config.json`):

```json
{
  "dataset": "ej2/dataset.csv",
  "target": "y",
  "klist": "3,4,5,6,8,10",
  "kfolds": 5,
  "test_fold": 1,
  "reps": 5,
  "epochs": 1000,
  "lr": 0.01,
  "activation": "tanh",
  "beta": 1.0,
  "out": "ej2/results/folds/cv_study.json",
  "save_all_folds_curves": true,
  "all_folds_curves_out": "ej2/results/folds/curves_all_folds_k5.json"
}
```



#### Comparación de activaciones y búsqueda de hiperparámetros
Ejecuta el barrido de **β** y **learning rate** para las activaciones `tanh`, `sigmoid` y `linear`:

```bash
python -m ej2.compare_activations --config ej2/config.json 
```

Guarda:
- `all_trials.csv`
- `bests.csv`

#### Visualización de comparaciones
Graficá los resultados anteriores (MSE vs β, MSE vs LR, comparación entre activaciones):

```bash
python -m ej2.plot_comparisons --results_dir ej2/results/compare --out_dir ej2/results/plots 
```

Genera:
- `*_mse_train_vs_beta.png`
- `*_mse_train_vs_lr.png`
- `comparative_best_models_bar.png`
- `all_best_train_histories.png`

#### Estudio de generalización (barrido de K y folds)
Evalúa cómo cambia la generalización con distintos K y folds:

```bash
python -m ej2.compare_folds --config ej2/config_fold.json 
```

Genera:
- `cv_study.json` → resumen de promedios, desviaciones y percentiles por K y por fold.
- `curves_all_folds_kX.json` → curvas `MSE_train` y `MSE_test` por época (si se activa `save_all_folds_curves`).

#### Visualización del estudio de generalización
Genera gráficos resumen del barrido de K y de folds:

```bash
python -m ej2.plot_generalization  --study ej2/results/cv_study.json  --outdir ej2/results/plots/folds --all_folds_curves ej2/results/curves_all_folds.json  
```

Produce:
- `bar_mse_vs_k.png`
- `bar_folds_in_bestk.png`
- `learning_curves_foldX.png`
- `test_scatter_foldX.png`


#### Análisis de datos por fold
Muestra cómo se distribuyen las features y el target en cada fold:

```bash
python -m ej2.plot_folds --config ej2/config_fold.json --study ej2/results/cv_study.json --out_dir ej2/results/plots/folds_analysis
```

Genera:
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