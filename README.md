# SIA-TP3

## Requisitos
- Python 3.8+  
- Librerías necesarias:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `tensorflow` 

Instalación rápida:  
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

---

## Ejercicio 1

### Descripción
Este proyecto implementa un **perceptrón simple** (sin librerías de machine learning) para resolver los problemas lógicos planteados.

Se entrenan y evalúan dos funciones lógicas:  
- **AND** (`and_problem.py`)  
- **XOR** (`xor_problem.py`)  

El script principal (`main.py`) ejecuta ambos experimentos y muestra los resultados.

### Ejecución

Para correr la simulación del Ejercicio 1, asegúrate de estar en la carpeta raíz del proyecto y ejecuta el siguiente comando en tu terminal:

```bash
python3 -m ej1.main
```

## Ejercicio 2

### Descripción
En este ejercicio se implementa un **perceptrón simple lineal** y un **perceptrón simple no lineal** (con activaciones `sigmoid` y `tanh`) para comparar su capacidad de aprendizaje sobre el conjunto de datos provisto (`TP3-ej2-conjunto.csv`).

Se entrenan los modelos utilizando **validación cruzada (k-fold)** y se evalúan con métricas como:
- **MSE** (Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R²** (Coeficiente de determinación)  

Además, se generan gráficos de curvas de aprendizaje, comparaciones de métricas y predicciones vs valores reales.


### Ejecución
Desde la carpeta raíz del proyecto, correr:

```bash
python -m ej2.exercise2
```

Esto hará lo siguiente:
1. Leer el dataset `TP3-ej2-conjunto.csv`.  
2. Entrenar perceptrones lineales y no lineales con distintas configuraciones de hiperparámetros (`beta`).  
3. Guardar resultados en la carpeta `results/`:  
   - `metrics_comparison.png` → comparación de métricas entre modelos.  
   - `Learning_Curves-<activation>.png` → curvas de aprendizaje para cada activación.  
   - `Predictions_vs_True_Values-...png` → gráfico de comparación entre predicciones y valores reales.  
   - `performance_metrics.csv` → archivo resumen con métricas de cada modelo.  

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
