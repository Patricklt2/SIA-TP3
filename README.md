# SIA-TP3

## Requisitos
- Python 3.8+  
- Librerías necesarias:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

Instalación rápida:  
```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Ejercicio 1

### Descripción
Este proyecto implementa un **perceptrón simple** (sin librerías de machine learning) para resolver los problemas lógicos planteados.

Se entrenan y evalúan dos funciones lógicas:  
- **AND** (`and_problem.py`)  
- **XOR** (`xor_problem.py`)  

El script principal (`main.py`) ejecuta ambos experimentos y muestra los resultados.

## Ejecución

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


## Ejecución
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