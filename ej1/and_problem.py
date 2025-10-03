import numpy as np
from perceptrons.simple.perceptron import Perceptron

def solve_and_problem():
    """
    Resuelve el problema de la función lógica AND con un perceptrón simple.
    """
    print("\n--- Resolviendo la función lógica: AND ---")
    
    # Datos de entrenamiento para AND
    x_and = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    # Crear y entrenar el perceptrón
    input_dim = x_and.shape[1]
    perceptron = Perceptron(input_size=input_dim, learning_rate=0.1)
    converged = perceptron.train(x_and, y_and)

    # Probar el perceptrón entrenado
    print("\nResultados después del entrenamiento para AND:")
    correct_predictions = 0
    for inputs_test, label_test in zip(x_and, y_and):
        prediction = perceptron.predict(inputs_test)
        is_correct = "Correcto" if prediction == label_test else "Incorrecto"
        if prediction == label_test:
            correct_predictions += 1
        print(f"Entrada: {inputs_test}, Salida Esperada: {label_test}, Predicción: {prediction} -> {is_correct}")
    
    accuracy = correct_predictions / len(x_and) * 100
    print(f"Precisión para AND: {accuracy}%")
    if converged:
        print("El perceptrón APRENDIÓ exitosamente la función AND.")
    else:
        print("El perceptrón NO PUDO aprender la función AND.")
    print("-" * 40)
    return converged, accuracy

if __name__ == "__main__":
    solve_and_problem()
