import numpy as np
from perceptrons.simple.perceptron import Perceptron

def solve_xor_problem():
    """
    Intenta resolver el problema de la función lógica XOR con un perceptrón simple.
    """
    print("\n--- Resolviendo la función lógica: XOR ---")
    
    # Datos de entrenamiento para XOR
    x_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y_xor = np.array([1, 1, -1, -1])

    # Crear y entrenar el perceptrón
    input_dim = x_xor.shape[1]
    perceptron = Perceptron(input_size=input_dim, learning_rate=0.1)
    converged = perceptron.train(x_xor, y_xor)

    # Probar el perceptrón entrenado
    print("\nResultados después del entrenamiento para XOR:")
    correct_predictions = 0
    for inputs_test, label_test in zip(x_xor, y_xor):
        prediction = perceptron.predict(inputs_test)
        is_correct = "Correcto" if prediction == label_test else "Incorrecto"
        if prediction == label_test:
            correct_predictions += 1
        print(f"Entrada: {inputs_test}, Salida Esperada: {label_test}, Predicción: {prediction} -> {is_correct}")
    
    accuracy = correct_predictions / len(x_xor) * 100
    print(f"Precisión para XOR: {accuracy}%")
    if converged:
        print("El perceptrón APRENDIÓ exitosamente la función XOR.")
    else:
        print("El perceptrón NO PUDO aprender la función XOR.")
    print("-" * 40)
    return converged, accuracy

if __name__ == "__main__":
    solve_xor_problem()
