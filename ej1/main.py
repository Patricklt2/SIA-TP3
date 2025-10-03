from .and_problem import solve_and_problem
from .xor_problem import solve_xor_problem

def run_exercise_1():
    """
    Ejecuta las pruebas para el Ejercicio 1, evaluando las funciones AND y XOR.
    """
    print("=============================================")
    print("=         EJERCICIO 1 - PERCEPTRÓN          =")
    print("=============================================")

    # Resolver AND
    solve_and_problem()

    # Resolver XOR
    solve_xor_problem()

if __name__ == "__main__":
    run_exercise_1()
