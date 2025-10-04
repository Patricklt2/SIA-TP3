# /home/pipemind/sia/SIA-TP3/ej3/data_utils.py
import numpy as np

def load_digits_data(file_path):
    """
    Carga los datos de los dígitos desde el archivo de texto.
    Esta versión es más robusta y maneja el formato del archivo correctamente.

    Args:
        file_path (str): La ruta al archivo 'TP3-ej3-digitos.txt'.

    Returns:
        tuple: Una tupla conteniendo:
            - X (np.array): Un array con 10 muestras, cada una un vector de 35 características.
            - y (np.array): Un array con las 10 etiquetas numéricas (0-9).
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Los dígitos están separados por una o más líneas en blanco.
    # Usamos split() para separar los bloques de dígitos.
    digit_blocks = content.strip().split('\n\n')
    
    X = []
    y = []

    for i, block in enumerate(digit_blocks):
        if not block.strip():
            continue

        digit_lines = block.strip().split('\n')
        digit_vector = []
        
        for line in digit_lines:
            # Reemplazar espacios con '0' y tomar solo los primeros 5 caracteres
            # para asegurar que cada línea tenga 5 píxeles.
            processed_line = line.replace(' ', '0')[:5]
            pixels = [int(p) for p in processed_line]
            digit_vector.extend(pixels)
        
        # Asegurarse de que el vector tenga 35 píxeles, rellenando si es necesario
        if len(digit_vector) < 35:
            digit_vector.extend([0] * (35 - len(digit_vector)))
        
        X.append(digit_vector[:35]) # Tomar solo los primeros 35
        y.append(i)

    num_digits = len(X)
    X = np.array(X).reshape(num_digits, 35, 1)
    y = np.array(y).reshape(num_digits, 1)
    
    return X, y

def get_parity_labels(y_digits):
    """Convierte las etiquetas de dígitos (0-9) a etiquetas de paridad (0=par, 1=impar)."""
    return (y_digits % 2).astype(int)

def get_one_hot_labels(y_digits):
    """Convierte las etiquetas de dígitos (0-9) a formato one-hot."""
    num_classes = 10
    one_hot_labels = np.zeros((len(y_digits), num_classes, 1))
    for i, digit in enumerate(y_digits):
        one_hot_labels[i][digit.item()] = 1
    return one_hot_labels

if __name__ == '__main__':
    # Ejemplo de uso para verificar que funciona
    file_path = 'TP3-ej3-digitos.txt'
    X, y_digits = load_digits_data(file_path)
    
    print("--- Verificación del Cargador de Datos ---")
    print(f"Forma de X (datos de entrada): {X.shape}")
    print(f"Forma de y (etiquetas de dígitos): {y_digits.shape}")
    
    # Probar la conversión a etiquetas de paridad
    y_parity = get_parity_labels(y_digits)
    print("\nEtiquetas de Dígitos (0-9):", y_digits.flatten())
    print("Etiquetas de Paridad (0=par, 1=impar):", y_parity.flatten())
    
    # Probar la conversión a one-hot
    y_one_hot = get_one_hot_labels(y_digits)
    print(f"\nForma de y_one_hot: {y_one_hot.shape}")
    print("Ejemplo de etiqueta one-hot para el dígito '3':")
    print(y_one_hot[3].flatten())
    print("-----------------------------------------")
