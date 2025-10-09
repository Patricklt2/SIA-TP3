# /home/pipemind/sia/SIA-TP3/ej3/data_utils.py
import numpy as np
import re

def load_digits_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo de dígitos en: {file_path}")

    all_pixels_str = re.sub(r'\s+', '', content)
    all_pixels = [int(p) for p in all_pixels_str]

    num_digits = len(all_pixels) // 35
    X = []
    for i in range(num_digits):
        start_index = i * 35
        end_index = start_index + 35
        digit_pixels = all_pixels[start_index:end_index]
        X.append(digit_pixels)

    if not X:
        raise ValueError("No se pudieron cargar los dígitos del archivo. Revisa el formato.")

    y = np.arange(num_digits)

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
