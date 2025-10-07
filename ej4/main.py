import numpy as np
import time
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_loader import load_data
from model import create_mnist_mlp

def solve_mnist_problem():
  
    X_train, y_train, X_test, y_test_one_hot, y_test_labels = load_data(
        train_samples=10000, #mas rapido menos accurate
        test_samples=2000
       # train_samples=60000, # Use all available training samples
        # test_samples=10000   # Use all available test samples
    )

    mlp = create_mnist_mlp()

    print("\nEmpezando el entrenamiento.")
    mlp.train(X_train, y_train, epochs=30, batch_size=32, verbose=True)
    print(f"Entrenamiento completado.")

   
    model_path = os.path.join(os.path.dirname(__file__), 'mnist_model.npz')
    mlp.save_weights(model_path)

    predictions = mlp.predict(X_test)
    predicted_digits = np.argmax(predictions, axis=1)
    
    accuracy = np.mean(predicted_digits.flatten() == y_test_labels.flatten()) * 100
    
    print(f"\nPrecisión final: {accuracy:.2f}%")

if __name__ == "__main__":
    solve_mnist_problem()