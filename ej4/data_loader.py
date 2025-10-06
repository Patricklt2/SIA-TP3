import numpy as np
from tensorflow import keras

def load_data(train_samples=10000, test_samples=2000):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()

    X_train_raw = x_train_full
    y_train_raw = y_train_full
    X_test_raw = x_test_full
    y_test_labels = y_test_full

    X_train = X_train_raw.astype('float32') / 255.0
    X_test = X_test_raw.astype('float32') / 255.0

    X_train = X_train.reshape(len(X_train), 28*28, 1)
    X_test = X_test.reshape(len(X_test), 28*28, 1)

    y_train = np.zeros((len(y_train_raw), 10, 1))
    for i, label in enumerate(y_train_raw):
        y_train[i][label] = 1

    y_test_one_hot = np.zeros((len(y_test_labels), 10, 1))
    for i, label in enumerate(y_test_labels):
        y_test_one_hot[i][label] = 1
        
    return X_train, y_train, X_test, y_test_one_hot, y_test_labels