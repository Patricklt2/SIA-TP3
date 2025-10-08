import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Softmax, ReLU
from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.loss import cce, cce_prime
from perceptrons.multicapa.optimizers import SGD, Adam, AdamW

def create_mnist_mlp():
    layers = [
        Dense(784, 256),
        ReLU(),
        Dense(256, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ]

    optimizer = AdamW(learning_rate=0.0015)
    mlp = MLP(layers, loss=cce, loss_prime=cce_prime, optimizer=optimizer)
    
    return mlp