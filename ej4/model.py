import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Softmax
from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.loss import cce, cce_prime
from perceptrons.multicapa.optimizers import Adam

def create_mnist_mlp():
    layers = [
        Dense(input_size=28*28, output_size=128),
        Tanh(),
        Dense(input_size=128, output_size=10),
        Softmax()
    ]

    optimizer = Adam(learning_rate=0.001)
    mlp = MLP(layers, loss=cce, loss_prime=cce_prime, optimizer=optimizer)
    
    return mlp