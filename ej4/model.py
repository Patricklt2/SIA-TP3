from perceptrons.multicapa.layers import Dense
from perceptrons.multicapa.activation import Tanh, Softmax, ReLU
from perceptrons.multicapa.mlp import MLP
from perceptrons.multicapa.loss import cce, cce_prime
from perceptrons.multicapa.optimizers import SGD

def create_mnist_mlp():
    layers = [
        Dense(input_size=28*28, output_size=128),
        ReLU(),
        Dense(input_size=128, output_size=10),
        Softmax()
    ]

    optimizer = SGD(learning_rate=0.001)
    mlp = MLP(layers, loss=cce, loss_prime=cce_prime, optimizer=optimizer)
    
    return mlp