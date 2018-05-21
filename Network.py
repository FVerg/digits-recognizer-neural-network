import numpy as np

# Input:
# - sizes: Vector containing number of neurons for each layer:
#     - [1,2,1]: Input - 1 Neuron | Hidden - 2 Neurons | Output - 1 Neuron
# Example usage: net = Network([1,2,1])


class Network (object):
    def __init__(self, sizes):
        # Number of layers of the network
        self.num_layers = len(sizes)
        # Number of neurons for each layer
        self.sizes = sizes
        # Initial biases, calculated randomly
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(
            sizes[:-1], sizes[1:])]   # Initial weights, calculated randomly
