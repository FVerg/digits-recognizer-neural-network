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
        # Usage: weights[0] - Weight matrix between level 0 (Input) and 1 (Hidden)
        # Usage:  weights[1] - Weight matrix between level 1 (Hidden) and 2 (Output)

    # Function that calculates the sigmoid of the given vector
    # z = wx + b
    # sigmoid(z) = 1 / 1 + exp(-z)
    # Meaning:
    #  - If z >> 0 -> exp(-z) = 0   -> sigmoid(z) = 1
    #  - If z << 0 -> exp(-z) = inf -> sigmoid(z) = 0
    def sigmoid(z):
        return 1.0/(1.0)+np.exp(-z)

    # Function that takes the input vector and return the output of the network
    def feedforward(self, a):
        # Returns the output of the network if a is an input vector
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            return a

    """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
