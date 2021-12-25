import numpy as np
from utils import sigmoid


class Generator(object):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.biases = np.array([np.random.normal() for i in range(4)])

    def forward(self, z):
        return sigmoid(z * self.weights + self.biases)

    def error(self, z, discriminator):
        x = self.forward(z)
        # we want prediction to be 0, so the error is negative log of (1 - prediction)
        y = discriminator.forward(x)
        return - np.log(y)

    def d(self, z, discriminator):
        discriminator_weights = discriminator.weights
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        
        factor = - (1 - y) * discriminator_weights * x * (1 - x)
        d_weights = factor * z
        d_bias = factor
        return {'weights': d_weights, 'bias': d_bias}

    def update(self, z, discriminator):
        error_before = self.error(z, discriminator)
        d = self.d(z, discriminator)

        self.weights -= self.learning_rate * d['weights']
        self.biases -= self.learning_rate * d['bias']

        error_after = self.error(z, discriminator)
