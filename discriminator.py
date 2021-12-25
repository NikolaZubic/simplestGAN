import numpy as np
from utils import sigmoid


class Discriminator(object):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.array([np.random.normal() for i in range(4)]) # since we have images 2 x 2
        self.bias = np.random.normal()

    def forward(self, x):
        return sigmoid(np.dot(x, self.weights) + self.bias)
    
    def error_from_image(self, image):
        # prediction must be 1, so the error will be negative log of the prediction
        prediction = self.forward(image)
        return - np.log(prediction)
    
    def d_from_image(self, image):
        prediction = self.forward(image)
        """
        derivative(E) / derivative(w_i) = (derivative(E) / derivative(D)) * (derivative(D) / derivative(w_i))
        derivativ(E) / derivative(w_i) = - (1 - D(x)) * x_i
        """
        d_weights = - (1 - prediction) * image
        """
        derivative(E) / derivative(b) = (derivative(E) / derivative(D)) * (derivative(D) / derivative(b))
        derivative(E) / derivative(b) = - (1 - D(x))
        """
        d_bias = - (1 - prediction)

        return {'weights': d_weights, 'bias': d_bias}

    def update_from_image(self, x):
        d = self.d_from_image(x)
        self.weights -= self.learning_rate * d['weights']
        self.bias -= self.learning_rate * d['bias']

    def error_from_noise(self, noise):
        # prediction must be 0, so the error will be negative log of the (1 - prediction)
        prediction = self.forward(noise)
        return - np.log(1 - prediction)
    
    def d_from_noise(self, noise):
        prediction = self.forward(noise)
        """
        derivative(E) / derivative(w_i) = (derivative(E) / derivative(D)) * (derivative(D) / derivative(w_i))
        derivativ(E) / derivative(w_i) = D(x) * x_i
        """
        d_weights = prediction * noise
        """
        derivative(E) / derivative(b) = (derivative(E) / derivative(D)) * (derivative(D) / derivative(b))
        derivative(E) / derivative(b) = D(x)
        """
        d_bias = prediction

        return {'weights': d_weights, 'bias': d_bias}

    def update_from_noise(self, noise):
        d = self.d_from_noise(noise)
        self.weights -= self.learning_rate * d['weights']
        self.bias -= self.learning_rate * d['bias']
