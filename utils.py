import numpy as np

def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))
