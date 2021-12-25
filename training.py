import numpy as np
from discriminator import Discriminator
from generator import Generator
from numpy import random
import pickle
import matplotlib.pyplot as plt


np.random.seed(42)

# Hyperparameters
lr = 0.01
epochs = 1000

def train_model():
    faces = [np.array([1,0,0,1]),
        np.array([0.9,0.1,0.2,0.8]),
        np.array([0.9,0.2,0.1,0.8]),
        np.array([0.8,0.1,0.2,0.9]),
        np.array([0.8,0.2,0.1,0.9])]
    
    # model
    D = Discriminator(learning_rate=lr)
    G = Generator(learning_rate=lr)

    # errors for generator and discriminator
    errors_discriminator = []
    errors_generator = []

    for _ in range(epochs):
        for face in faces:
            D.update_from_image(face)

            z = random.rand()

            errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))
            errors_generator.append(G.error(z, D))

            noise = G.forward(z)

            D.update_from_noise(noise)

            G.update(z, D)

    return D, G, errors_discriminator, errors_generator


def save_to_file(nn_type, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(nn_type, fp)


def save_errors(errors_discriminator, errors_generator):
    save_to_file(errors_discriminator, 'errors/discriminator.txt')
    save_to_file(errors_generator, 'errors/generator.txt')


def save_models(D, G):
    save_to_file(D, 'models/discriminator.txt')
    save_to_file(G, 'models/generator.txt')


def load_trained_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data


if __name__ == '__main__':
    D, G, errors_discriminator, errors_generator = train_model()
    save_errors(errors_discriminator, errors_generator)
    save_models(D, G)

    err_discriminator = load_trained_data('errors/discriminator.txt')
    err_generator = load_trained_data('errors/generator.txt')
    plt.plot(err_generator)
    plt.title("Generator error function")
    plt.legend("Generator")
    plt.show()
    plt.plot(err_discriminator)
    plt.legend('Discriminator')
    plt.title("Discriminator error function")
    plt.show()
