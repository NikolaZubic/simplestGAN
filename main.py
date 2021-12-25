from numpy import random
from training import load_trained_data
import matplotlib.pyplot as plt


def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharex=True, sharey=True)

    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2,2)), cmap='Greys_r')
    
    return fig, axes

if __name__ == '__main__':
    G = load_trained_data('models/generator.txt')
    D = load_trained_data('models/discriminator.txt')    
    
    generated_images = []
    
    for i in range(4):
        z = random.random()
        generated_image = G.forward(z)
        generated_images.append(generated_image)
        view_samples(generated_images, 1, 4)
    plt.show()
    for i in generated_images:
        print(i)
