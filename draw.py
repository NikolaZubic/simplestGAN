import numpy as np
from matplotlib import pyplot as plt


def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharex=True, sharey=True)

    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2,2)), cmap='Greys_r')
    
    return fig, axes


if __name__ == '__main__':
    noise = [np.random.randn(2,2) for i in range(20)]
    view_samples(noise, 4,5)

    plt.show()
