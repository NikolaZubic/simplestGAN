# simplestGAN
Simplest GAN implementation from scratch (using [NumPy](https://www.numpy.org/)) demonstrated on the following problem:<br>
We are given a matrix of dimension 2 x 2. Values of matrix are in range [0, 1], where 0 is white, and 1 is a black color. <br>
We start from a noisy distribution, and the goal is to generate an image where on the main diagonal we have black color, on side diagonal white color.

## Important files
* training.py - for the process of training
* main.py - to get the results from random vector z

## Results
From noisy images (random distribution):<br>
![](https://github.com/NikolaZubic/simplestGAN/blob/main/__pycache__/noisy.png)
<br><br>
To the desired results (wanted distribution):<br>
![](https://github.com/NikolaZubic/simplestGAN/blob/main/__pycache__/result.png)
