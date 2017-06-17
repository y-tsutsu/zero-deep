import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn
import math


def main():
    x = np.arange(0, 2 * math.pi, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, label='cos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin & cos')
    plt.legend()
    plt.show()

    img = imread('lena.png')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
