import numpy as np
import matplotlib.pyplot as plt
import seaborn


def step_function(x):
    return np.array(0 < x, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    y = relu(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
