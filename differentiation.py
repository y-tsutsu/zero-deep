import numpy as np
import matplotlib.pyplot as plt
import seaborn


def func(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def get_diff_func(f, x):
    a = numerical_diff(func, x)
    print(a)
    b = f(x) - a * x
    return lambda x: a * x + b


def main():
    x = np.arange(0.0, 20.0, 0.1)
    y = func(x)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(x, y)

    diff_func = get_diff_func(func, 5)
    y = diff_func(x)
    plt.plot(x, y)

    diff_func = get_diff_func(func, 10)
    y = diff_func(x)
    plt.plot(x, y)

    plt.xlim([0, 20])
    plt.ylim([0, 6])
    plt.show()


if __name__ == '__main__':
    main()
