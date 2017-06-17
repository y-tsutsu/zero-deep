import numpy as np


def common(x, w, b):
    x = np.array([x[0], x[1]])
    w = np.array([w[0], w[1]])
    if np.sum(x * w) + b <= 0:
        return 0
    return 1


def AND(x1, x2):
    '''
    >>> AND(0, 0)
    0
    >>> AND(0, 1)
    0
    >>> AND(1, 0)
    0
    >>> AND(1, 1)
    1
    '''
    return common((x1, x2), (1.0, 1.0), -1)


def OR(x1, x2):
    '''
    >>> OR(0, 0)
    0
    >>> OR(0, 1)
    1
    >>> OR(1, 0)
    1
    >>> OR(1, 1)
    1
    '''
    return common((x1, x2), (1.0, 1.0), -0.9)


def NAND(x1, x2):
    '''
    >>> NAND(0, 0)
    1
    >>> NAND(0, 1)
    1
    >>> NAND(1, 0)
    1
    >>> NAND(1, 1)
    0
    '''
    return common((x1, x2), (-1.0, -1.0), 1.1)


def XOR(x1, x2):
    '''
    >>> XOR(0, 0)
    0
    >>> XOR(0, 1)
    1
    >>> XOR(1, 0)
    1
    >>> XOR(1, 1)
    0
    '''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()
