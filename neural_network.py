import numpy as np
import matplotlib.pyplot as plt
import seaborn
from my_mnist import load_mnist
from PIL import Image
import pickle


def step_function(x):
    return np.array(0 < x, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def init_network():
    # network = {}
    # network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # network['b1'] = np.array([0.1, 0.2, 0.3])
    # network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    # network['b2'] = np.array([0.1, 0.2])
    # network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    # network['b3'] = np.array([0.1, 0.2])
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def check_show_img():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    show_img(img)


def check_accuracy():
    (_, __), (x_test, t_test) = load_mnist(normalize=True)
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x_test)):
        y = predict(network, x_test[i])
        p = np.argmax(y)
        if p == t_test[i]:
            accuracy_cnt += 1

    print('Accuracy: {0}'.format(accuracy_cnt / len(x_test)))


def check_accuracy_batch():
    (_, __), (x_test, t_test) = load_mnist(normalize=True)
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t_test[i:i + batch_size])

    print('Accuracy: {0}'.format(accuracy_cnt / len(x_test)))


def main():
    check_accuracy()
    check_accuracy_batch()


if __name__ == '__main__':
    main()
