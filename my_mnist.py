from sklearn.datasets import fetch_mldata
import numpy as np


def download_mnist_data(normalize=True):
    mnist = fetch_mldata('MNIST original', data_home='mnist_data')
    # mnist.data : 70,000件の784次元ベクトルデータ
    mnist.data = mnist.data.astype(np.float32)
    if normalize:
        mnist.data /= 255     # 0-1のデータに変換
    # mnist.target : 正解データ
    mnist.target = mnist.target.astype(np.int32)
    return mnist


def load_mnist(N=60000):
    mnist = download_mnist_data(normalize=False)
    x_train, x_test = np.split(mnist.data, [N])
    t_train, t_test = np.split(mnist.target, [N])
    return ((x_train, t_train), (x_test, t_test))


def main():
    mnist = download_mnist_data()
    N = 60000
    x_train, x_test = np.split(mnist.data, [N])
    t_train, t_test = np.split(mnist.target, [N])
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)


if __name__ == '__main__':
    main()
