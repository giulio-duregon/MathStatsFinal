import numpy as np
import pandas as pd


def make_label_multidimensional(y):
    """
    Make vector of labels (y) from 1d to 10d, with 1 at the index of the true class,
    0 otherwise i.e. y = [2] -> y = [0,0,1,0,0,0,0,0,0,0]
    :param y:
    :return:
    """

    output = []
    for val in y:
        # Number of classes to guess from
        temp = np.zeros(10)
        temp[val] = 1
        output.append(temp)

    return np.array(output).reshape(-1, 10)


def get_train():
    train_x = pd.read_csv("data/mnist_train.csv").iloc[:, 1:]
    train_y = pd.read_csv("data/mnist_train.csv")['label']
    train_x, train_y = train_x.to_numpy().reshape(-1, 28 * 28), train_y.to_numpy().reshape(-1, 1)
    train_y = make_label_multidimensional(train_y)
    assert len(train_x) == len(train_y)
    return train_x, train_y


def get_test():
    test_y = pd.read_csv("data/mnist_test.csv")['label']
    test_x = pd.read_csv("data/mnist_test.csv").iloc[:, 1:]
    test_x, test_y = test_x.to_numpy().reshape(-1, 28 * 28), test_y.to_numpy().reshape(-1, 1)
    test_y = make_label_multidimensional(test_y)
    assert len(test_x) == len(test_y)
    return test_x, test_y
