import pandas as pd


def get_train():
    #TODO: Fix these paths
    train_x = pd.read_csv("Path To/mnist_train.csv").iloc[:, 1:]
    train_y = pd.read_csv("Path To/mnist_train.csv")['label']
    return train_x, train_y


def get_test():
    test_y = pd.read_csv("Path To/mnist_test.csv")['label']
    test_x = pd.read_csv("Path To/mnist_test.csv").iloc[:, 1:]
    return test_x, test_y
