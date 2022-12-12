import numpy as np
from sklearn.metrics import mean_squared_error as mse


def input_checker(func):
    """
    Wrapper to run certain checks for different types of losses
    :param func:
    :return:
    """
    def loss_checking(y_hat, test_y):
        if test_y is None or y_hat is None:
            raise ValueError(f"One of inputs is None, test_y: {test_y}, y_hat:{y_hat}")

        if len(test_y) == 0 or len(y_hat) == 0:
            raise ValueError(f"One of inputs is 0, test_y: {test_y}, y_hat:{y_hat}")

        if len(test_y) != len(y_hat):
            raise ValueError(f"Unequal Inputs, test_y: {test_y}, y_hat:{y_hat}")

        return func(y_hat, test_y)

    return loss_checking



@input_checker
def zero_one_loss(test_y: np.array, y_hat: np.array) -> float:
    """
    Average of 0-1 loss for two sets
    :param test_y:
    :param y_hat:
    :return: Loss: float
    """
    # Initialize loss
    loss = 0

    for row, label_vector in enumerate(y_hat):
        correct_index = label_vector.argmax()
        if y_hat[row][correct_index] != 1:
            loss += 1

    return loss / len(y_hat)


@input_checker
def square_loss(test_y: np.array, y_hat: np.array) -> float:
    return mse(test_y, y_hat)