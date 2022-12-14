import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from functools import wraps
from time import time
from .losses import zero_one_loss, square_loss
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

LOSS_FACTORY = {
    "zero_one_loss": zero_one_loss,
    "mse": square_loss
}


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print(f"Function: {f.__name__}, Time Delta: {end - start}")
        return result

    return wrap


class AbstractModel(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class RandomForestWrapper(AbstractModel):
    def __init__(self, max_leaf_nodes, num_estimators, bootstrap: bool = False, ):
        self.model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, n_estimators=num_estimators,
                                            bootstrap=bootstrap)

    @timing
    def train(self, x, y):
        self.model.fit(x, y)

    @timing
    def get_loss(self, loss_type: str, y_true, y_hat):
        loss = LOSS_FACTORY[loss_type]
        return loss(y_true, y_hat)

    @timing
    def predict(self, x_test):
        x_test = x_test.reshape(-1, 784)
        return self.model.predict(x_test)


class ExperimentRunner:
    def __init__(self, model_cls, model_params_iter: list, train_data, test_data):
        if not issubclass(model_cls, AbstractModel):
            raise ValueError(f"model_cls must be a subclass of {AbstractModel.__class__}, got {model_cls} instead")
        self.model_constructor = model_cls
        self.model_params_iter = model_params_iter
        self.num_experiments = len(model_params_iter)
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data

        self.square_losses = {"train_losses": [],
                              "test_losses": [],
                              }
        self.zero_one_losses = {"train_losses": [],
                                "test_losses": []}

        self.param_axis_labels = self.parsed_params_arr()

    def parsed_params_arr(self):
        axis_labels = list()
        for params in self.model_params_iter:
            temp = []
            for key,value in params.items():
                temp.append(value)
            axis_labels.append(temp)
        return axis_labels

    def run(self):
        kwargs: dict
        model: AbstractModel

        for kwargs in self.model_params_iter:
            # Create model
            model = self.model_constructor(**kwargs)

            # Fit model
            model.train(self.train_x, self.train_y)

            # Get Preds
            train_y_hat = model.predict(self.train_x).reshape(-1,10)
            test_y_hat = model.predict(self.test_x).reshape(-1,10)

            # Get Train losses
            self.square_losses["train_losses"].append(model.get_loss("mse", self.train_y, train_y_hat))
            self.zero_one_losses["train_losses"].append(model.get_loss("zero_one_loss", self.train_y, train_y_hat))

            # Get Test losses
            self.square_losses["test_losses"].append(model.get_loss("mse", self.test_y, test_y_hat))
            self.zero_one_losses["test_losses"].append(
                model.get_loss("zero_one_loss", self.test_y, test_y_hat))

    def plot_zero_one_loss(self):
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(111)
        ax1.set_xticklabels(self.param_axis_labels)
        x_axis_arr = range(self.num_experiments)
        plt.plot(x_axis_arr, self.zero_one_losses["test_losses"], color='g')
        plt.plot(x_axis_arr, self.zero_one_losses["train_losses"], color='c')
        plt.ylabel('Average Zero/One Loss', size=30)
        plt.legend(['test_data', 'train_data'], loc='best')
        plt.xlabel("Model Increasing Complexity", size=30)
        plt.title("Model Parameters", size=30)
        plt.show()

    def plot_square_loss(self):
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(111)
        ax1.set_xticklabels(self.param_axis_labels)

        x_axis_arr = range(self.num_experiments)
        plt.plot(x_axis_arr, self.square_losses["test_losses"], color='g')
        plt.plot(x_axis_arr, self.square_losses["train_losses"], color='c')
        plt.ylabel('Average Squared Loss', size=30)
        plt.legend(['test_data', 'train_data'], loc='best')
        plt.xlabel("Model Increasing Complexity", size=30)
        plt.title("Model Parameters", size=30)
        plt.show()



        fig.tight_layout()

    def results_to_csv(self, fpath: str = None):
        # Collect the data
        train_zero_one_loss = self.zero_one_losses["train_losses"]
        test_zero_one_loss = self.zero_one_losses["test_losses"]
        train_squared = self.square_losses["train_losses"]
        test_squared = self.square_losses["test_losses"]
        max_leafs = [x["max_leaf_nodes"] for x in self.model_params_iter]
        num_estimators = [x["num_estimators"] for x in self.model_params_iter]

        # Store in column orientation
        columns = ["num_estimators","max_leaf_nodes", "train_squared_error","test_squared_error", "train_zero_one_loss", "test_zero_one_loss"]
        data = np.array([num_estimators, max_leafs, train_squared, test_squared, train_zero_one_loss, test_zero_one_loss]).T

        # Create dataframe
        df = pd.DataFrame(data=data, columns=columns)

        # Save
        if fpath:
            df.to_csv(fpath)
        else:
            df.to_csv("RF_Experiment_Results.csv")

