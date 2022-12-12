import pytest
from biasvariancetradeoff.random_forest import ExperimentRunner, RandomForestWrapper
import numpy as np

test_params = [{"max_leaf_nodes": 10,
                "num_estimators": 10,
                "bootstrap": False}]
num_experiments = len(test_params)
fake_data = np.array([1, 1, 1, 10, 10, 10]).reshape(-1, 1)
fake_train_data = [(fake_data, fake_data)]
fake_test_data = [(fake_data, fake_data)]



@pytest.mark.parametrize("model_params_iter", test_params)
@pytest.mark.parametrize("train_data", fake_train_data)
@pytest.mark.parametrize("test_data", fake_test_data)
def test_rf_model_runner(model_params_iter, train_data, test_data):
    # Run Experiments
    runner = ExperimentRunner(RandomForestWrapper, test_params, train_data, test_data)
    runner.run()

    # Assert that we have collected data for all experiments
    for key in runner.square_losses.keys():
        assert len(runner.square_losses[key]) == num_experiments

    runner.plot_zero_one_loss()
    runner.plot_square_loss()