import pytest
from biasvariancetradeoff.losses import *
from contextlib import nullcontext as does_not_raise

edge_cases = [
    ([0, ], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pytest.raises(ValueError), None),
    ([0, 0], [0, ], pytest.raises(ValueError), None),
    ([], [0, 1], pytest.raises(ValueError), None),
    (None, [0, 1], pytest.raises(ValueError), None),
    ([], [], pytest.raises(ValueError), None),
    ([], [], pytest.raises(ValueError), None),
    (list(range(1, 11)), [0]*4 + list(range(5, 11)), does_not_raise(), .4),
]


@pytest.mark.parametrize(("test_y", "y_hat", "expected","value"), edge_cases)
def test_classification_loss(test_y, y_hat, expected, value):
    with expected:
        assert value == zero_one_loss(test_y, y_hat)
