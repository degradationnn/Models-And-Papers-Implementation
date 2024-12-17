import numpy as np

from Model_Selection import KFolds, learning_curve


def test_KFolds():
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    kf = KFolds(n_splits=5, shuffle=False, random_state=42)
    folds = kf.split(X)
    assert len(folds) == 5
    assert len(folds[0]) == 2

    

