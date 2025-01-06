
# Importing Libraries
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
from Model import ClassificationModels, RegressionModels, Model
    
class KNN(Model):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def _pretreat(self, X, y):
        return super()._pretreat(X, y)
    
    def fit(self, X, y):
        X, y, n_samples, n_features = self._pretreat(X, y)
        self.X_train = X
        self.y_train = y
    
    def _distance(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))



class KNNClassifier(KNN, ClassificationModels):
    def __init__(self, k=3):
        super().__init__(k)
    
    def _pretreat(self, X, y):
        return super()._pretreat(X, y)
    
    def fit(self, X, y):
        super().fit(X, y)
    

    def _distance(self, x):
        return super()._distance(x)

    
    def _predict(self, x):
        distances = self._distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels.squeeze()).argmax()
        return most_common

    def predict(self, X):
        X, _, n_samples, n_features = self._pretreat(X, None)
        return np.array([self._predict(x) for x in X])
    




    
class KNNRegressor(KNN, RegressionModels):
    def __init__(self, k=3):
        super().__init__(self, k)
    
    def _pretreat(self, X, y):
        return super()._pretreat(X, y)
    
    def fit(self, X, y):
        X, y, _, _ = self.pretreat(X, y)
        super().fit(X, y)
    

    def _distance(self, x):
        return super()._distance(x)

    
    def _predict(self, x):
        distances = self._distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.mean(k_nearest_labels)

    def predict(self, X):

        X, _, n_samples, n_features = self._pretreat(X, None)
        return np.array([self._predict(x) for x in X])