
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..")))
from Model import ClassificationModels

class GaussianNaiveBayes(ClassificationModels):
    
    """
    Gaussian Naive Bayes classifier.
    
    Parameters:
    priors: array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
    var_smoothing: float, default=1e-9
        Portion of the largest variance of all features that is added to variances for calculation stability.
    
    Attributes:
    classes_: array-like of shape (n_classes,)
        Unique class labels.
    n_classes_: int
        Number of classes.
    n_features_: int       
        Number of features.
    theta_: array-like of shape (n_classes, n_features)
        Mean of each feature per class.
    sigma_: array-like of shape (n_classes, n_features)
        Variance of each feature per class.
    """
    
    def __init__(self, priors = None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes = None
        self.n_classes = None
        self.n_features = None
        self.theta = None
        self.sigma = None
    
    def fit(self, X, y):    
        X, y, _, self.n_features = super()._pretreat(X, y)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.theta = np.zeros((self.n_classes, self.n_features))
        self.sigma = np.zeros((self.n_classes, self.n_features))
        if self.priors is None:
            self.priors = np.array([np.mean(y == c) for c in self.classes])
        for i, c in enumerate(self.classes):
            idx = np.where(y == c)[0]  # Indices des échantillons de la classe c
            X_c = X[idx]  # Sélectionne uniquement les échantillons de la classe c
            self.theta[i] = X_c.mean(axis=0)  # Moyenne pour chaque feature
            self.sigma[i] = X_c.var(axis=0) + self.var_smoothing  # Variance pour chaque feature

    
    #gaussian probability density function
    def _pdf(self, class_idx, x):
        mean = self.theta[class_idx]
        sigma = self.sigma[class_idx]
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / (sigma))
        denominator = np.sqrt(2 * np.pi * sigma)
        return numerator / denominator
        
    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        X, _, _, _ = super()._pretreat(X, None)
        
        return np.array([self._predict(x) for x in X])
    
        
        
        



    