import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest


class Model:
    """
    Classe de base pour tous les modèles de machine learning.
    """

    def __init__(self):
        pass
        
    def fit(self, X, y):
        """Méthode d'entraînement, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'fit' doit être implémentée dans la sous-classe.")

    def predict(self, X):
        """Méthode de prédiction, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'predict' doit être implémentée dans la sous-classe.")

    def score(self, X, y):
        """Évalue les performances du modèle (par exemple, avec MSE ou précision)."""
        raise NotImplementedError("La méthode 'score' doit être implémentée dans la sous-classe.")
    
    def Transform_From_Pandas_To_Numpy(self, X):
        columns = None
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            columns = X.columns
            X = X.values
        return X, columns
    
    def Transform_From_Numpy_To_Pandas(self, X, columns):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=columns)
        return X

    def _pretreat(self, X, y):
        """
        Préparation des données : conversion en numpy et mise en forme.
        """
        n_samples, n_features = 0, 0
        if X is not None:
            X, X_columns = self.Transform_From_Pandas_To_Numpy(X)
            self.X_columns = X_columns

            if len(X.shape) > 1:
                n_samples, n_features = X.shape
            else:
                n_features = 1
                n_samples = X.shape[0]
                X = X.reshape(n_samples, n_features)
        if y is not None:
            y, y_columns = self.Transform_From_Pandas_To_Numpy(y)
            self.y_columns = y_columns
            if len(y.shape) == 1:
                y = y.reshape(n_samples, 1)

        return X, y, n_samples, n_features




class RegressionModels(Model): 
    """
    Classe de base pour les modèles de régression.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Méthode d'entraînement, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'fit' doit être implémentée dans la sous-classe.")

    def predict(self, X):
        """Méthode de prédiction, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'predict' doit être implémentée dans la sous-classe.")

    def score(self, X, y):
        """Evaluation d'un modèle de régression."""

        y_pred = self.predict(X)
        y = np.array(y).ravel()
        y_pred = np.array(y_pred).ravel()
        mse = np.mean((y - y_pred)**2)
        rmse = np.mean((y - y_pred)**2)**(1/2)
        mae = np.mean(np.abs(y - y_pred))
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    
    def plot(self, X, y):
        """Affiche un graphique de la régression."""
        plt.scatter(X, y, color='black')
        plt.plot(X, self.predict(X), color='blue', linewidth=3)
        plt.show()

class ClassificationModels(Model):
    """
    Classe de base pour les modèles de classification.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Méthode d'entraînement, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'fit' doit être implémentée dans la sous-classe.")

    def predict(self, X):
        """Méthode de prédiction, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'predict' doit être implémentée dans la sous-classe.")
    def score(self, X, y):
            
        """Evaluation d'un modèle de classification."""
        X, y, _, _ = self._pretreat(X, y)
        y_pred = self.predict(X)
        y_pred, y = y_pred.squeeze(), y.squeeze()
        accuracy = np.mean(y == y_pred)
        recall = np.sum((y == [1]) & (y_pred == [1])) / np.sum(y == [1])
        precision = np.sum((y == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1 Score": f1_score}


