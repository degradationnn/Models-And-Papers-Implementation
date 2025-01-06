import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../Preprocessing", "Preprocessing")))
sys.path.append(os.path.abspath(os.path.join("..")))


from Model import RegressionModels, ClassificationModels


class SGDRegression(RegressionModels):
    """
    Implémentation de la régression par descente de gradient stochastique (SGD).

    Cette classe implémente un modèle de régression linéaire utilisant la descente de gradient stochastique pour l'optimisation.

    Attributs:
    learning_rate (float): Taux d'apprentissage pour la descente de gradient.
    n_iters (int): Nombre d'itérations pour le processus d'entraînement.
    tolerance (float): Tolérance pour le critère d'arrêt.
    weights (numpy.ndarray): Poids du modèle linéaire.
    bias (float): Terme de biais du modèle linéaire.
    X_columns (list): Noms des colonnes des caractéristiques d'entrée.
    y_columns (list): Noms des colonnes de la variable cible.
    losses (list): Liste pour stocker les valeurs de perte pendant l'entraînement.
    learning_rate_method (str): Méthode pour mettre à jour le taux d'apprentissage. Peut être 'constant' ou 'invscaling'.
    pow_t (float): Paramètre de puissance pour la méthode de taux d'apprentissage 'invscaling'.
    """

    def __init__(
        self,
        n_iters=1000,
        learning_rate=0.001,
        tolerance=1e-5,
        learning_rate_method="constant",
        pow_t=0.25,
    ):
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.X_columns = None
        self.y_columns = None
        self.tolerance = tolerance
        self.losses = []
        self.learning_rate_method = learning_rate_method
        self.pow_t = pow_t
        self.learning_rate = learning_rate

    def _pretreat(self, X, y):
        X, X_columns = super().Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        y, y_columns = super().Transform_From_Pandas_To_Numpy(y)
        self.y_columns = y_columns
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        if len(y.shape) == 1:
            y = y.reshape(n_samples, 1)
        return X, y, n_samples, n_features

    def _stochastic_gradient_descent(self, xi, yi):
        # Prédiction pour un échantillon
        y_predicted = np.dot(xi, self.weights) + self.bias

        # Gradients basés sur un seul échantillon
        dw = np.dot(xi.T, (y_predicted - yi))
        db = np.sum(y_predicted - yi)
        return dw, db

    def fit(self, X, y):
        X, y, n_samples, n_features = self._pretreat(X, y)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):  # Parcourir chaque échantillon
                xi = X[i].reshape(1, -1)  # Extraire l'échantillon i
                yi = y[i].reshape(1, -1)  # Extraire la cible i

                if self.learning_rate_method == "invscaling":
                    learning_rate = self.learning_rate / (_ + 1) ** self.pow_t
                elif self.learning_rate_method == "constant":
                    learning_rate = self.learning_rate

                dw, db = self._stochastic_gradient_descent(xi, yi)

                # Mise à jour des poids et biais
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            if np.linalg.norm(dw, ord=2) < self.tolerance and abs(db) < self.tolerance:
                break
            # Calcul de la perte sur tout l'ensemble après chaque itération
            y_predicted_full = np.dot(X, self.weights) + self.bias
            self.losses.append(0.5 * np.mean((y_predicted_full - y) ** 2))

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        y_predicted = np.dot(X, self.weights) + self.bias
        if self.y_columns is not None:
            y_predicted = super().Transform_From_Numpy_To_Pandas(
                y_predicted, self.y_columns
            )
        return y_predicted


class LinearRegression(RegressionModels):
    """
    Implémentation de la régression linéaire avec la solution analytique.
    """

    def __init__(self):
        self.weights = None
        self.bias = None
        self.X_columns = None
        self.y_columns = None

    def _analytical_solution(self, X, y):
        try:
            self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        except np.linalg.LinAlgError:
            # In case of singular matrix, i.e., non-invertible matrix (there is 2 features that are linearly dependent)
            self.weights = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def fit(self, X, y):
        X, X_columns = super().Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        y, y_columns = super().Transform_From_Pandas_To_Numpy(y)
        self.y_columns = y_columns
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        if len(X.shape) == 1:
            y = y.reshape(n_samples, 1)
        X = np.c_[np.ones((n_samples, 1)), X]

        self._analytical_solution(X, y)

        if len(self.weights.shape) == 1:
            self.weights = self.weights.reshape(self.weights.shape[0], 1)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        y_predicted = np.dot(X, self.weights) + self.bias
        if self.y_columns is not None:
            y_predicted = super().Transform_From_Numpy_To_Pandas(
                y_predicted, self.y_columns
            )
        return y_predicted


class RidgeRegression(RegressionModels):
    """
    Ridge Regression implémentée avec régularisation L2.
    """

    def __init__(self, alpha=1):
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.X_columns = None
        self.y_columns = None

    def _analytical_solution(self, X, y):
        try:
            self.weights = np.dot(
                np.linalg.inv(np.dot(X.T, X) + self.alpha * np.eye(X.shape[1])),
                np.dot(X.T, y),
            )
        except np.linalg.LinAlgError:
            self.weights = np.dot(
                np.linalg.pinv(np.dot(X.T, X) + self.alpha * np.eye(X.shape[1])),
                np.dot(X.T, y),
            )
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

    def fit(self, X, y):
        X, X_columns = super().Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        y, y_columns = super().Transform_From_Pandas_To_Numpy(y)
        self.y_columns = y_columns
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        if len(X.shape) == 1:
            y = y.reshape(n_samples, 1)
        X = np.c_[np.ones((n_samples, 1)), X]
        self._analytical_solution(X, y)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        y_predicted = np.dot(X, self.weights) + self.bias
        if self.y_columns is not None:
            y_predicted = super().Transform_From_Numpy_To_Pandas(
                y_predicted, self.y_columns
            )
        return y_predicted


class LassoRegression(RegressionModels):
    """
    Lasso Regression implémentée avec régularisation L1.
    """

    def __init__(self, alpha=1, learning_rate=0.001, n_iters=1000, tolerance=1e-5):
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.n_iters = n_iters
        self.X_columns = None
        self.y_columns = None
        self.losses = []

    def _pretreat(self, X, y):
        X, X_columns = super().Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        y, y_columns = super().Transform_From_Pandas_To_Numpy(y)
        self.y_columns = y_columns
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        if len(y.shape) == 1:
            y = y.reshape(n_samples, 1)
        return X, y, n_samples, n_features

    def _stochastic_gradient_descent(self, X, y, n_samples):
        y_predicted = np.dot(X, self.weights) + self.bias
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(
            self.weights
        )
        db = (1 / n_samples) * np.sum(y_predicted - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        self.losses.append(0.5 * np.mean((y_predicted - y) ** 2))
        # Gestion explicite de weights = 0
        self.weights = np.where(
            np.abs(self.weights) < self.learning_rate * self.alpha, 0, self.weights
        )
        return dw, db

    def fit(self, X, y):
        X, y, n_samples, n_features = self._pretreat(X, y)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0

        for _ in range(self.n_iters):
            dw, db = self._stochastic_gradient_descent(X, y, n_samples)
            if np.linalg.norm(dw, ord=2) < self.tolerance and abs(db) < self.tolerance:
                break

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        y_predicted = np.dot(X, self.weights) + self.bias
        if self.y_columns is not None:
            y_predicted = super().Transform_From_Numpy_To_Pandas(
                y_predicted, self.y_columns
            )
        return y_predicted


class ElasticNet(RegressionModels):
    """
    Elastic Net Regression implémentée avec régularisation L1 et L2.
    """

    def __init__(
        self, alpha=1, l1_ratio=0.5, learning_rate=0.001, n_iters=1000, tolerance=1e-5
    ):
        self.weights = None
        self.bias = None
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.n_iters = n_iters
        self.l1_ratio = l1_ratio
        self.X_columns = None
        self.y_columns = None
        self.losses = []

    def _pretreat(self, X, y):
        X, X_columns = super().Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        y, y_columns = super().Transform_From_Pandas_To_Numpy(y)
        self.y_columns = y_columns
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        if len(y.shape) == 1:
            y = y.reshape(n_samples, 1)
        return X, y, n_samples, n_features

    def fit(self, X, y):
        X, y, n_samples, n_features = self._pretreat(X, y)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.alpha * (
                self.l1_ratio * np.sign(self.weights)
                + (1 - self.l1_ratio) * self.weights
            )
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.losses.append(0.5 * np.mean((y_predicted - y) ** 2))
            # Gestion explicite de weights = 0
            self.weights = np.where(
                np.abs(self.weights) < self.learning_rate * self.alpha, 0, self.weights
            )
            if np.linalg.norm(dw, ord=2) < self.tolerance and abs(db) < self.tolerance:
                break

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        y_predicted = np.dot(X, self.weights) + self.bias
        if self.y_columns is not None:
            y_predicted = super().Transform_From_Numpy_To_Pandas(
                y_predicted, self.y_columns
            )
        return y_predicted


class LogisticRegression(ClassificationModels):
    """
    Logistic Regression implémentée avec régularisation L2.

    """

    def __init__(self, alpha=1, learning_rate=0.001, n_iters=1000, tolerance=1e-4):
        self.weights = None
        self.bias = None
        self.alpha = alpha  # Coefficient de régularisation
        self.learning_rate = learning_rate  # Taux d'apprentissage
        self.tolerance = tolerance  # Critère d'arrêt
        self.n_iters = n_iters  # Nombre d'itérations
        self.X_columns = None
        self.y_columns = None
        self.losses = []  # Historique des pertes

    def _pretreat(self, X, y):
        return super()._pretreat(X, y)

    def _sigmoid(self, x):
        """
        Fonction sigmoïde.
        entrée :
        x : float
        sortie :
        float
        """
        return 1 / (1 + np.exp(-x + 1e-9))

    def fit(self, X, y):
        """
        Entraînement du modèle avec descente de gradient et régularisation L2.
        """
        X, y, n_samples, n_features = self._pretreat(X, y)

        # Initialisation des poids et du biais
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0

        for _ in range(self.n_iters):
            # Calcul des prédictions
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(z)

            # Calcul des gradients avec régularisation L2
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (
                self.alpha / n_samples
            ) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Mise à jour des poids et du biais
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calcul de la perte (fonction de coût) avec régularisation L2
            loss = -np.mean(
                y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted)
            ) + (self.alpha / (2 * n_samples)) * np.sum(self.weights**2)
            self.losses.append(loss)

            # Critère d'arrêt
            if np.linalg.norm(dw, ord=2) < self.tolerance and abs(db) < self.tolerance:
                break

    def predict(self, X):
        """
        Prédire les classes pour de nouvelles données.
        """
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)

        # Calcul des prédictions
        z = np.dot(X, self.weights) + self.bias

        y_pred = self._sigmoid(z)

        # Classification des prédictions (seuil à 0.5)
        y_pred_class = (y_pred >= 0.5).astype(int)

        if self.y_columns is not None:
            y_pred_class = super().Transform_From_Numpy_To_Pandas(
                y_pred_class, self.y_columns
            )

        return y_pred_class

    def plot_loss(self):
        """
        Fonction pour afficher la courbe de la perte au fil des itérations.
        """
        import matplotlib.pyplot as plt

        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel("Itérations")
        plt.ylabel("Perte")
        plt.title("Convergence de la régression logistique")
        plt.show()
