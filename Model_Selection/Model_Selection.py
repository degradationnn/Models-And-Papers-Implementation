import numpy as np


class KFolds:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices, random_state=self.random_state)
        fold_sizes = n_samples // self.n_splits
        indices_folds = []
        for i in range(self.n_splits):
            start = i * fold_sizes
            stop = (i + 1) * fold_sizes
            if i == self.n_splits - 1:
                stop = n_samples
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            indices_folds.append((train_indices, test_indices))

        return indices_folds

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))



def r2_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def accuracy_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])

    

def learning_curve(
    X,
    y,
    model,
    loss,
    train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.0]),
    n_folds=5,
    shuffle=False,
    random_state=None,
):
    """
    Trace la courbe d'apprentissage en fonction de la taille de l'ensemble d'entraînement.

    Arguments:
    - X: Caractéristiques d'entrée (numpy array ou pandas DataFrame).
    - y: Cible (numpy array ou pandas Series).
    - model: Modèle à utiliser (doit implémenter fit() et predict()).
    - loss: Fonction de perte à utiliser (ex: sklearn.metrics.mean_squared_error).
    - train_sizes: Proportions d'entraînement à évaluer.
    - n_folds: Nombre de plis pour la validation croisée.
    - shuffle: Si True, les données sont mélangées avant la validation croisée.
    - random_state: Graine aléatoire pour la reproductibilité.

    Retourne:
    - train_sizes_abs: Liste des tailles absolues d'entraînement.
    - train_scores: Moyenne des pertes d'entraînement pour chaque taille.
    - test_scores: Moyenne des pertes de test pour chaque taille.
    """
    # Initialisation de la validation croisée
    kf = KFolds(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    train_scores = []
    test_scores = []
    train_sizes_abs = []


    # Calcul des scores pour chaque proportion d'entraînement
    for train_size in train_sizes:
        train_loss = 0
        test_loss = 0
        for train_indices, test_indices in kf.split(X):
            # Convertir la proportion en taille absolue
            n_train_samples = int(len(train_indices) * train_size)
            subset_indices = train_indices[:n_train_samples]  
            
            # Ajuster et prédire
            model.fit(X[subset_indices], y[subset_indices])
            y_train_pred = model.predict(X[subset_indices])
            y_test_pred = model.predict(X[test_indices])

            # Calcul des pertes
            train_loss += loss(y[subset_indices], y_train_pred)
            test_loss += loss(y[test_indices], y_test_pred)

        # Stocker les scores moyens pour cette taille
        train_sizes_abs.append(int(len(X) * train_size))
        train_scores.append(train_loss / n_folds)
        test_scores.append(test_loss / n_folds)

    return train_sizes_abs, train_scores, test_scores
