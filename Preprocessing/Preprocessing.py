import numpy as np
import pandas as pd
import random
from itertools import combinations_with_replacement

def Transform_From_Pandas_To_Numpy(X):
        columns = None
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            columns = X.columns
            X = X.values
        return X, columns
    
def Transform_From_Numpy_To_Pandas(X, columns):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=columns)
    return X
    


def Train_And_Test(X, y, split_ratio = 0.8, random_seed = None):
    """
    Here, we will split the data into training and testing data.

    Parameters:
    X: numpy array / pandas dataframe
    y: numpy array / pandas dataframe
    split_ratio: float
    random_seed: int
    """
    
    X, X_columns = Transform_From_Pandas_To_Numpy(X)
    y, y_columns = Transform_From_Pandas_To_Numpy(y)
    
    if random_seed is not None:
        random.seed(random_seed)  
    
    shuffled_list = random.sample(range(len(X)), len(X))
    X_Train = X[shuffled_list[:int(len(X) * split_ratio)]]
    X_Test = X[shuffled_list[int(len(X) * split_ratio):]]
    y_Train = y[shuffled_list[:int(len(X) * split_ratio)]]
    y_Test = y[shuffled_list[int(len(X) * split_ratio):]]

    #Que si c'était bien un dataframe à la base
    if X_columns is not None:
        X_Train = Transform_From_Numpy_To_Pandas(X_Train, columns = X_columns)
        X_Test = Transform_From_Numpy_To_Pandas(X_Test, columns = X_columns)

    if y_columns is not None:
        y_Train = Transform_From_Numpy_To_Pandas(y_Train, columns = y_columns)
        y_Test = Transform_From_Numpy_To_Pandas(y_Test, columns = y_columns)
    
    if X_columns is not None:
        X_Train.columns = X_columns
        X_Test.columns = X_columns
    if y_columns is not None:
        y_Train.columns = y_columns
        y_Test.columns = y_columns

    return X_Train, X_Test, y_Train, y_Test


class PolynomialFeatures:
    def __init__(self, degree = 2, include_bias = True):
        self.degree = degree
        if include_bias:
            self.include_bias = 1
        else:
            self.include_bias = 0

    def fit_transform(self, X):
        """
        Generate polynomial features for input X.
        entry:
        X: numpy array of shape (n_samples, 1)
       return:
        poly_features: numpy array of shape (n_samples, number of combinations with replacement)
        """
        X, X_columns = Transform_From_Pandas_To_Numpy(X)
        self.X_columns = X_columns
        
        if len(X.shape) > 1:
            n_samples, n_features = X.shape
        else:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(n_samples, n_features)
        combi = []
        for rep in range(self.degree + 1):
            combi += list(combinations_with_replacement(range(n_features), rep))
        poly_features = np.zeros((n_samples, len(combi) - 1 + self.include_bias))
        for i, pol in enumerate(combi):
            if pol == () and self.include_bias == 1:
                poly_features[:, i] = 1
            
            poly_features[:, i - 1] = X[:, combi[i]].prod(axis = 1)
        return poly_features



class Scaler:

    """
    Super classe des Scalers. Permet de garder une transformation pour pouvoir la réappliquer plus tard sur 
    de nouvelles données.
    """
    def __init__(self):
        pass
        
    def fit(self, X):
        """Méthode d'entraînement, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'fit' doit être implémentée dans la sous-classe.")

    def transform(self, X):
        """Méthode de prédiction, à surcharger dans les sous-classes."""
        raise NotImplementedError("La méthode 'predict' doit être implémentée dans la sous-classe.")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScaler(Scaler):

    """
    Standardisation des données normale; moyenne nulle et écart type égal à 1.
    """
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



class MinMaxScaler(Scaler):

    """
    Standardisation des données entre 0 et 1.
    """
    
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min




A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Poly = PolynomialFeatures(degree = 2, include_bias=False)
print(Poly.fit_transform(A))

def test_Transform_From_Pandas_To_Numpy():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    np_array, columns = Transform_From_Pandas_To_Numpy(df)
    assert isinstance(np_array, np.ndarray)
    assert columns.tolist() == ['A', 'B']

def test_Transform_From_Numpy_To_Pandas():
    np_array = np.array([[1, 4], [2, 5], [3, 6]])
    columns = ['A', 'B']
    df = Transform_From_Numpy_To_Pandas(np_array, columns)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == columns

def test_Train_And_Test():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    X_train, X_test, y_train, y_test = Train_And_Test(X, y, split_ratio=0.75, random_seed=42)
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1

