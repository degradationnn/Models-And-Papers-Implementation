# Models-And-Papers-Implementation

Ce repository contient les implémentations des modèles de machine learning et - dans le futur - de papiers de recherche en Deep Learning.

Le premier objectif est d'implémenter tous les modèles de machine learning classiques de scikit learn, ainsi que les fonctions utiles aux méthodologies d'apprentissage automatique, et de les tester sur des jeux de données classiques.

Le deuxième objectif est d'implémenter les modèles de Deep Learning tel que le MLP, CNN, LSTM, et de les tester sur des jeux de données classiques.

Finalement, le troisième objectif est d'implémenter les papiers de recherche en vogue pour comprendre profondément leur fonctionnement (Attention is All You Need, GAN, etc).

## 0 - Installation

Pour installer le package, il suffit de cloner le repository et d'installer les dépendances avec pip:

- git clone <https://github.com/degradationnn/Models-And-Papers-Implementation.git>

- cd Models-And-Papers-Implementation

- pip install -r requirements.txt

## 1 - Modèles de Machine Learning

Les modèles de machine learning sont implémentés dans le module `models` et sont organisés en sous-modules selon leur type (linéaire, arbre de décision, etc). Chaque sous-module contient les classes de modèle.

Chaque classe hérite de la classe `Model` qui contient les méthodes `fit` et `predict`.
On a de même créé des classes 'RegressionModel' et 'ClassificationModel' qui héritent de 'Model' et qui contiennent des méthodes spécifiques aux problèmes de régression et de classification. Ces classes implémentent également les méthodes `score` pour évaluer les performances du modèle.

- [x] Model

Classe de base pour les modèles de machine learning. Contient les méthodes `fit` et `predict`.

méthodes:

- `fit(X, y)`: Entraîne le modèle sur les données X et les labels y.
- `predict(X)`: Prédit les labels pour les données X.

- [x] RegressionModel

Classe de base pour les modèles de régression. Hérite de `Model`.

méthodes:

- `score(X, y)`: Retourne le coefficient de détermination R² des prédictions sur les données X et les labels y ainsi que la RMSE, la MAE et la MSE.

- [x] ClassificationModel

Classe de base pour les modèles de classification. Hérite de `Model`.

méthodes:

- `score(X, y)`: Retourne l'accuracy des prédictions sur les données X et les labels y.

### 1.1 - Modèles linéaires

- [x] Régression linéaire

Modèle de régression linéaire utilisant la résolution analytique des moindres carrés pour trouver les coefficients optimaux.  

- [x] Stochastic Gradient Descent regressor

Modèle de régression linéaire utilisant la descente de gradient stochastique pour trouver les coefficients optimaux.

Paramètres d'entrée :

    learning_rate (float): Taux d'apprentissage pour la descente de gradient.
    n_iters (int): Nombre d'itérations pour le processus d'entraînement.
    tolerance (float): Tolérance pour le critère d'arrêt.
    learning_rate_method (str): Méthode pour mettre à jour le taux d'apprentissage. Peut être 'constant' ou 'invscaling'.
    pow_t (float): Paramètre de puissance pour la méthode de taux d'apprentissage 'invscaling'.

- [x] Régression logistique

Modèle de classification binaire utilisant la régression logistique pour trouver les coefficients optimaux.

Paramètres d'entrée:

    learning_rate (float): Taux d'apprentissage pour la descente de gradient.
    n_iters (int): Nombre d'itérations pour le processus d'entraînement.
    tolerance (float): Tolérance pour le critère d'arrêt.
    alpha (float): Paramètre de régularisation L2.

- [x] Régression Ridge

Modèle de régression linéaire utilisant la régression Ridge pour trouver les coefficients optimaux. Méthode de régularisation L2. Calcul des coefficients optimaux par résolution analytique.

Paramètres d'entrée:

    alpha (float): Paramètre de régularisation L2.

- [x] Régression Lasso

Modèle de régression linéaire utilisant la régression Lasso pour trouver les coefficients optimaux. Méthode de régularisation L1.

Paramètres d'entrée:

    learning_rate (float): Taux d'apprentissage pour la descente de gradient.
    n_iters (int): Nombre d'itérations pour le processus d'entraînement.
    tolerance (float): Tolérance pour le critère d'arrêt.
    alpha (float): Paramètre de régularisation L1.

- [x] Régression ElasticNet

Modèle de régression linéaire utilisant la régression ElasticNet pour trouver les coefficients optimaux. Méthode de régularisation L1 et L2.

Paramètres d'entrée:

    learning_rate (float): Taux d'apprentissage pour la descente de gradient.
    n_iters (int): Nombre d'itérations pour le processus d'entraînement.
    tolerance (float): Tolérance pour le critère d'arrêt.
    alpha (float): Paramètre de régularisation L1.
    l1_ratio (float): Ratio de régularisation L1/L2.

### 1.2 - Modèles Plus Proches Voisins

- [x] KNN Regressor

Modèle de classification utilisant l'algorithme des K plus proches voisins pour prédire les labels. La regression est faite en prenant la moyenne des labels des K plus proches voisins.

Paramètres d'entrée:

    k (int): Nombre de voisins à considérer pour la prédiction.

- [x] KNN Classifier

Modèle de classification utilisant l'algorithme des K plus proches voisins pour prédire les labels. La classification est faite en prenant le label le plus fréquent parmi les K plus proches voisins.

Paramètres d'entrée:

    k (int): Nombre de voisins à considérer pour la prédiction.

### 2 - Fonctions et classes utiles

### 2.1 - Prepocessing

- [x] From_Pandas_to_Numpy

Fonction pour transformer un DataFrame Pandas en un array Numpy.

Paramètres d'entrée:

    df (DataFrame): DataFrame Pandas à transformer.

- [x] From_Numpy_to_Pandas

Fonction pour transformer un array Numpy en un DataFrame Pandas.

Paramètres d'entrée:

    X (array): Array Numpy à transformer.
    columns (list): Liste des noms des colonnes du DataFrame.

- [x] StandardScaler

Classe pour standardiser les données en soustrayant la moyenne et en divisant par l'écart-type.

méthodes:

- `fit(X)`: Calcule la moyenne et l'écart-type des données X.
- `transform(X)`: Standardise les données X.
- `fit_transform(X)`: Calcule la moyenne et l'écart-type des données X et standardise les données X.

- [x] MinMaxScaler

Classe pour normaliser les données en les mettant à l'échelle entre 0 et 1.

méthodes:

- `fit(X)`: Calcule les valeurs minimales et maximales des données X.
- `transform(X)`: Normalise les données X.
- `fit_transform(X)`: Calcule les valeurs minimales et maximales des données X et normalise les données X.

- [x] PolynomialFeatures

Classe pour générer des features polynomiales à partir des features existantes.

méthodes:

- `fit(X)`: Calcule les features polynomiales à partir des features existantes X.
- `transform(X)`: Génère les features polynomiales à partir des features existantes X.
- `fit_transform(X)`: Calcule les features polynomiales à partir des features existantes X et génère les features polynomiales.

Paramètres d'entrée:

    degree (int): Degré du polynôme.

- [x] train_test_split

Fonction pour diviser les données en un ensemble d'entraînement et un ensemble de test.

Paramètres d'entrée:

    X (array): Données à diviser.
    y (array): Labels à diviser.
    test_size (float): Taille de l'ensemble de test.
    random_state (int): Graine pour la génération de nombres aléatoires.

## Evaluation des modèles

- [x] accuracy_score, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

Fonctions calculant les métriques de regressions et classifications.

Paramètres d'entrée:

    y_true (array): Labels réels.
    y_pred (array): Labels prédits.

- [x] learning_curve

Fonction pour tracer les courbes d'apprentissage des modèles.

Paramètres d'entrée:

    model (Model): Modèle à évaluer.
    X (array): Données à utiliser pour l'évaluation.
    y (array): Labels à utiliser pour l'évaluation.
    train_sizes (array): Tailles des ensembles d'entraînement à tester.
    scoring (str): Métrique à utiliser pour l'évaluation.
    cv (int): Nombre de folds pour la validation croisée.
