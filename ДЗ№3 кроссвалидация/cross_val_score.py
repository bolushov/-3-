from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define list of k values
k_values = [3, 5, 7, 9, 11]

# Define list of fold values
fold_values = [3, 5, 7, 10]

# Perform cross-validation for each k value and fold value
for k in k_values:
    for fold in fold_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        kf = KFold(n_splits=fold)
        scores = cross_val_score(knn, X, y, cv=kf)
        print(f'k={k}, Folds={fold}, Mean Accuracy: {np.mean(scores):.2f}, Standard Deviation: {np.std(scores):.2f}')
