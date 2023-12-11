from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy.optimize import differential_evolution

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from multi_omics_integration.func import *

######## WEIGHT OPTIMIZATION


# Gradient Descent
def gradient_descent_ensemble(
    X, y, base_learners, learning_rate=0.01, max_iter=1000, init_weights=None
):
    """
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples, )
    base_learners: list of Scikit-learn regression models
    learning_rate: float
    max_iter: int
    init_weights: numpy array of shape (n_base_learners, )
    returns: numpy array of shape (n_base_learners, ) representing the learned weights
    """
    # Initialize the weights
    if init_weights is None:
        init_weights = np.zeros(len(base_learners))
    weights = init_weights

    # Concatenate the predictions of the base learners to form a new design matrix
    # TO DO: FIGURE OUT HOW TO FIX THIS
    X_ensemble = np.column_stack([bl.predict_proba(X) for bl in base_learners])

    # Gradient descent loop
    for i in range(max_iter):
        # Compute the gradient of the loss function with respect to the weights
        grad = np.dot(X_ensemble.T, (np.dot(X_ensemble, weights) - y)) / len(y)

        # Update the weights
        weights = weights - learning_rate * grad

    # Return the learned weights
    return weights


# Differential Evolution
def ensemble_objective_function(weights, X, y, base_learners):
    """
    weights: numpy array of shape (n_base_learners, )
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples, )
    base_learners: list of Scikit-learn regression models
    returns: float representing the value of the objective function
    """
    # Combine the predictions of the base learners using the provided weights
    y_ensemble = np.dot(
        np.column_stack([bl.predict(X) for bl in base_learners]), weights
    )

    # Compute the mean squared error
    mse = ((y_ensemble - y) ** 2).mean()

    return mse


def optimize_weights_with_de(X, y, base_learners, bounds, max_iter=1000, seed=None):
    """
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples, )
    base_learners: list of Scikit-learn regression models
    bounds: list of tuples specifying the lower and upper bounds for each weight
    max_iter: int
    seed: int or None
    returns: numpy array of shape (n_base_learners, ) representing the learned weights
    """
    # Set the random seed
    np.random.seed(seed)

    # Define the objective function for differential evolution
    objective_function = lambda weights: ensemble_objective_function(
        weights, X, y, base_learners
    )

    # Run differential evolution to optimize the weights
    result = differential_evolution(objective_function, bounds=bounds, maxiter=max_iter)

    # Return the best set of weights found by differential evolution
    return result.x


######## HYPERPARAMETER TUNING


def tune_params(X, y, estimator, params):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Define the grid search object
    grid = GridSearchCV(
        estimator=estimator, param_grid=params, cv=5, scoring="accuracy", n_jobs=-1
    )

    # Fit the grid search object on the training data
    grid.fit(X_train, y_train)

    # Get the best estimator and evaluate it on the test set
    best_model = grid.best_estimator_

    return best_model


def tune_ensemble_params(X, y, pipes, params):
    for name, model in pipes.items():
        model = tune_params(X, y, model, params[model])

    return pipes


def main():
    data = "data/breast/prot_processedDat.txt"
    target = "data/breast/TCGA_BRCA_subtypes.txt"

    estimators = [
        (
            "logistic",
            LogisticRegression(solver="liblinear", penalty="l2"),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "lasso",
            LogisticRegression(solver="liblinear", penalty="l1"),
            {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        ),
        (
            "elasticnet",
            SGDClassifier(loss="log_loss", penalty="elasticnet"),
            {
                "alpha": [0.0001, 0.001, 0.01],
                "l1_ratio": [0.15, 0.5, 0.85],
                "max_iter": [1000, 2000, 3000],
                "eta0": [0.01, 0.1, 1],
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=100, random_state=0),
            {"n_estimators": [10, 50, 100, 200], "max_depth": [3, 5, 7, None]},
        ),
        (
            "deep_nn",
            MLPClassifier(),
            {
                "hidden_layer_sizes": [
                    (10,),
                    (50,),
                    (100,),
                    (10, 10),
                    (50, 50),
                    (100, 100),
                ],
                "activation": ["identity", "logistic", "tanh", "relu"],
                "solver": ["lbfgs", "sgd", "adam"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
        ),
        # ('ridge',CalibratedClassifierCV(RidgeClassifier())),
        # ('svc', CalibratedClassifierCV(LinearSVC(dual=True, C=10))),
        # ('knn',KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='auto'))
    ]

    # Create the ensemble model
    ensemble = VotingClassifier(estimators=estimators, voting="hard")

    dataset, target, train_x, val_x, eval_x, train_y, val_y, eval_y = load_data(
        data, target
    )

    best_ensemble = tune_base_learners(dataset, target, estimators)

    # model = model.train_model()


if __name__ == "__main__":
    main()
