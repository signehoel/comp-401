from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    cross_val_predict,
    train_test_split,
    KFold,
    StratifiedKFold,
)
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import re
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from itertools import combinations
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from imblearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone

from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution
from progressbar import ProgressBar, Percentage, Bar


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


estimators = [
    ("logistic", LogisticRegression(solver="saga", multi_class="multinomial")),
    ("lasso", LogisticRegression(solver="liblinear", penalty="l1")),
    ("elasticnet", SGDClassifier(loss="log_loss", penalty="elasticnet")),
    ("random_forest", RandomForestClassifier(n_estimators=100, random_state=0)),
    ("deep_nn", MLPClassifier(max_iter=500, random_state=0)),
    ("xgb", XGBClassifier(objective="multi:softprob", enable_categorical=True)),
    ("svc", CalibratedClassifierCV(LinearSVC(dual=True, C=10))),
]

# Define the hyperparameters for each base learner
estimator_parameters = {
    "logistic": {"C": [0.01, 0.1, 1, 10, 100]},
    "lasso": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    "elasticnet": {
        "alpha": [0.0001, 0.001, 0.01],
        "l1_ratio": [0.15, 0.5, 0.85],
        "max_iter": [1000, 2000, 3000],
        "eta0": [0.01, 0.1, 1],
    },
    "random_forest": {"n_estimators": [10, 50, 100, 200], "max_depth": [3, 5, 7, None]},
    "deep_nn": {
        "hidden_layer_sizes": [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "invscaling", "adaptive"],
    },
}

estimator_names = [
    "logistic",
    "lasso",
    "elasticnet",
    "random_forest",
    "deep_nn",
    "ridge",
    "xgb",
    "svc",
    "knn",
]

######### TRAINING


def train_models(estimators, data, target, transformer):
    pipes = {}

    for model in estimators:
        pipe = Pipeline(steps=[("data_prep", transformer), model])
        pipe.fit(data, target)
        pipes[pipe.steps[1][0]] = pipe

    return pipes


# get a stacking ensemble of models
def trainStackingModel(pipes, final_estimator, data, target, metrics, cv):
    # Create the transformer to impute missing values
    imputer = SimpleImputer(strategy="mean")

    level1 = Pipeline(steps=[("prep", imputer), ("lr", final_estimator)])

    # calculating scores
    cv = KFold(n_splits=cv, random_state=1, shuffle=True)

    # define the stacking ensemble
    model = Pipeline(
        steps=[
            ("data_prep", transformer),
            (
                "stacking",
                StackingClassifier(estimators=pipes, final_estimator=level1, cv=5),
            ),
        ]
    )

    scores = cross_validate(
        model, data, target, scoring=metrics, cv=cv, n_jobs=-1, error_score="raise"
    )

    return model, scores


######## PARAMETER TUNING


def tune_param(model, pipes, param_grid, refit, metrics, data, target, cv=5):
    param_grid = {model + "__" + key: param_grid[key] for key in param_grid.keys()}

    xgbcv = GridSearchCV(pipes[model], param_grid, scoring=metrics, refit=refit, cv=cv)
    xgbcv.fit(data, target.values.ravel())

    print("best score: " + str(xgbcv.best_score_))
    print("best params: " + str(xgbcv.best_params_))
    results = pd.DataFrame(xgbcv.cv_results_)


####### WEIGHT OPTIMIZATION


# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, predictions):
    # make predictions
    yhats = [predictions[model] for model in members]
    yhats = array(yhats)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testy, predictions):
    # make prediction
    yhat = ensemble_predictions(members, weights, predictions)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testy, predictions):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(members, normalized, testy, predictions)


def find_weights(estimators, members, val_target, predictions):
    n_members = len(estimators)
    # evaluate averaging ensemble (equal weights)
    weights = [1.0 / n_members for _ in range(n_members)]
    # define bounds on each weight
    bound_w = [(0.0, 1.0) for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (members, val_target, predictions)
    # global optimization of ensemble weights
    result = differential_evolution(
        loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7
    )
    # get the chosen weights
    weights = normalize(result["x"])

    return weights


######## FEATURE SELECTION


class SequentialForwardSelection:

    """
    Instantiate with Estimator and given number of features
    """

    def __init__(self, estimator, k_features, scoring):
        self.estimator = clone(estimator)
        self.k_features = k_features

    """
    X_train - Training data Pandas dataframe
    X_test - Test data Pandas dataframe
    y_train - Training label Pandas dataframe
    y_test - Test data Pandas dataframe
    """

    def fit(self, X_train, X_test, y_train, y_test):
        max_indices = tuple(range(X_train.shape[1]))
        total_features_count = len(max_indices)
        self.subsets_ = []
        self.scores_ = []
        self.indices_ = []
        """
        Iterate through the feature space to find the first feature
        which gives the maximum model performance
        """
        scores = []
        subsets = []
        self.estimator.fit(X_train, y_train)

        pbar = ProgressBar()
        pbar = ProgressBar(
            widgets=[Percentage(), Bar()], maxval=X_train.shape[1]
        ).start()
        i = 0
        for p in combinations(max_indices, r=1):
            score = self._calc_score(X_test, y_test.values, p)
            scores.append(score)
            subsets.append(p)
            i += 1
            pbar.update(i)
        pbar.finish()
        #
        # Find the single feature having best score
        #
        best_score_index = np.argmax(scores)
        self.scores_.append(scores[best_score_index])
        self.indices_ = list(subsets[best_score_index])
        self.subsets_.append(self.indices_)

        #
        # Add a feature one by one until k_features is reached
        #
        dim = 1
        while dim < self.k_features:
            scores = []
            subsets = []
            current_feature = dim
            """
            Add the remaining features one-by-one from the remaining feature set
            Calculate the score for every feature combinations
            """
            idx = 0
            while idx < total_features_count:
                if idx not in self.indices_:
                    indices = list(self.indices_)
                    indices.append(idx)
                    score = self._calc_score(X_test, y_test.values, indices)
                    scores.append(score)
                    subsets.append(indices)
                idx += 1

            #
            # Get the index of best score
            #
            best_score_index = np.argmax(scores)
            #
            # Record the best score
            #
            self.scores_.append(scores[best_score_index])
            #
            # Get the indices of features which gave best score
            #
            self.indices_ = list(subsets[best_score_index])
            #
            # Record the indices of features for best score
            #
            self.subsets_.append(self.indices_)

            dim += 1

        self.k_score_ = self.scores_[-1]

    """
    Transform training, test data set to the data set
    havng features which gave best score
    """

    def transform(self, X):
        return X.values[:, self.indices_]

    """
    Train models with specific set of features
    indices - indices of features
    """

    def _calc_score(self, X_test, y_test, indices):
        X_test[X_test.columns.difference(indices, sort=False)] = 0

        y_pred = self.estimator.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return score


class SequentialBackwardSelection:
    """
    Instantiate with Estimator and given number of features
    """

    def __init__(self, estimator, k_features):
        self.estimator = clone(estimator)
        self.k_features = k_features

    """
    X_train - Training data Pandas dataframe
    X_test - Test data Pandas dataframe
    y_train - Training label Pandas dataframe
    y_test - Test data Pandas dataframe
    """

    def fit(self, X_train, X_test, y_train, y_test):
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        self.estimator.fit(X_train, y_train)

        score = self._calc_score(X_test, y_test.values, self.indices_)
        self.scores_ = [score]
        """
        Iterate through all the dimensions until k_features is reached
        At the end of loop, dimension count is reduced by 1
        """
        while dim > k_features:
            scores = []
            subsets = []
            """
            Iterate through different combinations of features, train the model,
            record the score
            """
            pbar = ProgressBar()
            pbar = ProgressBar(
                widgets=[Percentage(), Bar()], maxval=X_train.shape[1]
            ).start()
            i = 0
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_test, y_test.values, p)
                scores.append(score)
                subsets.append(p)
                i += 1
                pbar.update(i)
            pbar.finish()
            #
            # Get the index of best score
            #
            best_score_index = np.argmax(scores)
            #
            # Record the best score
            #
            self.scores_.append(scores[best_score_index])
            #
            # Get the indices of features which gave best score
            #
            self.indices_ = subsets[best_score_index]
            #
            # Record the indices of features for best score
            #
            self.subsets_.append(self.indices_)
            dim -= 1  # Dimension is reduced by 1

    """
    Transform training, test data set to the data set
    havng features which gave best score
    """

    def transform(self, X):
        return X.values[:, self.indices_]

    """
    Train models with specific set of features
    indices - indices of features
    """

    def _calc_score(self, X_test, y_test, indices):
        X_test[X_test.columns.difference(indices, sort=False)] = 0
        y_pred = self.estimator.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        return score


class FeatureSelection:
    """
    Instantiate with Estimator and given number of features
    """

    def __init__(self, estimator, cv):
        self.estimator = clone(estimator)
        self.cv = cv

    """
    X_train - Training data Pandas dataframe
    X_test - Test data Pandas dataframe
    y_train - Training label Pandas dataframe
    y_test - Test data Pandas dataframe
    """

    def fit(self, data, target):
        """
        Iterate through all the dimensions until k_features is reached
        At the end of loop, dimension count is reduced by 1
        """
        dim = data.shape[1]
        self.feature_importances_ = []
        self.rank_ = []

        cv = KFold(n_splits=self.cv, random_state=1, shuffle=True)

        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            importances = []

            self.estimator.fit(X, y)

            pred = self.estimator.predict(X_test)
            score = accuracy_score(y_test, pred)

            for i in tqdm(range(dim)):
                importance = self._calc_importance(X_test, y_test.values, score, i)
                importances.append(importance)

            self.feature_importances_.append(importances)

        self.avg_importances = np.average(self.feature_importances_, axis=0)
        array = np.array(self.avg_importances)
        self.rank_ = (-array).argsort()

    def best_n_features(self, k_features):
        best_indices = self.rank_[:k_features]

        return best_indices

    def get_scores(self, data, target, k_features):
        self.scores_ = []

        cv = KFold(n_splits=self.cv, random_state=None, shuffle=True)
        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            self.estimator.fit(X, y)
            scores = []

            for i in tqdm(range(k_features)):
                indices = self.best_n_features(i)
                temp = X_test.copy()
                all_cols = temp.columns
                selected_cols = np.delete(np.arange(len(all_cols)), indices)
                temp.iloc[:, selected_cols] = 0
                y_pred = self.estimator.predict(temp)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            self.scores_.append(scores)

        self.scores_ = np.average(self.scores_, axis=0)

        return self.scores_

    def plot_scores(self, k_features):
        x = list(range(0, k_features))
        plt.plot(x, self.scores_, marker=",")
        plt.ylim([0, 1])
        plt.ylabel("Accuracy")
        plt.xlabel("Number of features")
        plt.grid()
        plt.tight_layout()
        plt.show()

    """
    Transform training, test data set to the data set
    havng features which gave best score
    """

    def transform(self, X):
        return X.values[:, self.indices_]

    """
    Train models with specific set of features
    indices - indices of features
    """

    def _calc_importance(self, X_test, y_test, original_score, index):
        temp = X_test.copy()
        temp.iloc[:, index] = 0
        y_pred = self.estimator.predict(temp)
        score = accuracy_score(y_test, y_pred)
        importance = original_score - score

        return importance

    def _calc_importance_train(self, X_test, y_test, original_score, index):
        temp = X_test.copy()
        temp.iloc[:, index] = 0
        y_pred = self.estimator.predict(temp)
        score = accuracy_score(y_test, y_pred)
        importance = original_score - score

        return importance


class EnsembleFeatureSelection:
    """
    Instantiate with Estimator and given number of features
    """

    def __init__(self, estimators, cv):
        self.estimators = estimators
        # self.model = clone(model)
        self.cv = cv

    """
    X_train - Training data Pandas dataframe
    X_test - Test data Pandas dataframe
    y_train - Training label Pandas dataframe
    y_test - Test data Pandas dataframe
    """

    def fit(self, data, target):
        """
        Iterate through all the dimensions until k_features is reached
        At the end of loop, dimension count is reduced by 1
        """
        dim = data.shape[1]
        self.feature_importances_ = []
        self.rank_ = []

        cv = KFold(n_splits=self.cv, random_state=1, shuffle=True)

        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            importances = []

            for name, estimator in estimators:
                estimator.fit(X, y)

            # Create the transformer to impute missing values
            imputer = SimpleImputer(strategy="mean")
            level1 = Pipeline(steps=[("prep", imputer), ("lr", LogisticRegression())])

            self.model = StackingClassifier(
                estimators, final_estimator=level1, cv="prefit"
            )
            self.model.fit(X, y)

            pred = self.model.predict(X_test)
            score = accuracy_score(y_test, pred)

            for i in tqdm(range(dim)):
                importance = self._calc_importance(X_test, y_test.values, score, i)
                importances.append(importance)

            self.feature_importances_.append(importances)

        self.avg_importances = np.average(self.feature_importances_, axis=0)
        array = np.array(self.avg_importances)
        self.rank_ = (-array).argsort()

    def best_n_features(self, k_features):
        best_indices = self.rank_[:k_features]

        return best_indices

    def get_scores(self, data, target, k_features):
        self.scores_ = []

        cv = KFold(n_splits=self.cv, random_state=1, shuffle=True)
        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            for name, estimator in estimators:
                estimator.fit(X, y)

            # Create the transformer to impute missing values
            imputer = SimpleImputer(strategy="mean")
            level1 = Pipeline(steps=[("prep", imputer), ("lr", LogisticRegression())])

            self.model = StackingClassifier(
                estimators, final_estimator=level1, cv="prefit"
            )
            self.model.fit(X, y)

            scores = []

            for i in tqdm(range(k_features)):
                indices = self.best_n_features(i)
                temp = X_test.copy()
                all_cols = temp.columns
                selected_cols = np.delete(np.arange(len(all_cols)), indices)
                temp.iloc[:, selected_cols] = 0
                y_pred = self.model.predict(temp)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            self.scores_.append(scores)

        self.scores_ = np.average(self.scores_, axis=0)

        return self.scores_

    def plot_scores(self, k_features):
        x = list(range(0, k_features))
        plt.plot(x, self.scores_, marker=",")
        plt.ylim([0, 1])
        plt.ylabel("Accuracy")
        plt.xlabel("Number of features")
        plt.grid()
        plt.tight_layout()
        plt.show()

    """
    Transform training, test data set to the data set
    havng features which gave best score
    """

    def transform(self, X):
        return X.values[:, self.indices_]

    """
    Train models with specific set of features
    indices - indices of features
    """

    def _calc_importance(self, X_test, y_test, original_score, index):
        temp = X_test.copy()
        temp.iloc[:, index] = 0
        y_pred = self.model.predict(temp)
        score = accuracy_score(y_test, y_pred)
        importance = original_score - score

        return importance

    def _calc_importance_train(self, X_test, y_test, original_score, index):
        temp = X_test.copy()
        temp.iloc[:, index] = 0
        y_pred = self.model.predict(temp)
        score = accuracy_score(y_test, y_pred)
        importance = original_score - score

        return importance


class BaseFeatureSelection:
    """
    Instantiate with Estimator and given number of features
    """

    def __init__(self, estimator, cv):
        self.estimator = clone(estimator)
        self.cv = cv

    """
    X_train - Training data Pandas dataframe
    X_test - Test data Pandas dataframe
    y_train - Training label Pandas dataframe
    y_test - Test data Pandas dataframe
    """

    def fit(self, data, target):
        """
        Iterate through all the dimensions until k_features is reached
        At the end of loop, dimension count is reduced by 1
        """
        dim = data.shape[1]
        self.feature_importances_ = []
        self.rank_ = []

        cv = KFold(n_splits=self.cv, random_state=1, shuffle=True)

        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            importances = []

            self.estimator.fit(X, y)

            pred = self.estimator.predict(X_test)
            score = accuracy_score(y_test, pred)

            for i in tqdm(range(dim)):
                importance = self._calc_importance(X_test, y_test.values, score, i)
                importances.append(importance)

            self.feature_importances_.append(importances)

        self.avg_importances = np.average(self.feature_importances_, axis=0)
        array = np.array(self.avg_importances)
        self.rank_ = (-array).argsort()

    def best_n_features(self, k_features):
        best_indices = self.rank_[:k_features]

        return best_indices

    def get_scores(self, data, target, k_features):
        self.scores_ = []

        cv = KFold(n_splits=self.cv, random_state=1, shuffle=True)
        for train_index, test_index in cv.split(data):
            X, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y, y_test = target.iloc[train_index], target.iloc[test_index]

            self.estimator.fit(X, y)
            scores = []

            for i in tqdm(range(k_features)):
                indices = self.best_n_features(i)
                temp = X_test.copy()
                all_cols = temp.columns
                selected_cols = np.delete(np.arange(len(all_cols)), indices)
                temp.iloc[:, selected_cols] = 0
                y_pred = self.estimator.predict(temp)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)

            self.scores_.append(scores)

        self.scores_ = np.average(self.scores_, axis=0)

        return self.scores_

    def plot_scores(self, k_features):
        x = list(range(0, k_features))
        plt.plot(x, self.scores_, marker=",")
        plt.ylim([0, 1])
        plt.ylabel("Accuracy")
        plt.xlabel("Number of features")
        plt.grid()
        plt.tight_layout()
        plt.show()

    """
    Transform training, test data set to the data set
    havng features which gave best score
    """

    def transform(self, X):
        return X.values[:, self.indices_]

    """
    Train models with specific set of features
    indices - indices of features
    """

    def _calc_importance(self, X_test, y_test, original_score, index):
        temp = X_test.copy()
        temp.iloc[:, index] = 0
        y_pred = self.estimator.predict(temp)
        score = accuracy_score(y_test, y_pred)
        importance = original_score - score

        return importance


def sequential_forward_selection(
    estimator,
    X,
    y,
    k_features,
    forward=True,
    floating=False,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
):
    # Sequential Forward Selection
    sfs = SFS(
        estimator,
        k_features=k_features,
        forward=forward,
        floating=floating,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
    )

    sfs = sfs.fit(X, y)

    return sfs


######## PLOTTING
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt
