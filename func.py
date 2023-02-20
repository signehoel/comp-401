from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold,cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score,roc_auc_score,log_loss
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pyreadr
import time

from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution

class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
    
class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))


####### PRE-PROCESSING

def split_data(data, target):    

    x, eval_data, y, eval_target = train_test_split(data, target, test_size=0.2, train_size=0.8)
    train_data, val_data, train_target, val_target = train_test_split(x, y, test_size = 0.25,train_size =0.75)

    return train_data, val_data, eval_data, train_target, val_target, eval_target

def load_data(dataset, subtypes):    

    subtypes = pd.read_csv(subtypes, sep='\t', index_col=0)        

    df = pd.read_csv(dataset, sep='\t')
    df = df.transpose()

    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header

    data = pd.concat([df, subtypes], axis=1, join="inner")

    data.columns = data.columns.str.strip()

    target = data['PAM50']
    target = correct_dtypes(target)

    data = data.drop(columns=['PAM50'])
    data = data.astype('float')

    x, eval_data, y, eval_target = train_test_split(data, target, test_size=0.2, train_size=0.8)
    train_data, val_data, train_target, val_target = train_test_split(x, y, test_size = 0.25,train_size =0.75)

    return data, target, train_data, val_data, eval_data, train_target, val_target, eval_target

def correct_dtypes(target):
    target=target.astype('category')
    target=target.map({'Basal-like':0,'HER2-enriched':1, 'Luminal A':2, 'Luminal B':3})

    return target

######### TRAINING

def train_models(estimators, data, target, transformer):
    pipes = {}

    for model in estimators:
        pipe=Pipeline(steps=[('data_prep',transformer),model])
        pipe.fit(data,target)
        pipes[pipe.steps[1][0]]=pipe
    
    return pipes
        
######### SCORING

def prediction_results(pipes, data):
    pred = pd.DataFrame()
    pred_proba = {}

    for name, pipe in pipes.items():
        pred[name] = pipe.predict(data)
        pred_proba[name] = pipe.predict_proba(data)
    
    return pred, pred_proba


def score_estimators(pipes, data, target, estimators, n_splits=5, metrics=['f1','auc','accuracy','logloss']):

    metrics={key : metrics[key] for key in metrics}
    scorers=[]
    labels=[]
    
    for pipe_name in pipes.keys():
        if pipe_name in estimators:
            pipe=pipes[pipe_name]
            labels.append(pipe_name)
            kf=KFold(n_splits)
            model_score=cross_validate(pipe,data,target,scoring=metrics,cv=kf)
            scorers.append(model_score)
             
    score_lists={}

    for metric in metrics:
        score_lists[metric]=[score['test_'+metric] for score in scorers]
        
    for  i,(title, _list) in enumerate(score_lists.items()):
        plt.figure(i)
        plot=sns.boxplot(data=_list).set_xticklabels(labels, rotation=45)
        plt.title(title)

def perf(pipes, eval_data, eval_target, results, cols_results):
    for pipe_name in pipes.keys():
        time_start = time.time()

        pipe=pipes[pipe_name]
        predictions = pipe.predict(eval_data)

        time_run = time.time()-time_start

        results = pd.concat([results, pd.DataFrame([[pipe_name,accuracy_score(eval_target,predictions),time_run]],columns=cols_results)],ignore_index=True)

    return results

######## PARAMETER TUNING

def tune_param(model,pipes,param_grid,refit, metrics, data, target, cv=5):
    
    param_grid={model+'__'+key : param_grid[key] for key in param_grid.keys()}

    xgbcv=GridSearchCV(pipes[model],param_grid,scoring=metrics,refit=refit,cv=cv)
    xgbcv.fit(data,target.values.ravel())

    print('best score: '+str(xgbcv.best_score_))
    print('best params: '+str(xgbcv.best_params_))
    results=pd.DataFrame(xgbcv.cv_results_)

######## PLOTTING
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
    summed = tensordot(yhats, weights, axes=((0),(0)))
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
    weights = [1.0/n_members for _ in range(n_members)]
    # define bounds on each weight
    bound_w = [(0.0, 1.0)  for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (members, val_target, predictions)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
    # get the chosen weights
    weights = normalize(result['x'])

    return weights

def main():

        data = 'data/breast/mRNA_processedDat.txt'
        target = 'data/breast/TCGA_BRCA_subtypes.txt'

        dataset, target = load_data(data, target)
        #model = model.train_model()

if __name__ == "__main__":
    main()
