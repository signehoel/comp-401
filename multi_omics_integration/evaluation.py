######### SCORING HELPER FUNCTIONS

import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import (
    f1_score,
    make_scorer,
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
import seaborn as sns
from tqdm import tqdm


def recall_multiclass(y_test, y_pred, label):
    return recall_score(y_test, y_pred, average=None)[label]


def get_cross_metrics(
    estimator,
    X,
    y,
    name="Model",
    cv=5,
    scoring={"f1": make_scorer(f1_score, average="weighted"), "accuracy": "accuracy"},
    n_jobs=-1,
    verbose=0,
    fit_params=None,
    return_train_score=False,
):
    metrics = {}
    train_metrics = {}

    for scorer in scoring:
        metrics[scorer] = []
        train_metrics[scorer] = []

    skf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=True)
    scores = cross_validate(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=skf,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        return_train_score=return_train_score,
        error_score="raise",
    )

    for scorer in scoring:
        metrics[scorer].append(
            "%0.3f +/- %0.3f"
            % (scores["test_%s" % scorer].mean(), scores["test_%s" % scorer].std())
        )
        if return_train_score:
            train_metrics[scorer].append(
                "%0.3f +/- %0.3f"
                % (
                    scores["train_%s" % scorer].mean(),
                    scores["train_%s" % scorer].std(),
                )
            )

    if return_train_score:
        df = pd.DataFrame(metrics, index=[name])
        train_df = pd.DataFrame(train_metrics, index=[f"{name} - train"])

        return scores, df, train_df

    df = pd.DataFrame(metrics, index=[name])
    return scores, df


def get_individual_scores(X, y, estimators, scoring, return_train_score=False):
    base_scores = []
    base_metrics_df = pd.DataFrame()
    base_metrics_df_train = pd.DataFrame()

    for name, clf in tqdm(estimators):
        if return_train_score:
            score, df, train_df = get_cross_metrics(
                clf, X, y, name, scoring=scoring, return_train_score=return_train_score
            )
            base_metrics_df_train = pd.concat([base_metrics_df_train, train_df])
        else:
            score, df = get_cross_metrics(
                clf, X, y, name, scoring=scoring, return_train_score=return_train_score
            )

        base_metrics_df = pd.concat([base_metrics_df, df])
        base_scores.append(score)

        print(name, df)

    if return_train_score:
        return base_scores, base_metrics_df, base_metrics_df_train
    else:
        return base_scores, base_metrics_df


def prediction_results(pipes, data):
    pred = pd.DataFrame()
    pred_proba = {}

    for name, pipe in pipes.items():
        pred[name] = pipe.predict(data)
        pred_proba[name] = pipe.predict_proba(data)

    return pred, pred_proba


def score_estimators(pipes, data, target, cv=3, metrics=["f1", "accuracy"]):
    metrics = {key: metrics[key] for key in metrics}
    scorers = []

    for name, pipe in pipes.items():
        kf = KFold(cv)
        model_scores = cross_validate(pipe, data, target, scoring=metrics, cv=kf)
        scorers.append(model_scores)

    return scorers


def plot_scores(estimator_names, scorers, metrics=["f1", "accuracy"]):
    score_lists = {}
    for metric in metrics:
        score_lists[metric] = [score["test_" + metric] for score in scorers]

    for i, (title, _list) in enumerate(score_lists.items()):
        plt.figure(i)
        sns.catplot(data=pd.DataFrame(_list).T, kind="box").set_xticklabels(
            estimator_names, rotation=45
        )
        plt.title(title)


def plot_box_scores(estimator_names, scorers, plot_title, metrics=["f1", "accuracy"]):
    score_lists = {}
    for metric in metrics:
        score_lists[metric] = [score["test_" + metric] for score in scorers]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, (title, _list) in enumerate(score_lists.items()):
        sns.boxplot(
            ax=axes[i],
            data=_list,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "8",
            },
            showfliers=False,
        )
        plt.title(title)
        axes[i].set_title(f"{plot_title} - {title}")
        axes[i].set_xlabel("Estimator")
        axes[i].set_ylabel("Score")
        axes[i].set_xticklabels(estimator_names, rotation=45)


def plot_bar_scores(estimator_names, scorers, plot_title, metrics=["f1", "accuracy"]):
    score_lists = {}
    for metric in metrics:
        score_lists[metric] = [score["test_" + metric] for score in scorers]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, (title, _list) in enumerate(score_lists.items()):
        sns.barplot(
            ax=axes[i],
            data=_list,
            palette="Blues_d",
            ci=None,
            orient="h",
            width=0.5,
            linewidth=2.5,
            dodge=False,
        )
        axes[i].set_title(f"{plot_title} - {title}")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Estimator")
        axes[i].set_yticks(ticks=range(len(_list)), labels=estimator_names)

    plt.tight_layout()
    plt.show()


def perf(pipes, eval_data, eval_target, results, cols_results):
    for pipe_name in pipes.keys():
        time_start = time.time()

        pipe = pipes[pipe_name]
        predictions = pipe.predict(eval_data)

        time_run = time.time() - time_start

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [[pipe_name, accuracy_score(eval_target, predictions), time_run]],
                    columns=cols_results,
                ),
            ],
            ignore_index=True,
        )

    return results
