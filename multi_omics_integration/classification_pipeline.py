from multi_omics_integration.func import *
from mlxtend.feature_selection import ColumnSelector


class BaseLearnerPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator,
        feature_select=LogisticRegressionCV(penalty="l2", cv=5, random_state=1),
    ):
        if feature_select != None:
            self.pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("feature_selection", SelectFromModel(feature_select)),
                    ("clf", estimator),
                ]
            )
        else:
            self.pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="mean")), ("clf", estimator)]
            )

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def cross_val_predict(self, X, y, method="predict", cv=5):
        return cross_val_predict(self.pipeline, X, y, method=method, cv=cv)

    def cross_validate(
        self,
        X,
        y,
        scoring={
            "f1": make_scorer(f1_score, average="weighted"),
            "accuracy": "accuracy",
        },
        cv=5,
    ):
        return cross_validate(self.pipeline, X, y, scoring=scoring, cv=cv)

    def get_params(self, deep=True):
        return self.pipeline.get_params()

    def set_params(self, **params):
        return self.pipeline.set_params(params)


class ModalityPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator_list=None,
        feature_select=None,
        final_estimator=LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1),
        cv=5,
        n_jobs=-1,
        stack_method="auto",
    ):
        self.cv = cv
        self.estimator_list = estimator_list
        self.feature_select = feature_select
        self.final_estimator = final_estimator
        self.n_jobs = n_jobs
        self.stack_method = stack_method

    def _get_model(self):
        self.cv_ = StratifiedKFold(n_splits=self.cv, random_state=0, shuffle=True)
        self.final_estimator_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("final", self.final_estimator),
            ]
        )

        stacking = StackingClassifier(
            estimators=self.estimator_list,
            final_estimator=self.final_estimator_,
            cv=self.cv_,
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
        )

        if self.feature_select != None:
            model = Pipeline(
                [("feature_selection", self.feature_select), ("clf", stacking)]
            )
        else:
            model = stacking

        return model

    def fit(self, X, y):
        self.model_ = self._get_model()
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.coef_ = self.model_.final_estimator_.named_steps["final"].coef_

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class MultiOmicsIntegrationClassifier(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator_dict=None,
        feature_select=None,
        final_estimator=LogisticRegression(random_state=0, n_jobs=-1),
        stack_method="auto",
        cv=5,
        n_jobs=-1,
    ):
        # calculating scores
        self.estimator_dict = estimator_dict
        self.feature_select = feature_select
        self.final_estimator = final_estimator
        self.stack_method = stack_method
        self.cv = cv
        self.n_jobs = n_jobs

    def _get_pipeline(self):
        self.classifiers_ = []
        self.cv_ = StratifiedKFold(n_splits=self.cv, random_state=0, shuffle=True)

        for dataset, columns in self.column_names_.items():
            stacking = Pipeline(
                [
                    ("column_selector", ColumnSelector(cols=columns)),
                    ("modality_clf", self.estimator_dict[dataset]),
                ]
            )
            self.classifiers_.append((dataset, stacking))

        self.final_estimator_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("final_estimator", self.final_estimator),
            ]
        )
        stacking = StackingClassifier(
            estimators=self.classifiers_,
            final_estimator=self.final_estimator_,
            cv=self.cv_,
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
        )

        return stacking

    def fit(self, X, y, column_names=None, **fit_params):

        self.column_names_ = column_names

        self.pipeline_ = self._get_pipeline()
        self.pipeline_.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)


class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything.

        print("Shape of Pre-processed Data:", data.shape)
        # print(pd.DataFrame(data).head())
        return data

    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation.

        return self
