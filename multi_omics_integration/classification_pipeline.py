from multi_omics_integration.func import *


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
        estimator_list,
        feature_select=LogisticRegressionCV(penalty="l2", cv=3, random_state=1),
    ):
        # calculating scores
        cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        stacking = StackingClassifier(
            estimators=estimator_list,
            final_estimator=LogisticRegression(random_state=1),
            cv=cv,
        )
        # self.pipeline = Pipeline([('transformer', transformer), ('feature_selection', SelectFromModel(feature_select)), ('clf', stacking)])
        if feature_select != None:
            self.pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("feature_selection", SelectFromModel(feature_select)),
                    ("clf", stacking),
                ]
            )
        else:
            self.pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="mean")), ("clf", stacking)]
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


class EnsemblePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_list, X, y):
        self.preds = {}

        for modality, estimator in estimator_list:
            self.preds[modality] = estimator.cross_val_predict(X[modality], y)

        self.pipeline = LogisticRegressionCV(cv=3, random_state=1)

    def fit(self, X, y):
        return self.pipeline

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def cross_val_predict(self, X, y, method="predict"):
        return cross_val_predict(self.pipeline, X, y, method)

    def cross_validate(self, X, y, method="predict"):
        return cross_validate(self.pipeline, X, y, method)

    def get_params(self):
        return self.pipeline.get_params()

    def set_params(self, **params):
        return self.pipeline.set_params(params)


class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything.

        print("Shape of Pre-processed Data:", data.shape)
        #print(pd.DataFrame(data).head())
        return data

    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation.

        return self
