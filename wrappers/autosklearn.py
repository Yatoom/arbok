from warnings import warn

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


class AutoSklearnWrapper(BaseSearchCV):
    def __init__(self, estimator, refit=True, verbose=False, retry_on_error=True):

        self.verbose = verbose
        self.refit = refit
        self.retry_on_error = retry_on_error

        # Redirect openml's call on self.best_estimator_.classes_, to self.classes_
        self.best_estimator_ = self
        self.classes__ = None

        # Define parameters
        self.cv_results_ = None
        self.best_index_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.param_distributions = {}

        # Call to super
        super(AutoSklearnWrapper, self).__init__(estimator)

    @property
    def classes_(self):
        return self.classes__

    def fit(self, X, y=None, groups=None, **fit_params):

        try:
            if self.verbose:
                print("AutoSklearnWrapper - fit")

            # Check that X and y have correct shape
            X, y = check_X_y(X, y)

            # Store the classes seen during fit
            self.classes__ = unique_labels(y)

            # Fit the wrapped estimator
            self.estimator.fit(X, y, **fit_params)

            # Get results and convert to lists, so that it is json serializable
            results = self.estimator.cv_results_
            lists = dict([(i, j if isinstance(j, list) else j.tolist()) for i, j in results.items()])

            # Store results
            self.cv_results_ = lists
            self.best_index_ = np.argmax(self.cv_results_['mean_test_score'])  # type: np.int64
            self.best_params_ = self.cv_results_['params'][self.best_index_]
            self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]

            if self.refit:
                self.estimator.refit(X, y)

        except ValueError as e:
            if self.retry_on_error:
                warn("Fitting failed. Attempting to fit again.")
                return self.fit(X, y)
            raise e

        return self

    def predict(self, X):
        if self.verbose:
            print("AutoSklearnWrapper - predict")

        # Check is fit had been called
        self._check_is_fitted('predict')

        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.verbose:
            print("AutoSklearnWrapper - predict_proba")

        # Check is fit had been called
        self._check_is_fitted('predict_proba')

        return self.estimator.predict_proba(X)
