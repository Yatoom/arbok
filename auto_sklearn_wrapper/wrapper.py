from warnings import warn

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

__version__ = 0.1


class AutoSklearnWrapper(BaseSearchCV):
    def __init__(self, estimator, verbose=False):

        self.verbose = verbose

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

        if self.verbose:
            print("AutoSklearnCV auto_sklearn_wrapper - fit")

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

        # Refit and retry if error
        def refit(X_, y_):
            try:
                self.estimator.refit(X_, y_)
            except ValueError:
                warn("Refitting failed. Attempting to fit again and refit.")
                self.estimator.fit(X_, y_, **fit_params)
                refit(X_, y_)

        refit(X, y)

        return self

    def predict(self, X):
        if self.verbose:
            print("AutoSklearnCV auto_sklearn_wrapper - predict")

        # Check is fit had been called
        self._check_is_fitted('predict')

        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.verbose:
            print("AutoSklearnCV auto_sklearn_wrapper - predict_proba")

        # Check is fit had been called
        self._check_is_fitted('predict_proba')

        return self.estimator.predict_proba(X)
