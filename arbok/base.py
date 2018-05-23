from warnings import warn

from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from arbok import out


class Wrapper(BaseSearchCV):
    def __init__(self, estimator, preprocessor=None, refit=True, verbose=False, retry_on_error=True):

        # Call to super
        super(Wrapper, self).__init__(self.estimator)

        self.retry_on_error = retry_on_error
        self.estimator = estimator
        self.verbose = verbose
        self.refit = refit
        self.preprocessor = preprocessor

        # Redirect openml's call on self.best_estimator_.classes_, to self.classes_
        self.best_estimator_ = self
        self.classes__ = None

        # Define parameters
        self.cv_results_ = None
        self.best_index_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.param_distributions = {}

    @property
    def classes_(self):
        return self.classes__

    def fit(self, X, y=None, groups=None, **fit_params):

        # Store the classes seen during fit
        self.classes__ = unique_labels(y)

        X_ = self.preprocessor.fit_transform(X) if self.preprocessor else X

        # Check that X and y have correct shape
        X, y = check_X_y(X_, y)

        try:
            if self.verbose:
                out.say("Wrapper is fitting.")

            # Fit the wrapped estimator
            self._fit(X_, y, **fit_params)

            # Store results
            cv_results_, best_index_, best_params_, best_score_ = self._get_cv_results(self.estimator)
            self.cv_results_ = cv_results_
            self.best_index_ = best_index_
            self.best_params_ = best_params_
            self.best_score_ = best_score_

            # Refit
            if self.refit:
                self._refit(X_, y)

        except ValueError as e:
            if self.retry_on_error:
                warn("Fitting failed. Attempting to fit again.")
                return self.fit(X_, y)
            raise e

        return self

    def get_params(self, deep=True):
        result = self.estimator.get_params(deep=deep)
        result['refit'] = self.refit
        result['verbose'] = self.verbose
        result['retry_on_error'] = self.retry_on_error
        result['preprocessor'] = self.preprocessor
        return result

    def set_params(self, **params):
        params = dict(self.get_params(), **params)
        self.refit = params.pop('refit')
        self.verbose = params.pop('verbose')
        self.retry_on_error = params.pop('retry_on_error')
        self.preprocessor = params.pop('preprocessor')
        self.estimator = self.estimator.set_params(**params)
        return self

    def predict(self, X):
        if self.verbose:
            out.say("Wrapper is predicting using predict().")

        # Check is fit had been called
        self._check_is_fitted('predict')

        X_ = self.preprocessor.transform(X) if self.preprocessor else X
        return self.estimator.predict(X_)

    def predict_proba(self, X):
        if self.verbose:
            out.say("Wrapper is predicting using predict_proba().")

        # Check is fit had been called
        self._check_is_fitted('predict_proba')

        X_ = self.preprocessor.transform(X) if self.preprocessor else X
        return self.estimator.predict_proba(X_)

    @staticmethod
    def _get_cv_results(estimator):
        return NotImplementedError()

    def _fit(self, X, y, **fit_params):
        return NotImplementedError()

    def _refit(self, X, y):
        return NotImplementedError()
