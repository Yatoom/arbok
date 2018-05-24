import numpy as np
from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import accuracy

from arbok.base import Wrapper
from arbok.out import say


class AutoSklearnWrapper(Wrapper):
    def __init__(self, preprocessor=None, refit=True, verbose=False, retry_on_error=True, **params):
        self.estimator = AutoSklearnClassifier(**dict(params))

        # Call to super
        super(AutoSklearnWrapper, self).__init__(estimator=self.estimator, preprocessor=preprocessor, refit=refit,
                                                 verbose=verbose,
                                                 retry_on_error=retry_on_error)

    def predict_proba(self, X):
        say("WARNING: predict_proba() not working well in Autosklearn. Raising AttributeError.")
        raise AttributeError()

    # Implementation of internal _fit
    def _fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params, metric=accuracy)

    # Implementation of internal _refit
    def _refit(self, X, y):
        self.estimator.fit(X, y)

    def _get_cv_results(self, estimator):
        # Get results and convert to lists, so that it is json serializable
        results = estimator.cv_results_
        lists = dict([(i, j if isinstance(j, list) else j.tolist()) for i, j in results.items()])

        # Store results
        cv_results_ = lists
        best_index_ = np.argmax(cv_results_['mean_test_score'])  # type: np.int64
        best_params_ = cv_results_['params'][best_index_]
        best_score_ = cv_results_['mean_test_score'][best_index_]

        return cv_results_, best_index_, best_params_, best_score_
