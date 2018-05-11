import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import unique_labels

from estimator import EstimatorWrapper

__version__ = 0.1


class AutoSklearnCV(BaseSearchCV):
    def __init__(self, estimator):
        self.classes__ = None
        self.param_distributions = {}
        super(AutoSklearnCV, self).__init__(estimator)

    @property
    def classes_(self):
        return self.classes__

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        print("Fitting wrapper")

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes__ = unique_labels(y)

        # Fit
        self.estimator.fit(X, y)

        # Get results
        results = self.estimator.cv_results_
        lists = dict([(i, j if isinstance(j, list) else j.tolist()) for i, j in results.items()])
        self.cv_results_ = lists
        self.best_index_ = np.argmax(self.cv_results_['mean_test_score'])
        self.best_params_ = self.cv_results_['params'][self.best_index_]
        self.best_score_ = self.cv_results_['mean_test_score'][self.best_index_]

        # Refit
        def refit(X, y):
            print("Refitting...")
            try:
                self.estimator.refit(X, y)
            except ValueError:
                self.estimator.fit(X, y)
                refit(X, y)

        refit(X, y)

        estimators = self.estimator.get_models_with_weights()
        self.best_estimator_ = EstimatorWrapper(self.estimator)
        self.best_estimator_.classes_ = self.classes__

        # # Store the only scorer not as a dict for single metric evaluation
        # super.scorer_ = scorers if self.multimetric_ else scorers['score']
        #
        # super.n_splits_ = n_splits

        return self

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.
        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        print("Predicting wrapper")
        self._check_is_fitted('predict')
        return self.estimator.predict(X)
