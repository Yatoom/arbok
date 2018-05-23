import collections
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from tpot import TPOTClassifier
from tpot.config import classifier

from arbok.base import Wrapper


class TPOTWrapper(Wrapper):
    def __init__(self, preprocessor=None, refit=True, verbose=False, retry_on_error=True, **params):

        # Create estimator
        self.estimator = TPOTClassifier(**params)

        # Call to super
        super(TPOTWrapper, self).__init__(estimator=self.estimator, preprocessor=preprocessor, refit=refit,
                                          verbose=verbose,
                                          retry_on_error=retry_on_error)

    # Override get_params to not include config_dict
    def get_params(self, deep=True):
        result = super(TPOTWrapper, self).get_params(deep=deep)
        result.pop('config_dict')
        return result

    # Implementation of internal _fit
    def _fit(self, X, y, **fit_params):
        # Fit the wrapped estimator
        self.estimator.fit(X, y, **fit_params)

        # Fix for LinearSVC not having a method `fit_proba`.
        last_step = self.estimator.fitted_pipeline_.steps[-1][1]

        if isinstance(last_step, LinearSVC):
            self.estimator.fitted_pipeline_ = CalibratedClassifierCV(self.estimator.fitted_pipeline_)

    # Implementation of internal _refit
    def _refit(self, X, y):
        self.estimator.fitted_pipeline_.fit(X, y)

    # Implementation of get internal _get_cv_results
    @staticmethod
    def _get_cv_results(estimator):
        # Get a dictionary of string representations of tested pipelines and their scores
        individuals = dict([(i, j['internal_cv_score']) for i, j in estimator.evaluated_individuals_.items()])

        # Select all parameters from representation as substrings
        param_strings = [re.findall("\w+=[\w0-9\.]*", i) for i in individuals.keys()]

        # Create a list of dictionaries that represent the parameter settings
        param_dicts = [dict([word.split("=") for word in strings]) for strings in param_strings]

        # Convert to dataframe, add prefix param_, replace np.nan with None and convert to list of dicts
        dataframe = pd.DataFrame(param_dicts)
        columns = list(dataframe.columns)

        # Load configuration from TPOT. We will use this to make sure all possible parameters are included.
        # We use a special flatten method, to make the parameters look like the ones in param_dicts.
        config = TPOTWrapper._flatten(classifier.classifier_config_dict, parent_key="", sep="__")
        config_keys = list(config.keys())

        # Instead of, for example, RFE__estimator__ExtratreesClassifier__n_estimators,
        # TPOT uses RFE__ExtratreesClassifier__n_estimators, so we need to replace those.
        config_keys = [i.replace("__estimator__", "__") for i in config_keys]

        # Show a warning if we're adding new keys.
        unknown_keys = [i for i in columns if i not in config_keys]
        if len(unknown_keys) > 0:
            warnings.warn(f"Unknown keys: {unknown_keys}")

        for i in config_keys:
            if i not in columns:
                dataframe[i] = np.nan

        # Add param_ prefixes
        dataframe = dataframe.add_prefix("param_")

        # Replace np.nan's with None's
        dataframe = dataframe.where(pd.notnull(dataframe), None)

        # Convert to dictionary of lists
        cv_results_ = dataframe.to_dict(orient="list")

        # Add the mean test score
        scores = list(individuals.values())
        scores = [max(0, i) for i in scores]  # Remove -inf's as OpenML can't deal with those
        cv_results_['mean_test_score'] = scores

        best_index_ = np.argmax(scores)  # type: np.int64
        best_params_ = param_dicts[best_index_]
        best_score_ = scores[best_index_]

        return cv_results_, best_index_, best_params_, best_score_

    @staticmethod
    def _flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            k = k.split(".")[-1]
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(TPOTWrapper._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
