import collections
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from tpot import TPOTClassifier
from tpot.config import classifier


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        k = k.split(".")[-1]
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def replace_last(s, old, new):
    return s[::-1].replace(old, new, 1)[::-1]


class TpotWrapper(BaseSearchCV):
    def __init__(self, verbose=False, refit=True, **params):
        # def __init__(self, generations=100, population_size=100, offspring_size=None,
        #              mutation_rate=0.9, crossover_rate=0.1,
        #              scoring=None, cv=5, subsample=1.0, n_jobs=1,
        #              max_time_mins=5, max_eval_time_mins=5,
        #              random_state=None, config_dict=None,
        #              warm_start=False, memory=None,
        #              periodic_checkpoint_folder=None, early_stop=None,
        #              verbosity=0, disable_update_check=False, verbose=False, refit=True):
        print(params['population_size'])
        self.estimator = TPOTClassifier(**params)
        # print(max_eval_time_mins)
        # self.estimator = None
        # self.set_params(generations=generations, population_size=population_size,
        #                 offspring_size=offspring_size,
        #                 mutation_rate=mutation_rate, crossover_rate=crossover_rate,
        #                 scoring=scoring, cv=cv, subsample=subsample, n_jobs=n_jobs,
        #                 max_time_mins=max_time_mins, max_eval_time_mins=max_eval_time_mins,
        #                 random_state=random_state, config_dict=config_dict,
        #                 warm_start=warm_start, memory=memory,
        #                 periodic_checkpoint_folder=periodic_checkpoint_folder, early_stop=early_stop,
        #                 verbosity=verbosity, disable_update_check=disable_update_check)

        self.verbose = verbose
        self.refit = refit

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
        super(TpotWrapper, self).__init__(self.estimator)

    def get_params(self, deep=True):
        result = self.estimator.get_params(deep=False)
        result['refit'] = self.refit
        result['verbose'] = self.verbose
        del result['config_dict']
        print("get params", result['population_size'])
        return result

    def set_params(self, **params):
        params = dict(self.estimator.get_params(deep=False), **params)
        self.estimator = self.estimator.set_params(**params)
        return self

    @property
    def classes_(self):
        return self.classes__

    def fit(self, X, y=None, groups=None, **fit_params):

        if self.verbose:
            print("TpotWrapper - fit")

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes__ = unique_labels(y)

        # Fit the wrapped estimator
        self.estimator.fit(X, y, **fit_params)

        # Get a dictionary of string representations of tested pipelines and their scores
        individuals = dict([(i, j['internal_cv_score']) for i, j in self.estimator.evaluated_individuals_.items()])

        # Extract the scores
        scores = list(individuals.values())

        # Select all parameters from representation as substrings
        param_strings = [re.findall("\w+=[\w0-9\.]*", i) for i in individuals.keys()]

        # Create a list of dictionaries that represent the parameter settings
        param_dicts = [dict([word.split("=") for word in strings]) for strings in param_strings]

        # Convert to dataframe, add prefix param_, replace np.nan with None and convert to list of dicts
        dataframe = pd.DataFrame(param_dicts)
        columns = list(dataframe.columns)

        # Load configuration from TPOT. We will use this to make sure all possible parameters are included.
        # We use a special flatten method, to make the parameters look like the ones in param_dicts.
        config = flatten(classifier.classifier_config_dict, parent_key="", sep="__")
        config_keys = list(config.keys())

        for i in config_keys:
            if i not in columns:
                dataframe[i] = np.nan

        # Add param_ prefixes
        dataframe = dataframe.add_prefix("param_")

        # Replace np.nan's with None's
        dataframe = dataframe.where(pd.notnull(dataframe), None)

        # Convert to dictionary of lists
        self.cv_results_ = dataframe.to_dict(orient="list")

        # Add mean test scores
        self.cv_results_['mean_test_score'] = scores

        self.best_index_ = np.argmax(scores)  # type: np.int64
        self.best_params_ = param_dicts[self.best_index_]
        self.best_score_ = scores[self.best_index_]

        # Fix for LinearSVC not having a method `fit_proba`.
        last_step = self.estimator.fitted_pipeline_.steps[-1][1]

        if isinstance(last_step, LinearSVC):
            self.estimator.fitted_pipeline_ = CalibratedClassifierCV(self.estimator.fitted_pipeline_)

        if self.refit:
            self.estimator.fitted_pipeline_.fit(X, y)

        return self

    def predict(self, X):
        if self.verbose:
            print("TpotWrapper - predict")

        # Check is fit had been called
        self._check_is_fitted('predict')

        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.verbose:
            print("TpotWrapper - predict_proba")

        # Check is fit had been called
        self._check_is_fitted('predict_proba')

        return self.estimator.predict_proba(X)
