import numbers
from collections import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ParamPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, types="detect"):

        # List of types for each column
        self.types = types
        self.mapping = None
        self.one_hot_encoder = None
        self.standard_scaler = None

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X, y=None):
        X_ = np.copy(X)
        types = self.types
        mapping = self.mapping

        # Preprocessing steps
        X_, types = ParamPreprocessor._remove_unsupported(X_, types)
        X_, types = ParamPreprocessor._split_mixed(X_, types)
        X_ = ParamPreprocessor._nominal_to_numerical(X_, types, mapping)
        X_ = ParamPreprocessor._booleans_to_numerical(X_, types)
        X_ = ParamPreprocessor._fix_null(X_)
        X_ = self.one_hot_encoder.transform(X_)
        X_ = self.standard_scaler.transform(X_)

        return X_

    def fit_transform(self, X, y=None):
        X_ = np.copy(X)
        types = self.types

        if self.types == "detect":
            unique = ParamPreprocessor._get_unique(X_)
            types = self._detect_types(unique)

        # Preprocessing steps
        X_, types = ParamPreprocessor._remove_unsupported(X_, types)
        X_, types = ParamPreprocessor._split_mixed(X_, types)
        distinct = ParamPreprocessor._get_unique(X_)
        mapping = ParamPreprocessor._create_mapping(distinct, types)
        X_ = ParamPreprocessor._nominal_to_numerical(X_, types, mapping)

        X_ = ParamPreprocessor._booleans_to_numerical(X_, types)
        X_ = ParamPreprocessor._fix_null(X_)

        # Save mapping
        self.mapping = mapping

        # Fit One Hot Encoder
        categorical_features = np.where(np.array(types) == "nominal")[0]
        n_values = [len(mapping[i]) for i in categorical_features]
        self.one_hot_encoder = OneHotEncoder(categorical_features=categorical_features, n_values=n_values,
                                             sparse=False)
        X_ = self.one_hot_encoder.fit_transform(X_)

        # Fit Standard Scaler
        self.standard_scaler = StandardScaler(with_mean=True)
        X_ = self.standard_scaler.fit_transform(X_)

        return X_

    @staticmethod
    def _detect_types(unique):
        types = []
        for values in unique:
            if any(isinstance(x, dict) for x in values):
                types.append("dict")
            elif any(not isinstance(x, str) and isinstance(x, Iterable) for x in values):
                types.append("iterable")
            elif any(isinstance(x, bool) for x in values):
                types.append("bool")
            elif all(x is None or isinstance(x, numbers.Number) for x in values):
                types.append("numeric")
            elif all(x is None or not isinstance(x, numbers.Number) for x in values):
                types.append("nominal")
            else:
                types.append("mixed")

        return types

    # Doesn't need fitting
    @staticmethod
    def _remove_unsupported(X, types):
        indices = [index for index, value in enumerate(types) if
                   value not in ["mixed", "numerical", "bool", "nominal"]]
        # indices = np.where(np.array(types) not in ["mixed", "numerical", "bool", "nominal"])[0]
        new_X = np.delete(X, indices, 1)
        new_types = np.delete(types, indices, 0).tolist()
        return new_X, new_types

    # Doesn't need fitting
    @staticmethod
    def _split_mixed(X, types):
        indices = np.where(np.array(types) == "mixed")[0]
        columns = np.copy(X.T)
        new_types = types
        for i in indices:
            # Get numerical and nominal
            col = columns[i]
            col_num = np.array([item if isinstance(item, numbers.Number) else np.nan for item in col])
            col_nom = np.array([item if not isinstance(item, numbers.Number) else "<unkn>" for item in col])

            # Update one column, add the other as new column
            columns[i] = col_num
            columns = np.vstack([columns, col_nom])

            # Add types
            new_types[i] = "numerical"
            new_types.append("nominal")

        new_X = columns.T

        return new_X, new_types

    # Needs a map created during fitting
    @staticmethod
    def _nominal_to_numerical(X, types, mapping):
        indices_nominal = np.where(np.array(types) == "nominal")[0]

        columns = np.copy(X.T)

        for i in indices_nominal:
            # Create a function that is applied to each item in a vector
            to_numerical = np.vectorize(lambda x: mapping[i][x] if x in mapping[i] else mapping[i]["<unkn>"])

            columns[i] = to_numerical(columns[i])

        return columns.T

    # Doesn't need fitting
    @staticmethod
    def _booleans_to_numerical(X, types):

        # Create a function that is applied to each item in a vector
        to_numerical = np.vectorize(lambda x: 0.5 if x is None else 0 if not x else 1)

        indices_bool = np.where(np.array(types) == "bool")[0]
        columns = np.copy(X.T)
        for i in indices_bool:
            columns[i] = to_numerical(columns[i])
        return columns.T

    @staticmethod
    def _create_mapping(unique, types):
        # Create a function that is applied to each item in a vector
        # replace_non_nominal = np.vectorize(lambda x: x if isinstance(x, str) else "<unkn>")

        # unique = ParamPreprocessor._get_unique(unique)

        indices_nominal = np.where(np.array(types) == "nominal")[0]
        result = dict([(i, {}) for i in indices_nominal])

        for i in indices_nominal:
            possible_values = ["<unkn>"] + sorted([i for i in unique[i] if isinstance(i, str) and i != "<unkn>"])
            result[i] = dict(zip(possible_values, range(len(possible_values))))

        return result

    @staticmethod
    def _get_unique(X):
        result = []
        for column in X.T:
            result.append(list(set(column)))
        return result

    @staticmethod
    def _fix_null(X):

        # Create a function that is applied to each item in a vector
        to_nan = np.vectorize(lambda x: x if x is not None else np.nan)

        X_ = np.array([to_nan(i) for i in X])

        return np.nan_to_num(X_)
