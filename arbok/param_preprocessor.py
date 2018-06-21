import numbers
from collections import Iterable
from copy import copy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class ParamPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, types="detect", names=None):

        # List of types for each column
        self.types = types
        self.mapping = None
        self.one_hot_encoder = None
        self.names = names

    def fit(self, X):
        self.fit_transform(X, )
        return self

    def transform(self, X, y=None):
        X_ = np.array(X, copy=True, dtype=object)
        types = self.types
        mapping = self.mapping

        # Preprocessing steps
        X_, types, _ = ParamPreprocessor._remove_unsupported(X_, types, None)
        X_, types, _ = ParamPreprocessor._split_mixed(X_, types, None)
        X_ = ParamPreprocessor._nominal_to_numeric(X_, types, mapping)
        X_ = ParamPreprocessor._booleans_to_numeric(X_, types)
        X_ = ParamPreprocessor._fix_null(X_)
        X_ = self.one_hot_encoder.transform(X_)

        return X_

    def fit_transform(self, X, y=None, **kwargs):
        X_ = np.array(X, copy=True, dtype=object)
        types = self.types
        names = self.names

        if self.types == "detect":
            unique = ParamPreprocessor._get_unique(X_)
            types = self._detect_types(unique)
            self.types = types

        # Preprocessing steps
        X_, types, names = ParamPreprocessor._remove_unsupported(X_, types, names)
        X_, types, names = ParamPreprocessor._split_mixed(X_, types, names)
        distinct = ParamPreprocessor._get_unique(X_)
        mapping = ParamPreprocessor._create_mapping(distinct, types)
        X_ = ParamPreprocessor._nominal_to_numeric(X_, types, mapping)

        X_ = ParamPreprocessor._booleans_to_numeric(X_, types)
        X_ = ParamPreprocessor._fix_null(X_)

        # Save mapping
        self.mapping = mapping

        # Fit One Hot Encoder
        categorical_features = np.where(np.array(types) == "nominal")[0]
        n_values = [len(mapping[i]) for i in categorical_features]
        self.one_hot_encoder = OneHotEncoder(categorical_features=categorical_features, n_values=n_values,
                                             sparse=False)
        X_ = self.one_hot_encoder.fit_transform(X_)

        # Expand names according to one hot encoding and store result
        self.names = self.one_hot_encode_names(names, self.one_hot_encoder, mapping)

        return X_

    @staticmethod
    def one_hot_encode_names(names, encoder, mapping):

        # Check if names is None
        if names is None:
            return None

        # Get categorical features
        categorical_features = encoder.categorical_features

        # Convert mask to array of indices if needed
        if isinstance(categorical_features[0], bool):
            categorical_features = np.where(categorical_features)[0]

        # Convert to list
        categorical_features = list(categorical_features)

        # Split categorical and numeric names
        categorical_names = np.array(names)[categorical_features].tolist()
        numeric_names = [name for index, name in enumerate(names) if index not in categorical_features]

        # Get the categorical values from the mapping
        categorical_values = [list(i.keys()) for i in mapping.values()]

        # Add one hot encoded names
        result = []
        for index, name in enumerate(categorical_names):
            amount = encoder.n_values[index]
            for i in range(amount):
                value = categorical_values[index][i]
                result.append(f"{name}__{value}")

        return result + numeric_names

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
    def _remove_unsupported(X, types, names):
        indices = [index for index, value in enumerate(types) if
                   value not in ["mixed", "numeric", "bool", "nominal"]]
        # indices = np.where(np.array(types) not in ["mixed", "numeric", "bool", "nominal"])[0]
        new_X = np.delete(X, indices, 1)
        new_types = np.delete(types, indices, 0).tolist()

        # Update names
        new_names = None
        if names is not None:
            new_names = np.delete(names, indices).tolist()

        return new_X, new_types, new_names

    # Doesn't need fitting
    @staticmethod
    def _split_mixed(X, types, names):
        indices = np.where(np.array(types) == "mixed")[0]
        columns = np.array(X.T, copy=True, dtype=object)
        new_types = types
        new_names = copy(names)
        for i in indices:
            # Get numeric and nominal
            col = columns[i]
            col_num = np.array([item if isinstance(item, numbers.Number) else np.nan for item in col])
            col_nom = np.array([item if not isinstance(item, numbers.Number) else "<unkn>" for item in col])

            # Update one column, add the other as new column
            columns[i] = col_num
            columns = np.vstack([columns, col_nom])

            # Add types
            new_types[i] = "numeric"
            new_types.append("nominal")

            # Update names as well
            if names is not None:
                new_names[i] = f"{names[i]}__numeric"
                new_names.append(f"{names[i]}__nominal")

        new_X = columns.T

        return new_X, new_types, new_names

    # Needs a map created during fitting
    @staticmethod
    def _nominal_to_numeric(X, types, mapping):
        indices_nominal = np.where(np.array(types) == "nominal")[0]

        columns = np.array(X.T, copy=True, dtype=object)

        for i in indices_nominal:
            # Create a function that is applied to each item in a vector
            to_numeric = np.vectorize(lambda x: mapping[i][x] if x in mapping[i] else mapping[i]["<unkn>"])

            columns[i] = to_numeric(columns[i])

        return columns.T

    # Doesn't need fitting
    @staticmethod
    def _booleans_to_numeric(X, types):

        # Create a function that is applied to each item in a vector
        to_numeric = np.vectorize(lambda x: 0.5 if x is None else 0 if not x else 1)

        indices_bool = np.where(np.array(types) == "bool")[0]
        columns = np.array(X.T, copy=True, dtype=object)
        for i in indices_bool:
            columns[i] = to_numeric(columns[i])
        return columns.T

    @staticmethod
    def _create_mapping(unique, types):

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
        X_ = np.array([np.array(i, dtype=np.float64) for i in X], dtype=np.float64)
        return np.nan_to_num(X_)
