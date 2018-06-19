import numbers
from collections import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ParamPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, types="detect", scaler=StandardScaler(), names=None):

        # List of types for each column
        self.types = types
        self.mapping = None
        self.one_hot_encoder = None
        self.scaler = scaler
        self.names = names

    def fit(self, X):
        self.fit_transform(X, )
        return self

    def transform(self, X, y=None):
        X_ = np.copy(X)
        types = self.types
        mapping = self.mapping

        # Preprocessing steps
        X_, types, _ = ParamPreprocessor._remove_unsupported(X_, types, None)
        X_, types, _ = ParamPreprocessor._split_mixed(X_, types, None)
        X_ = ParamPreprocessor._nominal_to_numerical(X_, types, mapping)
        X_ = ParamPreprocessor._booleans_to_numerical(X_, types)
        X_ = ParamPreprocessor._fix_null(X_)
        X_ = self.one_hot_encoder.transform(X_)
        X_ = self.scaler.transform(X_)

        return X_

    def fit_transform(self, X, y=None, **kwargs):
        X_ = np.copy(X)
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

        # Expand names according to one hot encoding and store result
        self.names = self.one_hot_encode_names(categorical_features, self.one_hot_encoder.n_values, names)

        # Fit scaler
        X_ = self.scaler.fit_transform(X_)

        return X_

    @staticmethod
    def one_hot_encode_names(categorical_indices, categorical_n_values, names):

        # Check if names is None
        if names is None:
            return None

        # Convert mask to array of indices if needed
        if isinstance(categorical_indices[0], bool):
            categorical_indices = np.where(categorical_indices)[0]

        # Convert to list
        categorical_indices = list(categorical_indices)

        result = []
        for index, name in enumerate(names):
            if index in categorical_indices:
                amount = categorical_n_values[categorical_indices.index(index)]

                for i in range(amount):
                    result.append(f"{name}__{i}")
            else:
                result.append(name)

        return result

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
                   value not in ["mixed", "numerical", "bool", "nominal"]]
        # indices = np.where(np.array(types) not in ["mixed", "numerical", "bool", "nominal"])[0]
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
        columns = np.copy(X.T)
        new_types = types
        new_names = names
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

            # Update names as well
            if names is not None:
                new_names[i] = f"{names[i]}__numerical"
                new_names.append(f"{names[i]}__nominal")

        new_X = columns.T

        return new_X, new_types, new_names

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
        X_ = np.array([np.array(i, dtype=np.float64) for i in X], dtype=np.float64)
        return np.nan_to_num(X_)
