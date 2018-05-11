from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class EstimatorWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, acutal_estimator):
        self.actual_estimator = acutal_estimator

    def fit(self, X, y):
        print("Fitting estimator")
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.actual_estimator.fit(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        print("Predicting estimator")
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.actual_estimator.predict(X)

    def predict_proba(self, X):
        print("Predicting proba estimator")
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.actual_estimator.predict_proba(X)
