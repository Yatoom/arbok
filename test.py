from copy import copy

import openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from arbok import AutoSklearnWrapper, TPOTWrapper
from arbok.preprocessing import ConditionalImputer

task = openml.tasks.get_task(31)
_, categorical, names = task.get_dataset().get_data(return_categorical_indicator=True, return_attribute_names=True)

preprocessor = make_pipeline(
    ConditionalImputer(categorical_features=categorical[:-1]),
    OneHotEncoder(categorical_features=categorical[:-1], handle_unknown="ignore", sparse=False)
)

# Get the AutoSklearn wrapper and pass parameters like you would to AutoSklearn
clf = AutoSklearnWrapper(preprocessor=preprocessor, time_left_for_this_task=26, per_run_time_limit=5, verbose=True)

# Or get the TPOT wrapper and pass parameters like you would to TPOT
# clf = TPOTWrapper(preprocessor=preprocessor, generations=2, population_size=2, verbosity=2, verbose=True)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()
