# AutoSklearn wrapper
This is a wrapper that provides a compatibility layer between Auto-Sklearn and OpenML.

The wrapper extends Sklearn's `BaseSearchCV` and provides all the internal parameters that OpenML needs, such as 
`cv_results_`, `best_index_`, `best_params_`, `best_score_` and `classes_`.

## Installation
```
pip install auto-sklearn-wrapper
```

## Example usage
```python
import openml
from auto_sklearn_wrapper import AutoSklearnWrapper
from autosklearn.classification import AutoSklearnClassifier

task = openml.tasks.get_task(31)

# Instantiate an AutoSklearn Classifier like usual
autosklearn_clf = AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=360)

# Put the classifier inside the wrapper
clf = AutoSklearnWrapper(autosklearn_clf)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
```