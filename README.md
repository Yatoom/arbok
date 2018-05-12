# Arbok

Arbok (**A**utoml w**r**apper tool**b**ox for **o**penml **c**ompatibility) provides wrappers 
for TPOT and Auto-Sklearn, as a compatibility layer between these tools and OpenML.

The wrapper extends Sklearn's `BaseSearchCV` and provides all the internal parameters that OpenML needs, such as 
`cv_results_`, `best_index_`, `best_params_`, `best_score_` and `classes_`.

## Installation
```
pip install arbok
```

## Example usage
```python
import openml
from arbok import AutoSklearnWrapper, TPOTWrapper

task = openml.tasks.get_task(31)

# Get the AutoSklearn wrapper and pass parameters like you would to AutoSklearn
clf = AutoSklearnWrapper(time_left_for_this_task=25, per_run_time_limit=5)

# Or get the TPOT wrapper and pass parameters like you would to TPOT
clf = TPOTWrapper(generations=2, population_size=2, verbosity=2)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
```