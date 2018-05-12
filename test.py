import openml
from autosklearn.estimators import AutoSklearnClassifier
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from wrappers import AutoSklearnWrapper
from wrappers.tpot import TpotWrapper

task = openml.tasks.get_task(31)

# clf = AutoSklearnWrapper(AutoSklearnClassifier(time_left_for_this_task=25, per_run_time_limit=5))
clf = TpotWrapper(generations=2, population_size=2, verbosity=2, refit=True, verbose=False)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))