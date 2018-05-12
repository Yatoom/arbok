import openml
from wrappers import AutoSklearnWrapper
from autosklearn.classification import AutoSklearnClassifier

task = openml.tasks.get_task(31)

# Instantiate an AutoSklearn Classifier like usual
autosklearn_clf = AutoSklearnClassifier(time_left_for_this_task=36, per_run_time_limit=12)

# Put the classifier inside the wrapper
clf = AutoSklearnWrapper(autosklearn_clf)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))