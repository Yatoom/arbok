import openml
from arbok import AutoSklearnWrapper, TPOTWrapper

task = openml.tasks.get_task(31)

# Get the AutoSklearn wrapper and pass parameters like you would to AutoSklearn
clf = AutoSklearnWrapper(time_left_for_this_task=26, per_run_time_limit=5, verbose=True)

# Or get the TPOT wrapper and pass parameters like you would to TPOT
# clf = TPOTWrapper(generations=2, population_size=2, verbosity=2)

# Execute the task
run = openml.runs.run_model_on_task(task, clf)
run.publish()