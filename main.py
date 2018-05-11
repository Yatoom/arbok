import openml
import autosklearn.classification
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

from wrapper import AutoSklearnCV

task = openml.tasks.get_task(31)

# Auto sklearn
askl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=20)
clf = AutoSklearnCV(askl)

# Run
run = openml.runs.run_model_on_task(task, clf)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
