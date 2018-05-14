import json
import os
import sys

import openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from arbok import TPOTWrapper, AutoSklearnWrapper, ConditionalImputer


class Benchmark:

    def __init__(self, config, python_interpreter, project_root, wrapper_config, tpot_config, autosklearn_config):
        self.config = config
        self.python_interpreter = python_interpreter
        self.project_root = project_root
        self.wrapper_config = wrapper_config
        self.tpot_config = tpot_config
        self.autosklearn_config = autosklearn_config

    @staticmethod
    def get_tasks_for_study(study_id):
        study = openml.study.get_study(study_id)
        return study.tasks

    def create_jobs_for_study(self, tasks, classifiers=None):
        if classifiers is None:
            classifiers = ["tpot", "autosklearn"]

        for clf in classifiers:
            for task_id in tasks:
                self.create_job(task_id, clf)

        return self

    def submit_jobs(self):
        files = [f for f in os.listdir("jobs/") if os.path.isfile(os.path.join("jobs/", f))]
        for file in files:
            print(f"Submitting jobs/{file}")
            os.subprocess.call(["qsub", f"jobs/{file}"])
        return self

    def create_job(self, task_id, clf_name, preprocessor="default"):
        if clf_name == "tpot":
            config = self.tpot_config
        elif clf_name == "autosklearn":
            config = self.autosklearn_config
        else:
            raise ValueError(f"Classifier name {clf_name} unknown")

        if not os.path.exists("jobs"):
            os.makedirs("jobs")

        with open(f"jobs/{clf_name}_{task_id}.sh", "w+") as f:
            f.write(self.config + "\n")
            f.write(f"{self.python_interpreter} {self.project_root}/arbok/bench.py {clf_name} ")
            f.write(f"{task_id} {json.dumps(self.wrapper_config)} {json.dumps(config)} {preprocessor}")
        return self

    @staticmethod
    def get_preprocessor(task_id, name):

        if name is None:
            return None

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        _, categorical = dataset.get_data(return_categorical_indicator=True)
        categorical = categorical[:-1]  # Remove last index (which is the class)

        if name == "default":

            preprocessor = make_pipeline(

                ConditionalImputer(
                    categorical_features=categorical,
                    strategy="mean",
                    strategy_nominal="most_frequent"
                ),

                OneHotEncoder(
                    categorical_features=categorical, handle_unknown="ignore", sparse=False
                )
            )
        else:
            raise ValueError(f"Preprocessor {name} unknown")

        return preprocessor

    @staticmethod
    def run_job(clf_name, task_id, wrapper_config, config, preprocessor):

        preprocessor = Benchmark.get_preprocessor(task_id, name=preprocessor)

        if clf_name == "tpot":
            clf = TPOTWrapper(preprocessor=preprocessor, **wrapper_config, **config)
        elif clf_name == "autosklearn":
            clf = AutoSklearnWrapper(preprocessor=preprocessor, **wrapper_config, **config)
        else:
            raise ValueError(f"Classifier name {clf_name} unknown")

        task = openml.tasks.get_task(task_id)
        run = openml.runs.run_model_on_task(task, clf)
        run.publish()

        print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))


if __name__ == "main":
    args = sys.argv
    Benchmark.run_job(args[1], args[2], json.loads(args[3]), json.loads(args[4]), args[5])
