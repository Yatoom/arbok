import json
import os

import click
import openml
from click import ClickException
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from arbok import TPOTWrapper, AutoSklearnWrapper, ConditionalImputer


class Benchmark:

    def __init__(self, headers, python_interpreter, jobs_dir="jobs", config_file="config.json"):
        self.python_interpreter = python_interpreter
        self.jobs_dir = jobs_dir
        self.config_file = config_file
        self.headers = headers

        if not os.path.exists(jobs_dir):
            os.makedirs(jobs_dir)

    @staticmethod
    def create_config_file(file_name, tpot, autosklearn, wrapper):
        with open(file_name, "w+") as f:
            json.dump({
                "tpot": tpot,
                "autosklearn": autosklearn,
                "wrapper": wrapper,
            }, f, indent=4, sort_keys=True)

        return file_name

    @staticmethod
    def get_tasks_for_study(study_id):
        study = openml.study.get_study(study_id)
        return study.tasks

    def create_jobs(self, tasks, classifiers=None):
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
            # subprocess.call(["qsub", f"jobs/{file}"])
        return self

    def create_job(self, task_id, clf_name, preprocessor="default"):
        with open(f"jobs/{clf_name}_{task_id}.sh", "w+") as f:
            f.write(self.headers + "\n")
            # f.write(f"{self.python_interpreter} {self.project_root}/arbok/bench.py {clf_name} ")
            f.write(f"{self.python_interpreter} -m arbench ")
            f.write(f"--classifier {clf_name} ")
            f.write(f"--task-id {task_id} ")
            f.write(f"--config {self.config_file} ")
            f.write(f"--preprocessor {preprocessor} ")
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
    def run_job(clf_name, task_id, wrapper_config, tpot_config, autosklearn_config, preprocessor, apikey=None):

        if apikey:
            openml.config.apikey = apikey

        preprocessor = Benchmark.get_preprocessor(task_id, name=preprocessor)

        if clf_name == "tpot":
            clf = TPOTWrapper(preprocessor=preprocessor, **wrapper_config, **tpot_config)
        elif clf_name == "autosklearn":
            clf = AutoSklearnWrapper(preprocessor=preprocessor, **wrapper_config, **autosklearn_config)
        else:
            raise ValueError(f"Classifier name {clf_name} unknown")

        task = openml.tasks.get_task(task_id)
        run = openml.runs.run_model_on_task(task, clf)
        run.publish()

        print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))


@click.command()
@click.option('--classifier', default='tpot', help="Specify 'tpot' or 'autosklearn'.")
@click.option('--task-id', help="An id of an OpenMl task.")
@click.option('--config', default='config.json', help="A JSON configuration file for the classifiers and wrappers.")
@click.option('--preprocessor', default='default', help="Specify the preprocessor.")
@click.option('--apikey', default=None, help="Set the OpenML API Key which is required to upload the runs.")
def cli(classifier, task_id, config, preprocessor, apikey):
    if not task_id:
        raise ClickException("Please specify a task id.")
    elif not os.path.isfile(config):
        raise ClickException("The configuration file does not exist.")

    with open(config, "r") as f:
        cfg = json.load(f)

    tpot = cfg['tpot']
    autosklearn = cfg['autosklearn']
    wrapper = cfg['wrapper']

    print(f"Running {classifier} on task {task_id}.")
    Benchmark.run_job(classifier, task_id, wrapper, tpot, autosklearn, preprocessor, apikey=apikey)
