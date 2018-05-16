Arbok
=====

Arbok (**A**\ utoml w\ **r**\ apper tool\ **b**\ ox for **o**\ penml
**c**\ ompatibility) provides wrappers for TPOT and Auto-Sklearn, as a
compatibility layer between these tools and OpenML.

The wrapper extends Sklearn’s ``BaseSearchCV`` and provides all the
internal parameters that OpenML needs, such as ``cv_results_``,
``best_index_``, ``best_params_``, ``best_score_`` and ``classes_``.

Installation
------------

::

    pip install arbok

Simple example
--------------

.. code:: python

    import openml
    from arbok import AutoSklearnWrapper, TPOTWrapper


    task = openml.tasks.get_task(31)
    dataset = task.get_dataset()

    # Get the AutoSklearn wrapper and pass parameters like you would to AutoSklearn
    clf = AutoSklearnWrapper(
        time_left_for_this_task=3600, per_run_time_limit=360
    )

    # Or get the TPOT wrapper and pass parameters like you would to TPOT
    clf = TPOTWrapper(
        generations=100, population_size=100, verbosity=2
    )

    # Execute the task
    run = openml.runs.run_model_on_task(task, clf)
    run.publish()

    print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

Preprocessing data
------------------

To make the wrapper more robust, we need to preprocess the data. We can
fill the missing values, and one-hot encode categorical data.

First, we get a mask that tells us whether a feature is a categorical
feature or not.

.. code:: python

    dataset = task.get_dataset()
    _, categorical = dataset.get_data(return_categorical_indicator=True)
    categorical = categorical[:-1]  # Remove last index (which is the class)

Next, we setup a pipeline for the preprocessing. We are using a
``ConditionalImputer``, which is an imputer which is able to use
different strategies for categorical (nominal) and numerical data.

.. code:: python

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from arbok import ConditionalImputer

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

And finally, we put everything together in one of the wrappers.

.. code:: python

    clf = AutoSklearnWrapper(
        preprocessor=preprocessor, time_left_for_this_task=3600, per_run_time_limit=360
    )

Limitations
~~~~~~~~~~~

-  Currently only the classifiers are implemented. Regression is
   therefore not possible.
-  For TPOT, the ``config_dict`` variable can not be set, because this
   causes problems with the API.

Benchmarking
------------

Installing the ``arbok`` package includes the ``arbench`` cli tool. We
can generate a json file like this:

.. code:: python

    from arbok.bench import Benchmark
    bench = Benchmark()
    config_file = bench.create_config_file(
           
        # Wrapper parameters
        wrapper={"refit": True, "verbose": False, "retry_on_error": True},
        
        # TPOT parameters
        tpot={
            "max_time_mins": 6,              # Max total time in minutes
            "max_eval_time_mins": 1          # Max time per candidate in minutes
        },
        
        # Autosklearn parameters
        autosklearn={
            "time_left_for_this_task": 360,  # Max total time in seconds
            "per_run_time_limit": 60         # Max time per candidate in seconds
        }
    )

And then, we can call arbench like this:

.. code:: bash

    arbench --classifier autosklearn --task-id 31 --config config.json

Or calling arbok as a python module:

.. code:: bash

    python -m arbok --classifier autosklearn --task-id 31 --config config.json

Running a benchmark on batch systems
------------------------------------

To run a large scale benchmark, we can create a configuration file like
above, and generate and submit jobs to a batch system as follows.

.. code:: python

    # We create a benchmark setup where we specify the headers, the interpreter we
    # want to use, the directory to where we store the jobs (.sh-files), and we give
    # it the config-file we created earlier.
    bench = Benchmark(
        headers="#PBS -lnodes=1:cpu3\n#PBS -lwalltime=1:30:00",
        python_interpreter="python3",  # Path to interpreter
        root="/path/to/project/",
        jobs_dir="jobs",
        config_file="config.json",
        log_file="log.json"
    )

    # Create the config file like we did in the section above
    config_file = bench.create_config_file(
           
        # Wrapper parameters
        wrapper={"refit": True, "verbose": False, "retry_on_error": True},
        
        # TPOT parameters
        tpot={
            "max_time_mins": 6,              # Max total time in minutes
            "max_eval_time_mins": 1          # Max time per candidate in minutes
        },
        
        # Autosklearn parameters
        autosklearn={
            "time_left_for_this_task": 360,  # Max total time in seconds
            "per_run_time_limit": 60         # Max time per candidate in seconds
        }
    )

    # Next, we load the tasks we want to benchmark on from OpenML.
    # In this case, we load a list of task id's from study 99.
    tasks = openml.study.get_study(99).tasks

    # Next, we create jobs for both tpot and autosklearn.
    bench.create_jobs(tasks, classifiers=["tpot", "autosklearn"])

    # And finally, we submit the jobs using qsub
    bench.submit_jobs()
