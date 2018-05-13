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
-----------

-  Currently only the classifiers are implemented. Regression is
   therefore not possible.
-  For TPOT, the ``config_dict`` variable can not be set, because this
   causes problems with the API.
