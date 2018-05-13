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

Example usage
-------------

.. code:: python

    import openml
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from arbok import AutoSklearnWrapper, TPOTWrapper, ConditionalImputer


    task = openml.tasks.get_task(31)
    dataset = task.get_dataset()
    _, categorical, names = dataset.get_data(
        return_categorical_indicator=True, 
        return_attribute_names=True
    )
    mask = categorical[:-1]  # Remove last index (which is the class)

    # Optionally create a preprocessor that fixes missing data and one hot encodes 
    # categorical values.
    preprocessor = make_pipeline(

        # Imputer that uses different strategies for categorical and numerical data
        ConditionalImputer(
            categorical_features=mask
        ),
        OneHotEncoder(
            categorical_features=mask, handle_unknown="ignore", sparse=False
        )
    )

    # Get the AutoSklearn wrapper and pass parameters like you would to AutoSklearn
    clf = AutoSklearnWrapper(
        preprocessor=preprocessor, time_left_for_this_task=25, per_run_time_limit=5
    )

    # Or get the TPOT wrapper and pass parameters like you would to TPOT
    clf = TPOTWrapper(
        preprocessor=preprocessor, generations=2, population_size=2, verbosity=2
    )

    # Execute the task
    run = openml.runs.run_model_on_task(task, clf)
    run.publish()

    print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
