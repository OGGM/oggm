#######################
Developer documentation
#######################

.. currentmodule:: oggm


Workflow
========

Tools to set-up and run OGGM.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    workflow.init_glacier_regions
    workflow.execute_entity_task
    workflow.gis_prepro_tasks
    workflow.climate_tasks
    workflow.inversion_tasks


.. _apientitytasks:

Entity tasks
============

Entity tasks are tasks which are applied on single glaciers individually
and do not require information from other glaciers (this encompasses
the majority of OGGM's tasks). They are parallelizable.


.. autosummary::
    :toctree: generated/
    :nosignatures:

    tasks.define_glacier_region
    tasks.glacier_masks
    tasks.compute_centerlines
    tasks.initialize_flowlines
    tasks.compute_downstream_line
    tasks.compute_downstream_bedshape
    tasks.catchment_area
    tasks.catchment_intersections
    tasks.catchment_width_geom
    tasks.catchment_width_correction
    tasks.process_cru_data
    tasks.process_custom_climate_data
    tasks.process_cesm_data
    tasks.local_mustar
    tasks.apparent_mb
    tasks.apparent_mb_from_linear_mb
    tasks.mu_candidates
    tasks.prepare_for_inversion
    tasks.volume_inversion
    tasks.distribute_thickness
    tasks.init_present_time_glacier
    tasks.random_glacier_evolution
    tasks.iterative_initial_glacier_search
    tasks.run_from_climate_data
    tasks.run_constant_climate

Global tasks
============

Global tasks are tasks which are run on a set of glaciers (most often: all
glaciers in the current run). They are not parallelizable.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tasks.compute_ref_t_stars
    tasks.distribute_t_stars
    tasks.crossval_t_stars
    tasks.optimize_inversion_params

Classes
=======

Listed here are the classes which are relevant at the API level (i.e. classes
which are used and re-used across modules and tasks).

TODO: add the model classes, etc.



.. autosummary::
    :toctree: generated/
    :nosignatures:

    GlacierDirectory
    Centerline
    Flowline


Mass-balance
============

Mass-balance models found in the ``core.massbalance`` module.

.. currentmodule:: oggm.core.massbalance

They follow the :py:func:`MassBalanceModel` interface. Here is a quick summary
of the units and conventions used by all models:

Units
-----

The computed mass-balance is in units of [m ice s-1] ("meters of ice per
second"), unless otherwise specified (e.g. for the utility function
``get_specific_mb``).
The conversion from the climatic mass-balance ([kg m-2 s-1] ) therefore assumes
an ice density given by ``cfg.RHO`` (currently: 900 kg m-3).

Time
----

.. currentmodule:: oggm

The time system used by OGGM is a simple "fraction of year" system, where
the floating year can be used for conversion to months and years:

.. ipython:: python

    from oggm.utils import year_to_date, date_to_year
    date_to_year(1982, 12)
    year_to_date(1.2)


Interface
---------


.. currentmodule:: oggm.core.massbalance

.. autosummary::
    :toctree: generated/
    :nosignatures:

    MassBalanceModel
    MassBalanceModel.get_monthly_mb
    MassBalanceModel.get_annual_mb
    MassBalanceModel.get_specific_mb
    MassBalanceModel.temp_bias

Models
------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    LinearMassBalance
    PastMassBalance
    ConstantMassBalance
    RandomMassBalance
