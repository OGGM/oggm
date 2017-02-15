#######################
Developer documentation
#######################


.. _apientitytasks:

Entity tasks
============

Entity tasks are tasks which are applied on single glaciers individually
and do not require information from other glaciers (this encompasses
the majority of OGGM's tasks). They are parallelizable.

.. currentmodule:: oggm.tasks

.. autosummary::
    :toctree: generated/
    :nosignatures:

    define_glacier_region
    glacier_masks
    compute_centerlines
    compute_downstream_lines
    catchment_area
    initialize_flowlines
    catchment_width_geom
    catchment_width_correction
    process_custom_climate_data
    process_cru_data
    mu_candidates
    prepare_for_inversion
    volume_inversion
    distribute_thickness
    init_present_time_glacier

Global tasks
============

Global tasks are tasks which are run on a set of glaciers (most often: all
glaciers in the current run). They are not parallelizable.

.. currentmodule:: oggm.tasks

.. autosummary::
    :toctree: generated/
    :nosignatures:

    compute_ref_t_stars
    distribute_t_stars
    crossval_t_stars
    optimize_inversion_params

Classes
=======

Listed here are the classes which are relevant at the API level (i.e. classes
which are used and re-used across modules and tasks).

TODO: add the flowline classes, the model classes, etc.

.. currentmodule:: oggm

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GlacierDirectory
    Centerline
    InversionFlowline


Mass-balance
============

.. currentmodule:: oggm

Mass-balance models found in the ``core.models.massbalance`` module.

.. currentmodule:: oggm.core.models.massbalance

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


.. currentmodule:: oggm.core.models.massbalance

.. autosummary::
    :toctree: generated/
    :nosignatures:

    MassBalanceModel
    MassBalanceModel.get_mb
    MassBalanceModel.get_annual_mb
    MassBalanceModel.get_specific_mb
    MassBalanceModel.temp_bias

Models
------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    LinearMassBalanceModel
    PastMassBalanceModel
    ConstantMassBalanceModel
    RandomMassBalanceModel
