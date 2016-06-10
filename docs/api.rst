#############
API reference
#############

Here we will add the documentation for selected modules.

.. currentmodule:: oggm.tasks

Entity tasks
============

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
    mu_candidates
    prepare_for_inversion
    volume_inversion
    distribute_thickness
    init_present_time_glacier

Global tasks
============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    distribute_climate_data
    compute_ref_t_stars
    distribute_t_stars
    optimize_inversion_params

.. currentmodule:: oggm

Classes
=======

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GlacierDirectory

.. currentmodule:: oggm.core.models.massbalance

Mass-balance
============

Mass-balance models in the ``oggm.core.models.massbalance`` module

.. autosummary::
    :toctree: generated/
    :nosignatures:

    TstarMassBalanceModel
    BackwardsMassBalanceModel
    TodayMassBalanceModel
    HistalpMassBalanceModel