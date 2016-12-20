#############
API reference
#############

Here we will add the documentation for selected modules.

.. currentmodule:: oggm.tasks


.. _apientitytasks:

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
    process_custom_climate_data
    process_cru_data
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

    process_histalp_nonparallel
    compute_ref_t_stars
    distribute_t_stars
    crossval_t_stars
    optimize_inversion_params

.. currentmodule:: oggm

Classes
=======

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GlacierDirectory
    Centerline
    InversionFlowline


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
    PastMassBalanceModel