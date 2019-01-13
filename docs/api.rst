#############
API Reference
#############

.. currentmodule:: oggm

This page lists all available functions and classes in OGGM. It is a hard
work to keep everything up-to-date, so don't hesitate to let us know
(see :ref:`contact`) if something's missing, or help us (see
:ref:`contributing`) to write a better documentation!


Workflow
========

Tools to set-up and run OGGM.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    cfg.initialize
    cfg.set_logging_config
    cfg.set_intersects_db
    cfg.reset_working_dir
    workflow.init_glacier_regions
    workflow.execute_entity_task
    workflow.gis_prepro_tasks
    workflow.climate_tasks
    workflow.inversion_tasks
    utils.compile_glacier_statistics
    utils.compile_run_output
    utils.compile_climate_input
    utils.compile_task_log
    utils.copy_to_basedir


Troubleshooting
===============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    utils.show_versions


Input/Output
============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    utils.get_demo_file
    utils.get_rgi_dir
    utils.get_rgi_region_file
    utils.get_rgi_glacier_entities
    utils.get_rgi_intersects_dir
    utils.get_rgi_intersects_region_file
    utils.get_rgi_intersects_entities
    utils.get_cru_file
    utils.get_histalp_file
    utils.get_ref_mb_glaciers


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
    tasks.process_histalp_data
    tasks.process_custom_climate_data
    tasks.process_gcm_data
    tasks.process_cesm_data
    tasks.process_cmip5_data
    tasks.local_t_star
    tasks.mu_star_calibration
    tasks.apparent_mb_from_linear_mb
    tasks.glacier_mu_candidates
    tasks.prepare_for_inversion
    tasks.mass_conservation_inversion
    tasks.filter_inversion_output
    tasks.distribute_thickness_per_altitude
    tasks.distribute_thickness_interp
    tasks.init_present_time_glacier
    tasks.run_random_climate
    tasks.run_constant_climate
    tasks.run_from_climate_data

Global tasks
============

Global tasks are tasks which are run on a set of glaciers (most often: all
glaciers in the current run). They are not parallelizable at the user level
but might use multiprocessing internally.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    tasks.compute_ref_t_stars
    tasks.compile_glacier_statistics
    tasks.compile_run_output
    tasks.compile_climate_input
    tasks.compile_task_log

Classes
=======

Listed here are the classes which are relevant at the API level (i.e. classes
which are used and re-used across modules and tasks).

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GlacierDirectory
    Centerline
    Flowline

.. _glacierdir:

Glacier directories
===================

Glacier directories (see also: :py:class:`~oggm.GlacierDirectory`) are folders
on disk which store the input and output data **for a single glacier** during
an OGGM run. The data are on disk to be persistent, i.e. they won't be deleted
unless you ask OGGM to. You can start a run from an existing directory,
avoiding to re-do unnecessary computations.

Initialising a glacier directory
--------------------------------

The easiest way to initialize a glacier directory is to start from a
pre-processed state available on the OGGM servers:


.. ipython:: python

    import os
    import oggm
    from oggm import cfg, workflow
    from oggm.utils import gettempdir
    cfg.initialize()  # always initialize before an OGGM task
    # The working directory is where OGGM will store the run's data
    cfg.PATHS['working_dir'] = os.path.join(gettempdir(), 'Docs_GlacierDir')
    gdirs = workflow.init_glacier_regions('RGI60-11.00897', from_prepro_level=1,
                                          prepro_border=10)
    gdir = gdirs[0]  # init_glacier_regions always returns a list


We just downloaded the minimal input for a glacier directory. The
``GlacierDirectory`` object contains the glacier metadata:

.. ipython:: python

    gdir
    gdir.rgi_id

It also points to a specific directory in disk, where the files are written
to:

.. ipython:: python

    gdir.dir
    os.listdir(gdir.dir)

Users usually don't have to care about *where* the data is located.
``GlacierDirectory`` objects help you to get this information:


.. ipython:: python

    fdem = gdir.get_filepath('dem')
    fdem
    import xarray as xr
    @savefig plot_gdir_dem.png width=80%
    xr.open_rasterio(fdem).plot(cmap='terrain')

This persistence on disk allows for example to continue a workflow that has
been previously interrupted. Initialising a GlacierDirectory from a non-empty
folder won't erase its content:

.. ipython:: python

    gdir = oggm.GlacierDirectory('RGI60-11.00897')
    os.listdir(gdir.dir)  # the directory still contains the data


.. include:: _generated/basenames.txt

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
an ice density given by ``cfg.PARAMS['ice_density']`` (currently: 900 kg m-3).

Time
----

.. currentmodule:: oggm

The time system used by OGGM is a simple "fraction of year" system, where
the floating year can be used for conversion to months and years:

.. ipython:: python

    from oggm.utils import floatyear_to_date, date_to_floatyear
    date_to_floatyear(1982, 12)
    floatyear_to_date(1.2)


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
    MassBalanceModel.get_ela

Models
------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    LinearMassBalance
    PastMassBalance
    ConstantMassBalance
    RandomMassBalance
    UncertainMassBalance
    MultipleFlowlineMassBalance
