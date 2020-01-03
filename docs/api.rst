#############
API Reference
#############

.. currentmodule:: oggm

This page lists all available functions and classes in OGGM. It is a hard
work to keep everything up-to-date, so don't hesitate to let us know
(see :ref:`contact`) if something's missing, or help us (see
:ref:`contributing`) to write a better documentation!

.. _api-workflow:

Workflow
========

Tools to set-up and run OGGM.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    cfg.initialize
    cfg.initialize_minimal
    cfg.set_logging_config
    cfg.set_intersects_db
    cfg.reset_working_dir
    workflow.init_glacier_regions
    workflow.execute_entity_task
    workflow.gis_prepro_tasks
    workflow.climate_tasks
    workflow.inversion_tasks
    workflow.merge_glacier_tasks
    utils.compile_glacier_statistics
    utils.compile_run_output
    utils.compile_climate_input
    utils.compile_task_log
    utils.compile_task_time
    utils.copy_to_basedir
    utils.gdir_to_tar
    utils.base_dir_to_tar


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
    utils.write_centerlines_to_shape


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
    tasks.compile_task_time

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

They follow the :py:class:`MassBalanceModel` interface. Here is a quick summary
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

Model Flowlines
===============

Flowlines are what represent a glacier in OGGM.
During the preprocessing stage of a glacier simulation these lines evolve from
simple topography following lines to more abstract objects within the model.

This chapter provides an overview on the different line types, how to access
the model flowlines and their attributes.

Hierarchy
---------

First a short summary of the evolution or the hierarchy of these different
flowline stages:

.. currentmodule:: oggm.core.centerlines

geometrical centerlines
   These lines are calculated following the algorithm of
   `Kienholz et al., (2014)`_. They determine where the subsequent flowlines
   are situated and how many of them are needed for one glacier. The main task
   for this calculation is :py:func:`compute_centerlines`. These lines are not
   stored within a specific OGGM class but are stored within the
   GlacierDirectory as shapely objects.

inversion flowlines
   To use the geometrical lines within the model they are transformed to
   :py:class:`~oggm.Centerline` objects. This is done within
   :py:func:`~oggm.core.centerlines.initialize_flowlines` where the lines are projected onto a xy-grid
   with regular spaced line-points. This step also takes care of tributary
   junctions. These lines are then in a later step also used for the bed
   inversion, hence their name.

downstream line
    This line extends the glacier's centerline along the unglaciated glacier
    bed which is necessary for advancing glaciers. This line is calculated
    along the valley floor to the end of the domain boundary within
    :py:func:`compute_downstream_line`.

.. currentmodule:: oggm.core.flowline

model flowlines
   The dynamical OGGM model finally uses lines of the class
   :py:class:`Flowline` for all simulations. These flowlines are created by
   :py:func:`init_present_time_glacier`, which combines information from
   several preprocessing steps, the downstream line and the bed inversion.
   From a user prespective, especially if preprocessed directories are used,
   these model flowlines are the most importent ones and further informations
   on the class interface and attributes are given below.

.. _Kienholz et al., (2014): http://www.the-cryosphere.net/8/503/2014/

Access
------

The easiest and most common way to access the model flowlines of a glacier is
with its GlacierDirectory. For this we initialize a minimal working example
including the model flowlines. This is achieved by choosing preprocessing level
4.

.. ipython:: python

    import os
    import oggm
    from oggm import cfg, workflow
    from oggm.utils import gettempdir
    cfg.initialize()  # always initialize before an OGGM task
    # The working directory is where OGGM will store the run's data
    cfg.PATHS['working_dir'] = os.path.join(gettempdir(), 'Docs_GlacierDir2')
    gdirs = workflow.init_glacier_regions('RGI60-11.00897', from_prepro_level=4,
                                          prepro_border=10)
    gdir = gdirs[0]  # init_glacier_regions always returns a list

    fls = gdir.read_pickle('model_flowlines')
    fls
    [fl.order for fl in fls]

This glacier has three flowlines of type `MixedBedFlowline` provided as a list.
And the flowlines are orderd by ascending Strahler numbers, where the last entry
in the list is always the longest and very often most relevant flowline of
that glacier.

Type of model flowlines
-----------------------

.. currentmodule:: oggm.core.flowline

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Flowline
    MixedBedFlowline
    ParabolicBedFlowline
    RectangularBedFlowline
    TrapezoidalBedFlowline


Important flowline functions
----------------------------

.. currentmodule:: oggm.core

.. autosummary::
    :toctree: generated/
    :nosignatures:

    centerlines.initialize_flowlines
    centerlines.compute_centerlines
    centerlines.compute_downstream_line
    flowline.init_present_time_glacier
