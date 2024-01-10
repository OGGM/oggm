.. currentmodule:: oggm

Version history
===============

v1.6.2 (unreleased)
-------------------

Enhancements
~~~~~~~~~~~~

- There is now a possibility for initializing a elevation-band flowline using
  external thickness data and conduct a dynamic run with it (:pull:`1658`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- The default minimum thickness for the dynamic spinup was changed from 10 m
  to 2 m. The new value was found in a local study and makes a larger
  difference for smaller (thinner) glaciers (:pull:`1667`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_

Bug fixes
~~~~~~~~~

- The binned variables in the elevation band flowlines did not use the
  glacier mask when preserving the total values. This is a bad
  bug that is now fixed (:pull:`1661`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_


v1.6.1 (August 27, 2023)
------------------------

A new minor release of the OGGM with several improvements and bug fixes.
We recommend all users currently using 1.6.0 to switch to this version if they
are still in the testing/learning phase. If you rely on your results
staying stricly the same, you should stick to the version you are currently
running.


Breaking changes
~~~~~~~~~~~~~~~~

OGGM 1.6.1 should be fully compatible with 1.6.0 code. We have updated the
pre-procecessed directories however, and recommend users to switch to the
new OGGM version if possible.

The new pre-processed directories are available in the 2023.3 version:
https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3

These incorporate all the changes listed below.

Bug fixes
~~~~~~~~~

- Corrected a small bug in the W5E5 climate files, which led to some glaciers
  getting climate data from a grid point further away than they should. This
  should affect the results of a few thousand glaciers in a minimal way.
  Fix (:pull:`1547`, :pull:`1557`) by `Lilian Schuster <https://github.com/lilianschuster>`_
- Added more flexibility to ``compile_run_output``. It is now possible to
  compile runs with different data variables (the default is NaN). It is
  needed to compile different spinup strategies together, as some include
  special data variables (e.g. ``is_fixed_geometry_spinup``) (:pull:`1563`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Fixed a problem in the hydro outputs where on some occasions ``melt_on_glacier``
  would be negative. We changed this term to become a positive term for
  snowfall on glacier instead (:pull:`1584`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- Fixed an issue with volume below water level computations with lake-terminating
  glaciers (:pull:`1584`). This affected only diagnostic computations of the
  ``volume_below_water`` variable and should be insignificant.
  By `Fabien Maussion <https://github.com/fmaussion>`_

Enhancements
~~~~~~~~~~~~

- There is now the possibility to compute distributed area and thickness
  changes from the flowline projections (:pull:`1576`, :pull:`1585`,
  :pull:`1619`, :pull:`1623`). The
  functionality is currently in the sandbox but is documented in the tutorials.
  By `Anouk Vlug <https://github.com/anoukvlug>`_, `Patrick Schmitt <https://github.com/pat-schmitt>`_
  and `Fabien Maussion <https://github.com/fmaussion>`_
- Added three new flowline diagnostic variables: thickness change in one year
  (``dhdt``), forcing climatic mass balance (``climatic_mb``) and flux divergence
  (``flux_divergence``). All variables are in units meter of ice per year
  (:pull:`1595`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added more flexibility to ``run_dynamic_spinup``. Users can now specify a target
  year and a desired value to match. The default is still the same, matching area
  or volume at the RGI date (:pull:`1600`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added possibility to use MultiPolygon outlines together with elevation bands.
  That can be useful when working with local glacier inventories with multiple
  outlines (e.g. older outline single polygon but newer outline multi polygon for
  the same glacier) (:pull:`1604`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- OGGM can now read RGI7 files. OGGM was used to generate the RGI-TOPO dataset
  as well as auxiliary products for RGI7 (:pull:`1572`, :pull:`1590`).
  By `Alexander Fischer <https://github.com/afisc>`_
  and `Fabien Maussion <https://github.com/fmaussion>`_
- The pre-processed directories are now run with dynamical spinup and calibration
  as the standard option. Dynamical calibration is run with a lower error
  tolerance than before, improving results in all regions (:pull:`1558`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- OGGM now provides "standard" projections, attached to a specific OGGM version
  (:pull:`1627`). This will be a huge asset for many users and will help to
  track important changes as OGGM continues to improve.
  By `Lilian Schuster <https://github.com/lilianschuster>`_


v1.6.0 (March 10, 2023)
-----------------------

A new major release of the OGGM with several important changes. We recommend
all users to switch to this version only if they are ready for a new study,
or are prepared to rerun their simulations with changed results.

Breaking changes
~~~~~~~~~~~~~~~~

- we removed the ``init_glacier_regions`` task, which was deprecated since
  a few OGGM versions. Similarly, other old functions
  (e.g. ``process_cmip5_data``) were also removed.
- several default parameters were updated to new values. See "migrating guide"
  (in construction) to navigate through these changes.
- the calibration of the mass balance models with the :math:`t^*` ("T star")
  method is no longer supported. The new calibration scheme is considerably
  more flexible, but relies on new parameter names.
- as a result, old workflows and old glacier directories cannot be used (after
  Level 3) in OGGM v1.6 anymore.

.. warning::

    Because of the many changes, the list below is not exhaustive at all. We
    preferred to focus on what is new in the tutorials, and recommend all returning
    users to go over the new tutorials to familiarize themselves with the changes.

Enhancements
~~~~~~~~~~~~

- added support for new reference data (W5E5), a bias corrected dataset based
  on ERA5. This will become the new default in OGGM (:pull:`1435`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- added support for a precipitation factor varying per glacier (:pull:`1435`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- Added a new entity task ``run_dynamic_melt_f_calibration``. This task
  dynamically calibrates the temperature sensitivity mu star to a geodetic
  mass-balance observation. There are different options available how this is
  done, the default incorporates an inversion and a dynamic spinup in each
  iteration (:pull:`1425`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added two new output variables in ``FlowlineModel.run_until_and_store()``
  (``area_m2_min_h`` and ``volume_m3_min_h``) which are needed for a dynamic
  mu star calibration which should include the minimum ice thickness argument
  of an dynamic spinup (needed as a filter for interannual changes of especially
  the area). Also included ``cfg.PARAMS['dynamic_spinup_min_ice_thick']`` to be
  able to globally define the used minimum ice thickness for the dynamic spinup
  (:pull:`1425`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Rearanged the entity tasks ``run_dynamic_melt_f_calibration`` and
  ``run_dynamic_spinup`` with all help functions in new modul
  ``oggm.core.dynamic_spinup`` (:pull:`1425`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Rearanged the prepro levels. Level 4 now adds a historical run (previously
  done in level 5) and a spinup historical run (using dynamic mu star calibration).
  Level 5 now replaces level 4 and creates the minigdirs (where only the files
  for a model run are kept and no inversion is possible anymore) (:pull:`1425`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added support for Millan et al 2022 velocity and thickness in the shop (:pull:`1443`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added support for Hugonnet et al 2021 dhdt in the shop (:pull:`1529`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added trapezoidal downstream line (:pull:`1491`). Can be selected with
  ``cfg.PARAMS['downstream_line_shape']``, with the options ``'parabol'`` (default)
  or ``'trapezoidal'`` before calling ``init_present_time_glacier(gdir)``.
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added option to plot flowline velocities in ``graphics.plot_modeloutput_map()``
  (:pull:`1496`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added option to extend the plot limits when plotting multiple gdirs. Could be
  used with ``extend_plot_limits=True``, e.g.
  ``graphics.plot_modeloutput_map(gdirs, extend_plot_limits=True)``
  (:pull:`1508`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added new argument ``add_fixed_geometry_spinup`` to extend the model run of
  ``run_dynamic_spinup`` with a fixed-geometry-spinup if the spinup period is
  shortened(:pull:`1514`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added SemiImplicitModel for a single trapezoid or rectangular flowline developed
  by `Dan Goldberg <https://github.com/dngoldberg>`_ (:pull:`1507`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Adapted ``filter_inversion_output`` to conserve the bed shape during
  filtering (:pull:`1502`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Adapted calculation of inversion flux to avoid zero thickness at last grid
  point (:pull:`1502`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added the possibility to use the UTM map proj instead of the local TM used
  by OGGM usually (:pull:`1526`). Leads to qualitative and quantitative
  differences when used.
  By `Fabien Maussion <https://github.com/fmaussion>`_


Bug fixes
~~~~~~~~~

- corrected a but in ``apparent_mb_from_any_mb``, where only two years of MB
  would be used instead of a range of years (:pull:`1426`).
  By `Bowen <https://github.com/bowenbelongstonature>`_
- Corrected ``source`` argument in ``tasks.define_glacier_region`` to handle a
  list of DEM sources. (:pull:`1506`).
  By `Daniel Otto <https://github.com/d-otto>`_


v1.5.3 (02.04.2022)
-------------------

New release of the OGGM model, setting the ground for a major update to be
released soon. Several major improvements available for testing, and will
become the default in a future major release:

- new ``MassRedistributionCurveModel`` which uses the Huss curves to parameterize
  glacier retreat.
- new ``mu_star_calibration_from_geodetic_mb`` task which now calibrates
  each glacier individually
- new ``run_dynamic_spinup`` task which used the ice dynamics model to spinup the
  model (instead of the default equilibrium assumption)
- and much more! See below.


Breaking changes
~~~~~~~~~~~~~~~~

- In the process of adding new output diagnostic files (:pull:`1308`), the
  signature and return output of ``FlowlineModel.run_until_and_store``
  changed. We hope that this change won't affect too many of our users but
  if it does, it should be relatively straightforward to update your code:
  users now control the number of outputs with the ``fl_diag_path`` and
  ``geom_path`` kwargs. Most users will probably have used the ``run_*``
  tasks anyway, and won't be affected by this change (except maybe for the
  point below). By `Fabien Maussion <https://github.com/fmaussion>`_
- Furthermore, ``cfg.PARAMS['store_model_geometry']`` is now set to ``False``
  per default. If you were relying on these files (e.g. for a run with spinup
  or similar), your code will fail with ``FileNotFoundError`` for the
  ``model_geometry`` files. Setting ``cfg.PARAMS['store_model_geometry']``
  back to ``True`` should solve the issue.
- Copernicus DEM 90m is now called ``'COPDEM90'`` instead of ``'COPDEM'`` and also uses the
  2021 release with additional data corrections (:pull:`1364`).


Enhancements & bug fixes
~~~~~~~~~~~~~~~~~~~~~~~~

- Added the ELA back as a variable. Unlike before it is now a diagnostic
  variable that can be computed independent of a model run with
  ``global_tasks.compile_ela`` (:pull:`1333`).
  By `Anouk Vlug <https://github.com/anoukvlug>`_
- Added a ``prescribe_years`` kwarg to ``RandomMassBalance`` to control
  which years are picked instead of the random number generator (:pull:`1310`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a ``stop_criterion`` kwarg to ``run_until_and_store``, so that
  users can specify when a simulation has to stop based on their chosen
  criteria (:pull:`1303`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Implemented a ``MassRedistributionCurveModel`` under the FlowlineModel
  interface, which uses the Huss curves to parameterize glacier retreat.
  There is some parameterisation for advance as well, but a very coarse one
  (:pull:`1288`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new ``mu_star_calibration_from_geodetic_mb`` task which now calibrates
  each glacier individually from the reference geodetic MB data. This is a rather
  quick solution for now, but it opens new avenues (:pull:`1286`).
  A bug in the new feature was later corrected (:pull:`1351`).
  By `Fabien Maussion <https://github.com/fmaussion>`_ and
  `Lilian Schuster <https://github.com/lilianschuster>`_
- Added a new ``utils.get_geodetic_mb_dataframe`` which returns the
  reference geodetic MB data, currently from Hugonnet et al 2021.
  Also changed the behavior of ``cfg.DATA`` to be shared
  across processes (:pull:`1285`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new output diagnostic files containing variables along the flowlines
  instead of aggregated at the glacier level (:pull:`1308`). These files are
  stored in the glacier directory (``gdir.get_filepath('fl_diagnostics')``) and
  are not saved per default. Set ``cfg.PARAMS['store_fl_diagnostics'] = True``
  to activate it.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new keyword argument to ``run_with_hydro``, ``run_from_climate_data``,
  and ``run_until_and_store``: ``fixed_geometry_spinup_yr`` which allows to
  "start" a simulation at an earlier date than the RGI date (:pull:`1327`). In practice,
  it computes the glacier volume change from SMB only (fixed geometry),
  ignoring glacier area change in this period. Therefore, it is only valid for
  short periods of times (years, not decades).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added new keyword arguments to ``run_with_hydro`` which allow to
  select which glacier area should be used as reference for the hydro
  computations (:pull:`1331`). Furthermore, it also allows to use a previous
  geometry file for the computations, i.e. allowing for continuous
  historical to projections outputs (if needed).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The package "descartes" is no longer a required dependency for oggm.
- Added a new entity task ``run_dynamic_spinup``. This task dynamically spinup
  the glacier to match the area or volume at the RGI date. To do so the glacier
  is simulated from the recent past (default 1980) to the RGI date. The unknown
  glacier geometry at the start of the simulation is iteratively changed with
  a short constant climate run with a varying temperature bias
  (:pull:`1342`, :pull:`1361`).
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added new options to `write_centerlines_to_shape` which allow to output
  smoother and more correct centerlines (:pull:`1357`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added the 30m version of the
  `Copernicus DEM <https://spacedata.copernicus.eu>`_.
  This DEM can be set with ``source='COPDEM30'`` and can be useful for smaller sized glaciers.
  An account with Copernicus is required to access the DEM (free for academics).
  (:pull:`1364`). By `Matthias Dusch <https://github.com/matthiasdusch>`_
- Added a new `merge_consecutive_run_outputs` entity task which allows to merge
  two output files together (:pull:`1379`). This is useful to merge e.g.
  spinup + historical or historical + projection runs in post-processing.
  By `Fabien Maussion <https://github.com/fmaussion>`_


v1.5.2 (29.08.2021)
-------------------

Very minor release to remove the dependency on the ``python-colorspace`` package.

Breaking changes
~~~~~~~~~~~~~~~~

- Removed use and dependency on the ``python-colorspace`` package. Although
  not very dramatic, this change might change the way your plots look like
  (:pull:`1284`). Users can specify their own colormaps anyways to get back to
  the old plots.
  By `Fabien Maussion <https://github.com/fmaussion>`_

v1.5.1 (28.08.2021)
-------------------

This is a minor release of OGGM, containing mostly bugfixes and a few new
features.

Breaking changes
~~~~~~~~~~~~~~~~

This version should be fully backwards compatible.

Enhancements
~~~~~~~~~~~~

- Added ``cook_rgidf()`` function in ``oggm.utils`` to simplify the use
  of a non-RGI glacier inventory in OGGM (:pull:`1251`).
  By `Li Fei <https://github.com/Keeptg>`_
- Added support for last millennium reanalysis data to the shop (:pull:`1257`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new ``apply_func``  argument in ``utils.compile_glacier_statistics``
  that allows user to compute any new statistics from a gdir themselves (:pull:`1259`)
  By `Li Fei <https://github.com/Keeptg>`_
- Added a new ``workflow.match_geodetic_mb_for_selection`` function to match
  the MB bias for any selection of glaciers (:pull:`1248`)
  By `Patrick Schmitt <https://github.com/pat-schmitt>`_
- Added functionality to control the area over which the hydrological
  output is computed (:pull:`1264`, :pull:`1276`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new (wrong) way to compute equilibrium runs based on the average
  climate (:pull:`1275`).
  By `Fabien Maussion <https://github.com/fmaussion>`_

Bug fixes
~~~~~~~~~

- Fixed a quite bad bug where monthly runoff data would have very large artifacts
  (:pull:`1283`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Small bug fix to ensure backwards compatibility of ``gdir.get_filepath('model_run')``.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Various backwards compatibility fixes (mostly xarray).

v1.5.0
------

This a new update of the OGGM model. It should be largely compatible with
OGGM v1.4.0. **The main addition in this release is the computation of
hydrological diagnostics.** Check-out the new tutorial at
https://oggm.org/tutorials !

Breaking changes
~~~~~~~~~~~~~~~~

- Mass balance models now do their computations with float64 arrays instead
  of float32 (:pull:`1211`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- `prcp_bias` renamed to `prcp_fac` in mass balance models (:pull:`1211`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- Various name changes (with deprecation cycle, i.e. old code should still
  work): ``gdir.get_filepath('model_run')`` renamed to ``gdir.get_filepath('model_geometry')``;
  ``run_path`` kwarg in ``run_until_and_store`` renamed to ``geom_path``.


Enhancements
~~~~~~~~~~~~

- Mass balance models now properly refer to ``prcp_fac`` (was incorrectly named
  ``prcp_bias``) (:pull:`1211`).
  Additionally, the ``run_*`` tasks in ``oggm.core.flowline`` can now also adjust
  the precipitation factor for sensitivity experiments.
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- Users can now choose the variables that are written in diagnostics files
  and can also choose not to write the glacier geometry files during the run
  (:pull:`1219`). The respective global parameters are ``store_diagnostic_variables``
  and ``store_model_geometry``.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added experimental ``run_with_hydro`` task which computes hydrological
  diagnostics after a standard dynamical run (:pull:`1224`).
  This is highly experimental, will likely change in the future.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added monthly output to ``run_with_hydro`` (:pull:`1232`).
  By `Sarah Hanus <https://github.com/sarah-hanus>`_ and
  `Fabien Maussion <https://github.com/fmaussion>`_
- Added new diagnostic variables (`terminus_thick_0` ...) to better track
  the thickness at the terminus (:pull:`1230`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Temperature and precipitation corrections can now also be applied on a
  monthly basis for the PastMassBalanceModel (:pull:`1247`).
  By `Fabien Maussion <https://github.com/fmaussion>`_


Bug fixes
~~~~~~~~~

- Fixed bug in hydro date / calendar date conversions with month=1 (i.e.
  no conversion) (:pull:`1220`).
  By `Lilian Schuster <https://github.com/lilianschuster>`_
- Fixed bug in ``graphics.plot_distributed_thickness`` which led to an error with
  elevation bands flowlines (:pull:`1241`).
  By `Li Fei <https://github.com/Keeptg>`_

v1.4.0 (17.02.2021)
-------------------

This a new major update of the OGGM model. It it the result of one year
of development, with several non-backwards compatible changes.

We recommend all users to update to this version.

.. admonition:: **Major new release 1.4!**

    There have been a large number of additions too long to be summarized,
    and the list below is far from complete (we have waited way too long for
    this release). Here are the highlights:

       - new option to compute centerlines: "elevation band flowlines"
       - new option to calibrate OGGM mass balance regionally to geodetic
         estimates
       - new option to calibrate the creep parameter Glen A to match the ice
         thickness to the Farinotti et al. (2019) consensus
       - users can now choose from a variety of pre-processed directories,
         including with new climate data (e.g. ERA5)
       - OGGM now has a calving parameterization (switched off per default)
       - OGGM shop, to download several new input datasets
       - Historical runs ("spin-up") are now available per default and can
         be readily used for projections
       - and much much more....


**Note:** not all changes since v1.3.1 are documented below. We'll try to be
better with documenting changes in the future.

Breaking changes
~~~~~~~~~~~~~~~~

- The dynamical model now has a "real" parameterization for calving (WIP).
  A blog post explaining it can be found
  `on the website <https://oggm.org/2020/02/16/calving-param/>`_
  (:pull:`945`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The calving parameterization has been extended and made more useful
  by logging its output to compiled files. See :pull:`996` for code changes
  and watch out for upcoming changes in the documentation and notebooks.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The new default when applying GCM data to a glacier is to correct for
  temperature standard deviation (:pull:`978`). The previous default was
  wrong and should not be the default.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a new "glacier directory initialization" global task:
  `init_glacier_directories` (:pull:`983`, :issue:`965`). It replaces
  `init_glacier_regions` and covers all its functionality, except that
  it does *not* process the DEM data (this was a confusing "feature" of
  `init_glacier_regions`). The old task `init_glacier_regions` is officially
  deprecated but without warnings for now. Since it is a very widely used
  task, we prefer to deprecate it in a slow cycle: first, change the
  documentation, deprecate later.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- More climate datasets are now supported in OGGM (:pull:`1036`).
  A new task (`historical_climate_qc`) has been added to quality check the
  climate timeseries.
  This has not been tested intensively yet and is still a WIP.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The order of the tasks applied to  the preprocessed levels has
  changed, climate data comes in later (:pull:`1038`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The default DEMS used for each glacier have changed for more modern ones
  (:pull:`1073`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The inversion tasks now can invert for trapezoid shapes (:pull:`1045`). This
  has non-trivial consequences for the model workflow. First and foremost,
  the decision about using a trapezoid bed (instead of parabolic when the
  parabola is too "flat") is taken *at the inversion step* and not afterwards.
  The forward model and the inversion are therefore much more consistent.
  Furthermore, the task `filter_inversion_output` was simplified to take the
  estimated downstream bedshapes into account and now preserves glacier area,
  not volume. This also is a step towards more physical consistency between
  inverse and forward model.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- The `vascaling` module has been removed from oggm core (:pull:`1065`). It
  is now available via a separate package (`oggm-vas <https://github.com/OGGM/oggm-vas>`_,
  maintained by Moritz Oberrauch).
- New options to compute the length of a glacier during a run:
  `PARAMS['min_ice_thick_for_length']` and `PARAMS['glacier_length_method']`
  (:pull:`1069`). By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Several further important changes to be documented later in
  (:pull:`1099`). By `Fabien Maussion <https://github.com/fmaussion>`_.


Enhancements
~~~~~~~~~~~~

- Added Copernicus DEM GLO-90 as optional DEM. Requires credentials to
  ``spacedata.copernicus.eu`` stored in a local ``.netrc`` file. Credentials
  can be added on the command line via ``$ oggm_netrc_credentials``
  (:pull:`961`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Added NASADEM as optional DEM. This is a improved version of  SRTM and could
  replace the current SRTM (https://lpdaac.usgs.gov/products/nasadem_hgtv001/).
  (:pull:`971`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Added support for "squeezed" flowlines from Huss & Farinotti 2012
  (:pull:`1040`). The corresponding tasks are
  :py:func:`tasks.elevation_band_flowline` and
  :py:func:`tasks.fixed_dx_elevation_band_flowline`.
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added a `calibrate_inversion_from_consensus` global task which
  calibrates Glen A so that the volume of glaciers in a selection is
  matched (:pull:`1043`).
  By `Fabien Maussion <https://github.com/fmaussion>`_
- Added support for writing a NetCDF variable in ``gridded_data.nc`` file to
  a georeferenced GeoTiff file (:pull:`1118`). The new task are
  :py:func:`tasks.gridded_data_var_to_geotiff`. 
  By `Li Fei <https://github.com/Keeptg>`_
- Added a `find_inversion_calving_from_any_mb` task which uses the Recinos et
  al. approach, but on any mass balance profile (:pull:`1043`).
  By `Fabien Maussion <https://github.com/fmaussion>`_


Bug fixes
~~~~~~~~~

- Maintenance updates for upstream libraries and various small bug fixes
  (:pull:`957`, :pull:`967`, :pull:`968`, :pull:`958`, :pull:`974`, :pull:`977`,
  :pull:`976`, :pull:`1124`).
  By `Fabien Maussion <https://github.com/fmaussion>`_,
  `Matthias Dusch <https://github.com/matthiasdusch>`_ and 
  `Li Fei <https://github.com/Keeptg>`_.


v1.3.1 (16.02.2020)
-------------------

Minor release with small improvements but an important and necessary change in
multiprocessing.

Enhancements
~~~~~~~~~~~~

- After a recent change in multiprocessing, creating a pool of workers became
  very slow. This change was necessary because of race conditions in GDAL,
  but these conditions are rarely relevant to users. We now make this
  change in multiprocessing optional (:pull:`937`)
- various improvements and changes in the dynamical model - mass balance model
  API. These were necessary to allow compatibility with the PyGEM model
  (:pull:`938`, :pull:`946`, :pull:`949`, :pull:`953`, :pull:`951`).
  By `Fabien Maussion <https://github.com/fmaussion>`_ and
  `David Rounce <https://github.com/drounce>`_.
- added a "flux gate" to allow for precise mass-conservation checks in
  numerical experiments (:pull:`944`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


v1.3.0 (02.02.2020)
-------------------

The time stepping scheme of OGGM has been fixed for several flaws.
`This blog post <https://oggm.org/2020/01/18/stability-analysis/>`_
explains it in detail. We expect some changes in OGGM results after this
release, but they should not be noticeable in a vast majority of the cases.

We recommend all users to update to this version.

Breaking changes
~~~~~~~~~~~~~~~~

- The adaptive time stepping scheme of OGGM has been fixed for several flaws
  which lead to unstable results in certain conditions.
  See `the blog post <https://oggm.org/2020/01/18/stability-analysis/>`_
  for a full description. The API didn't change in the process, but the
  OGGM results are likely to change slightly in some conditions.
  (:issue:`731`, :issue:`860`, :pull:`931`).
  By `Fabien Maussion <https://github.com/fmaussion>`_ and
  `Alex Jarosch <https://github.com/alexjarosch>`_.

Enhancements
~~~~~~~~~~~~

- The `test_models` test module has been refactored to use pytest fixtures
  instead of unittest classes (:pull:`934` and :pull:`922`).
  By `Chris Merrill <https://github.com/C-Merrill>`_.


v1.2.0 (04.01.2020)
-------------------

**OGGM is released under a new license.** We now use the
`BSD-3-Clause <https://github.com/OGGM/oggm/blob/master/LICENSE.txt>`_ license.

v1.1.3 (03.01.2020)
-------------------

Minor release of the OGGM model with several small improvements.
We don't expect major changes in the model results due to this release.

**Important:** this will be the last release under a GPL license. The next
release (v1.2) will be done without modifications but under a BSD-3-Clause
License.

Enhancements
~~~~~~~~~~~~

- New function ``cfg.add_to_basenames`` now allows users to define their own
  entries in glacier directories (:issue:`731`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- New function ``inversion.compute_inversion_velocities`` writes the section and
  surface veloicites in the inversion output (:issue:`876`).
  By `Beatriz Recinos <https://github.com/bearecinos>`_.
- Added ASTER v3 as optional DEM. Requires credentials to
  ``urs.earthdata.nasa.gov`` stored in a local ``.netrc`` file. Credentials
  can be added on the command line via ``$ oggm_netrc_credentials``
  (:pull:`884`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Added a global task (``tasks.compile_task_time`` and the associated method at
  the GlacierDirectory level ``get_task_time``) to time the execution of
  entity tasks (:issue:`918`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Improved performance of numerical core thanks to changes in our calls to
  `np.clip` (:pull:`873` and :pull:`903`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added a function `cfg.initialize_minimal` to run the flowline model
  without enforcing a full download of the demo files (:pull:`921`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Bug fixes
~~~~~~~~~

- Small bugs in DEM processing fixed - internal refactoring followed,
  hopefully for the good (:pull:`890` and :pull:`886`).
  By `Fabien Maussion <https://github.com/fmaussion>`_ and
  `Matthias Dusch <https://github.com/matthiasdusch>`_.


v1.1.2 (12.09.2019)
-------------------

Minor release of the OGGM model, with several substantial improvements, most
notably:

- update in the inversion procedure for calving glaciers (Recinos et al., 2019)
- new glacier evolution model based on Marzeion et al., 2012

We don't expect major changes in the model results due to this release.


Breaking changes
~~~~~~~~~~~~~~~~

- ``run_until`` now makes sure that the years (months) are not crossed by
  the adaptive time-stepping scheme (:issue:`710`). ``run_until`` and
  ``run_until_and_store`` should now be consistent. The change is unlikely to
  affect the majority of users (which used ``run_until_and_store``), but
  the results or ``run_until`` can be affected (:pull:`726`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- ``find_inversion_calving`` has been renamed to
  ``find_inversion_calving_loop`` and will probably be deprecated soon
  (:pull:`794`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- ``use_rgi_area=False`` now also recomputes CenLon and CenLat on the fly.
  (:issue:`838`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Enhancements
~~~~~~~~~~~~

- Added new ``gridded_attributes`` and ``gridded_mb_attributes`` tasks to
  add raster glacier attributes such as slope, aspect, mass balance...
  to the glacier directory (:pull:`725`). This can be useful for statistical
  modelling of glacier thickness.
  By `Matteo Castellani <https://github.com/MatCast>`_.
- Added support for a new DEM dataset: Mapzen, found on Amazon cloud
  (:issue:`748`, :pull:`759`). Also added some utility functions to handle
  DEMs, to be improved further in the near future.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added support for a new DEM dataset: REMA (:pull:`759`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added an option to pre-process all DEMs at once (:pull:`771`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added support for another evolution model: the volume-area-scaling based
  model of Marzeion et al., 2012 (:pull:`662`). This is a major enhancement
  to the code base as it increases the number of choices available to users
  and demonstrates the modularity of the model.
  By `Moritz Oberrauch <https://github.com/oberrauch>`_.
- Changed the way the calving flux is computed during the ice thickness
  inversion. This no longer relies on an iteration over mu*, but solves
  for `h` instead. The new function is likely to replace the "old"
  calving loop (:pull:`794`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- ``compile_climate_input`` and ``compile_run_output`` are now faster for
  larger numbers of glaciers thanks to temporary files (:pull:`814`).
  By `Anouk Vlug <https://github.com/anoukvlug>`_. Could be made faster with
  multiprocessing one day.
- OGGM can now run in "minimal mode", i.e. without many of the hard
  dependencies (:issue:`420`). This is useful for teaching or idealized
  simulations, but won't work in production.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- the flowline model gives access to new diagnostics such as ice velocity and
  flux along the flowline. The numerical core code changed in the process,
  and we will monitor performance after this change (:pull:`853`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


Bug fixes
~~~~~~~~~

- Preprocessed directories at the level 3 now also have the glacier flowlines
  ready for the run (:issue:`736`, :pull:`771`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Nominal glaciers now error early in the processing chain (:issue:`832`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Specific MB (not used operationaly) was wrongly computer for zero ice
  thickness rectangular or parabolic sections. This is now corrected
  (:issue:`828`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Fixed a bug in model output files where SH glaciers were wrongly attributed
  with NH calendar dates (:issue:`824`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


v1.1.1 (24.04.2019)
-------------------

Minor release of the OGGM model, with several bugfixes and some improvements.

We don't expect any change in the model results due to this release.

Enhancements
~~~~~~~~~~~~

- Adapted ``graphics.plot_domain``, ``graphics.plot_centerlines`` and
  ``graphics_plot_modeloutput_map`` to work with merged glaciers (:pull:`726`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Added (and updated) an official task to find the calving flux based on the
  mass-conservation inversion (`inversion.find_inversion_calving`). This
  is still in experimentation phase! (:pull:`720`).
  By `Beatriz Recinos <https://github.com/bearecinos>`_.
- Added a mechanism to add custom MB data to OGGM (:issue:`724`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The ALOS Global Digital Surface Model "ALOS World 3D - 30m" DEM from JAXA can
  now be used as alternative DEM within OGGM (:pull:`734`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- Switch to setuptools-scm as a version control system (:issue:`727`).
  By `Timo Rothenpieler <https://github.com/TimoRoth>`_.

Bug fixes
~~~~~~~~~

- Fixed several problems with the file download verification algorithm.
  By `Timo Rothenpieler <https://github.com/TimoRoth>`_.
- Fixed a timing problem in the benchmark command line tool (:pull:`717`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


v1.1 (28.02.2019)
-----------------

This is a major new release of the OGGM model, with substantial improvements
to version 1. We recommend to use this version from now on. It coincides
with the publication of our publication in
`Geoscientific Model Development <https://www.geosci-model-dev.net/12/909/2019/>`_.

New contributors to the project:

- **Matthias Dusch** (PhD student, University of Innsbruck), added extensive
  cross-validation tools and an associated website.
- **Philipp Gregor** (Master student, University of Innsbruck), added options
  to switch on lateral bed stress in the flowline ice dynamics
- **Nicolas Champollion** (PostDoc, University of Bremen), added GCM data
  IO routines.
- **Sadie Bartholomew** (Software Engineer, UK Met Office), added ability to
  replace colormaps in graphics with HCL-based colors using python-colorspace.

Breaking changes
~~~~~~~~~~~~~~~~

- The utils.copy_to_basedir() function is changed to being an entity task. In
  addition gcm_data files, when present, will from now on always be copied
  when using this task (:issue:`467` & :pull:`468`).
  By `Anouk Vlug <https://github.com/anoukvlug>`_.
- Accumulation Area Ratio (AAR) is now correctly computed (:issue:`361`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The method used to apply CRU and GCM anomalies to the climatology has
  changed for precipitation: we now use scaled anomalies instead of the
  standard anomalies (:pull:`393`). The previous method might have lead to
  negative values in some cases. The corresponding reference t* have also
  been updated (:pull:`407`). This change has some consequences on the
  the model results: cross-validation indicates very similar scores, but
  the influence on global model output has not been assessed yet.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- It is now possible to run a simulation with spinup in the standard
  workflow (:pull:`411`). For this to happen it was necessary to clean up
  the many `*filesuffix` options. The new names are more explicit
  but not backwards compatible. The previous `filesuffix` is now
  called `output_filesuffix`. The previous `input_filesuffix` is now
  called `climate_input_filesuffix`. The `random_glacier_evolution` task
  is now called `run_random_climate` for consistency with the other tasks
  See the PR linked above for more info.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- RGI version 4 isn't supported anymore (:issue:`142`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Rework of the 2d interpolation tasks for ice thickness in the context of
  `ITMIX2 <http://oggm.org/2018/05/21/g2ti/>`_. The new interpolation
  are better, but not backwards compatible. Aside of me I don't think
  anybody was using them (:pull:`465`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Diagnostic variables (length, area, volume, ELA) are now stored at annual
  steps instead of monthly steps (:pull:`488`). The old behavior can still be
  used with the ``store_monthly_step`` kwarg. Most users should not notice
  this change because the regionally compiled files were stored at yearly
  steps anyways.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The list of reference t* dates is now generated differently: instead of
  the complex (and sort of useless) nearest neighbor algorithm we are now
  referring back to the original method of Marzeion et al. (2012). This comes
  together with other breaking changes, altogether likely to change the
  results of the mass balance model for some glaciers. For more details see
  the PR: :pull:`509`
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The ice dynamics parameters (Glen A, N, ice density) are now "real"
  parameters accessible via ``cfg.PARAMS`` (:pull:`520`, :issue:`511` and
  :issue:`27`). Previously, they were also accessible via a module attribute
  in ``cfg``, which was more confusing than helping. Deprecated and removed
  a couple of other things on the go, such as the dangerous `
  ``optimize_inversion_params`` task (this cannot be optimized yet) and the
  useless ``volume_inversion`` wrapper (now called
  ``mass_conservation_inversion``)
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The temperature sensitivity mu* is now flowline specific, instead of
  glacier wide. This change was necessary because it now allows low-lying
  tributaries to exist, despite of too high glacier wide mu*. This change
  had some wider reaching consequences in the code base and in the
  mass balance models in particular: :pull:`539`. This will also allow to
  merge neighboring glaciers in the future.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The "human readable" mu* information is now stored in a JSON dict instead
  of a csv: :pull:`568`.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The global task `glacier_characteristics` has been renamed to
  `compile_glacier_statistics` (:pull:`571`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The ``process_cesm_data`` task has been been moved to `gcm_climate.py`
  addressing: :issue:`469` & :pull:`582`.
  By `Anouk Vlug <https://github.com/anoukvlug>`_.
- The shapefiles are now stored in the glacier directories as compressed
  tar files, addressing :issue:`367` & :issue:`615`. This option can be
  turned off with `cfg.PARAMS['use_tar_shapefiles'] = False`.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Enhancements
~~~~~~~~~~~~

- Added a utility function to easily get intersects files (:pull:`402`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The old GlaThiDa file linking the total volume of glaciers (T database) to
  RGI has been updated to RGI Version 6 (:pull:`403`).
  Generally, we do not recommend to use these data for calibration or
  validation because of largely unknown uncertainties.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The computing efficiency of the 2D shallow ice model has been increased
  by a factor 2 (:pull:`415`), by avoiding useless repetitions of indexing
  operations. The results shouldn't change at all.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added optional shape factors for mass-conservation inversion and
  FluxBasedModel to account for lateral drag dependent on the bed shape
  (:pull:`429`). Accepted settings for shape factors are `None`,
  `'Adhikari'` (Adhikari & Marshall 2012), `'Nye'` (Nye, 1965; equivalent to
  Adhikari) and `'Huss'` (Huss & Farinotti 2012). Thorough tests with
  applied shape factors are still missing.
  By `Philipp Gregor <https://github.com/phigre>`_.
- Some amelioration to the mass balance models (:pull:`434`). Added a
  ``repeat`` kwarg to the ``PastMassBalance`` in order to loop over a
  selected period. Added an ``UncertainMassBalance`` model which wraps
  an existing model and adds random uncertainty to it.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The DEM sources are now clearly stated in each glacier directory,
  along with the original data citation (:pull:`441`). We encourage
  to always cite the original data providers.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added some diagnostic tools which make it easier to detect dubious glacier
  outlines or DEM errors (:pull:`445`). This will be used to report to the
  RGI authors.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added a new parameter (``PARAMS['use_rgi_area']``), which specifies whether
  OGGM should use the reference area provided by RGI or the one computed
  from the local map and reprojected outlines  (:pull:`458`, default: True).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- A new ``simple_glacier_masks`` tasks allows to compute glacier rasters in
  a more robust way than the default OGGM method (:pull:`476`). This is useful
  for simpler workflows or to compute global statistics for external tools
  like `rgitools <http://rgitools.readthedocs.io/en/latest/>`_. This task
  also computes hypsometry files much like RGI does.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Reference glaciers now have mass balance profiles attached to them, if
  available. You can get the profiles with ``gdir.get_ref_mb_profile()``
  (:pull:`493`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- New ``process_histalp_data`` task to run OGGM with HISTALP data
  automatically. The task comes with a list of predefined t* like CRU and
  with different default parameters
  (see `blog <https://oggm.org/2018/08/10/histalp-parameters/>`_). The PR
  also adds some safety checks at the calibration and computation of the
  mass balance to make sure there is no misused parameters (:pull:`493`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The ``process_cesm_data`` function has been split into two functions, to make
  it easier to run oggm with the climate of other GCM's: ``process_cesm_data``
  reads the CESM files and handles the CESM specific file logic.
  ``process_gcm_data`` is the general task able to handle all kind of data.
  ``process_cesm_data`` can also be used as an example when you plan make a
  function for running OGGM with another GCM (:issue:`469` & :pull:`582`).
  `Anouk Vlug <https://github.com/anoukvlug>`_.
- New ``process_dummy_cru_file`` task to run OGGM with randomized CRU data
  (:pull:`603`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Colormaps in some graphics are replaced with Hue-Chroma-Luminance (HCL) based
  improvements when python-colorspace is (optionally) installed (:pull:`587`).
  By `Sadie Bartholomew <https://github.com/sadielbartholomew>`_.
- Added a workflow ``merge_glacier_tasks`` which merges tributary/surrounding
  glaciers to a main glacier, allowing mass exchange between them. This is
  helpful/necessary/intended for growing glacier experiments (e.g.
  paleoglaciology) (:pull:`624`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- New ``oggm_prepro`` command line tool to run the OGGM preprocessing tasks
  and compress the directories (:pull:`648`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- `init_glacier_regions` task now accepts RGI Ids strings as input instead of
  only Geodataframes previously (:pull:`656`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The ``entity_task`` decorator now accepts a fallback-function which will be
  executed if a task fails and `cfg.PARAMS['continue_on_error'] = True`. So far
  only one fallback function is implemented for `climate.local_t_star`
  (:pull:`663`).
  By `Matthias Dusch <https://github.com/matthiasdusch>`_.
- New `process_gcm_data` task to handle CMIP5 files.
  By `Nicolas Champollion <https://github.com/nchampollion>`_.


Bug fixes
~~~~~~~~~

- Remove dependency to deprecated matplotlib._cntr module (:issue:`418`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Fixed a bug in tidewater glaciers terminus position finding, where
  in some rare cases the percentile threshold was too low (:pull:`444`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Fixed a caching bug in the test suite, where some tests used to fail when run
  for a second time on a modified gdir (:pull:`448`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Fixed a problem with netCDF4 versions > 1.3 which returns masked arrays
  per default. We now prevent netCDF4 to return masked arrays altogether
  (:issue:`482`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


Internals
~~~~~~~~~

- We now use a dedicated server for input data such as modified RGI files
  (:pull:`408`). By `Fabien Maussion <https://github.com/fmaussion>`_.
- Test fix for googlemaps.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added a utility function (:py:func:`~utils.idealized_gdir`) useful
  to dow flowline experiments without have to create a local map (:pull:`413`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


.. _whats-new.1.0:

v1.0 (16 January 2018)
----------------------

This is the first official major release of OGGM. It is concomitant to the
submission of a manuscript to
`Geoscientific Model Development (GMD) <https://www.geoscientific-model-development.net>`_.

This marks the stabilization of the codebase (hopefully) and implies that
future changes will have to be documented carefully to ensure traceability.

New contributors to the project:

- **Anouk Vlug** (PhD student, University of Bremen), added the CESM
  climate data tools.
- **Anton Butenko** (PhD student, University of Bremen), developed the
  downstream bedshape algorithm
- **Beatriz Recinos** (PhD student, University of Bremen), participated to the
  development of the calving parametrization
- **Julia Eis** (PhD student, University of Bremen), developed the glacier
  partitioning algorithm
- **Schmitty Smith** (PhD student,  Northand College, Wisconsin US), added
  optional parameters to the mass balance models


.. _whats-new.0.1.1:

v0.1.1 (16 February 2017)
-------------------------

Minor release: changes in ITMIX to handle the synthetic glacier cases.

It was tagged only recently for long term documentation purposes and storage
on `Zenodo <https://zenodo.org/record/292630>`_.

.. _whats-new.0.1.0:

v0.1 (29 March 2016)
--------------------

Initial release, used to prepare the data submitted to ITMIX (see
`here <http://www.fabienmaussion.info/2016/06/18/itmix-experiment-phase1/>`_).


This release is the result of several months of development (outside of GitHub
for a large part). Several people have contributed to this release:

- **Michael Adamer** (intern, UIBK), participated to the development of the
  centerline determination algorithm (2014)
- **Kévin Fourteau** (intern, UIBK, ENS Cachan), participated to the
  development of the inversion and the flowline modelling algorithms
  (2014-2015)
- **Alexander H. Jarosch** (Associate Professor, University of Iceland),
  developed the MUSCL-SuperBee model (:pull:`23`)
- **Johannes Landmann** (intern, UIBK), participated to the
  `links between databases`_ project (2015)
- **Ben Marzeion** (project leader, University of Bremen)
- **Fabien Maussion** (project leader, UIBK)
- **Felix Oesterle** (Post-Doc, UIBK) provided the
  AWS deployment script (:pull:`25`)
- **Timo Rothenpieler** (programmer, University of Bremen), participated to the
  OGGM deployment script (e.g. :pull:`34`, :pull:`48`), and developed OGGM
  `installation`_ tools
- **Christian Wild** (master student, UIBK), participated to the development of
  the centerline determination algorithm (2014)

.. _links between databases: https://github.com/OGGM/databases-links
.. _installation: https://github.com/OGGM/OGGM-Anaconda
