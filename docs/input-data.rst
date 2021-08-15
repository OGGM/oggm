.. _shop:

.. currentmodule:: oggm

OGGM Shop
=========

    .. figure:: _static/logos/logo_shop.png
        :width: 100%

OGGM needs various data files to run. **We rely exclusively on
open-access data that can be downloaded automatically for the user**. We like
to see this service as a "shop", allowing users to define a shopping list
of data that they wish to add to their :ref:`glacierdir`.

This page describes the various products you will find in the shop.

.. important::

    Don't forget to set-up or check your system (:ref:`system-settings`) before
    downloading new data! (you'll need to
    do this only once per computer)

.. _preprodir:

Pre-processed directories
-------------------------

The simplest way to run OGGM is to rely on :ref:`glacierdir` which have been
prepared for you by the OGGM developers. Depending on your use case,
you can start from various stages in the processing chain, map sizes,
and model set-ups.

The default directories have been generated with the default parameters
of the current stable OGGM version (and a few alternative combinations).
If you want to change some of these parameters, you *may* have to start a
run from a lower processing level and re-run the processing tasks.
Whether or not this is necessary depends on the stage of the workflow
you'd like your computations to diverge from the
defaults (this will become more clear as we provide an example below).

To start from a pre-processed state, simply use the
:py:func:`workflow.init_glacier_directories` function with the
``from_prepro_level`` and ``prepro_border`` keyword arguments set to the
values of your choice. This will fetch the default directories: there are
more options to these, which we explain below.

Processing levels
~~~~~~~~~~~~~~~~~

Currently, there are six available levels of pre-processing:

- **Level 0**: the lowest level, with directories containing the glacier
  outlines only.
- **Level 1**: directories now contain the glacier topography data as well.
- **Level 2**: at this stage, the flowlines and their downstream lines are
  computed and ready to be used.
- **Level 3**: adding the baseline climate timeseries (CRU or ERA5, see below)
  to the directories. Adding all necessary pre-processing tasks
  for a dynamical run, including the mass-balance calibration, bed inversion,
  up to the :py:func:`tasks.init_present_time_glacier` task included.
  These directories still contain all data that were necessary for the processing,
  i.e. the largest in size but also the most flexible since
  the processing chain can be re-run from any stage in them.
- **Level 4**: same as level 3 but with all intermediate output files removed.
  The strong advantage of level 4 files is that their size is considerably
  reduced, at the cost that certain operations (like plotting on maps or
  running the bed inversion algorithm again) are not possible anymore.
- **Level 5**: on top of level 4, an additional historical simulation is run
  from the RGI date to the last possible date of the baseline climate file
  (currently January 1st 2020 at 00H for CRU and ERA5).
  The state of the glacier (currently set as month 01 in hydrological year 2020) can then be
  used for future projections.

In practice, most users are going to use level 2, level 3 or level 5 files. Here
are some example use cases:

1. *Running OGGM from GCM / RCM data with the default settings*: **start at level 5**
2. *Using OGGM's flowlines but running your own baseline climate,
   mass-balance or ice thickness inversion models*: **start at level 2** (and maybe
   use OGGM's workflow again for glacier dynamic evolution?). This is the workflow used
   by associated model `PyGEM <https://github.com/drounce/PyGEM>`_ for example.
3. *Run sensitivity experiments for the ice thickness inversion*: start at level
   3 (with climate data available) and re-run the inversion steps.


Glacier map size: the prepro_border argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The size of the local glacier map is given in number of grid points *outside*
the glacier boundaries. The larger the map, the largest the glacier can
become. Here is an example with Hintereisferner in the Alps:

.. ipython:: python
   :suppress:

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from oggm import cfg, tasks, workflow, graphics
    from oggm.utils import gettempdir

    cfg.initialize()
    cfg.PATHS['working_dir'] = os.path.join(gettempdir(), 'Docs_BorderSize')

.. ipython:: python
    :okwarning:

    f, axs = plt.subplots(2, 2, figsize=(8, 6))
    for ax, border in zip(np.array(axs).flatten(), [10, 40, 80, 160]):
        gdir = workflow.init_glacier_directories('RGI60-11.00897',
                                                 from_prepro_level=1,
                                                 prepro_border=border)
        graphics.plot_domain(gdir, ax=ax, title='Border: {}'.format(border),
                             add_colorbar=False,
                             lonlat_contours_kwargs={'add_tick_labels':False})
    @savefig plot_border_size.png width=100%
    plt.tight_layout(); plt.show()


Users should choose the map border parameter depending
on the expected glacier growth in their simulations. For simulations into
the 21st century, a border value of 40 is
sufficient, but 80 is safer in case temperature is stabilizing or cooling in
certain regions / scenarios.
For runs into the Little Ice Age, a border value of 160 is recommended.

Users should be aware that the amount of data to download isn't small,
especially for full directories at processing level 3. Here is an indicative
table for the total amount of data with ERA5 centerlines for all 19 RGI regions:

======  =====  =====  =====  =====
Level   B  10  B  40  B  80  B 160
======  =====  =====  =====  =====
**L0**  927M   927M   927M   927M
**L1**  3.2G   7.3G   17G    47G
**L2**  11G    23G    51G    144G
**L3**  13G    26G    54G    147G
**L4**         3.5G   3.7G   4.1G
**L5**         7.2G   7.5G   8.3G
======  =====  =====  =====  =====

*L4 and L5 data are not available for border 10 (the domain is too small for
the downstream lines)*.

Certain regions are much smaller than others of course. As an indication,
with prepro level 3 and a map border of 160, the Alps are 2.1G large, Greenland
21G, and Iceland 664M.

Therefore, it is recommended to always pick the smallest border value suitable
for your research question, and to start your runs from level 5 if possible.

.. note::

  The data download of the preprocessed directories will occur one single time
  only: after the first download, the data will be cached in OGGM's
  ``dl_cache_dir`` folder (see :ref:`system-settings`).


Available pre-processed configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: **New in version 1.4!**

    OGGM now has several configurations and directories to choose from,
    and the list is getting larger. Don't hesitate to ask us if you are
    unsure about which to use, or if you'd like to have more configurations
    to choose from!

    To choose from a specific model configuration, use the ``prepro_base_url``
    argument in your call to :py:func:`workflow.init_glacier_directories`,
    and set it to one of the urls listed below.

    See `this tutorial <https://oggm.org/tutorials/notebooks/elevation_bands_vs_centerlines.html>`_
    for an example.


A. Default
^^^^^^^^^^

If not provided with a specific ``prepro_base_url`` argument,
:py:func:`workflow.init_glacier_directories` will download the glacier
directories from the default urls. Here is a summary of the default configuration:

- model parameters as of the ``oggm/params.cfg`` file at the published model version
- flowline glaciers computed from the geometrical centerlines (including tributaries)
- baseline climate from CRU (not available for Antarctica)
- baseline climate quality checked with :py:func:`tasks.historical_climate_qc` with ``N=3``
- mass-balance parameters calibrated with the standard OGGM procedure. No calibration
  against geodetic MB (see options below for regional calibration)
- ice volume inversion calibrated to match the ice volume from [Farinotti_etal_2019]_
  **at the RGI region level**, i.e. glacier estimates might differ. If not specified otherwise,
  it's also the precalibrated paramaters that will be used for the dynamical run.
- frontal ablation by calving (at inversion and for the dynamical runs) is switched off

To see the code that generated these directories (for example if you want to
make your own, visit :py:func:`cli.prepro_levels.run_prepro_levels`
or this `file on github <https://github.com/OGGM/oggm/blob/master/oggm/cli/prepro_levels.py>`_).

The urls used by OGGM per default are in the following ftp servor:

`https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/ <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/>`_ :

- `L1-L2_files/centerlines <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/centerlines/>`_ for level 1 and level 2
- `L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match/>`_ for level 3 to 5

If you are new to this, we recommend to explore these directories to familiarize yourself
to their content. Of course, when provided with an url such as above,
OGGM will know where to find the respective files
automatically, but is is good to understand how they are structured. The `summary folder`
(`example <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/centerlines/RGI62/b_080/L2/summary/>`_)
contains diagnostic files which can be useful as well.


B. Option: Geometrical centerlines or elevation band flowlines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The type of flowline to use (see :ref:`flowlines`) can be decided at level 2 already.
Therefore, the two configurations available at level 2 from these urls:

- `L1-L2_files/centerlines <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/centerlines/>`_ for centerlines
- `L1-L2_files/elev_bands <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/>`_ for elevation bands

The default pre-processing set-ups are also available with each of these
flowline types. For example with CRU:

- `L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match/>`_ for centerlines
- `L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match/>`_ for elevation bands

C. Option: Baseline climate data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the two most important default configurations (CRU or ERA5 as baseline climate),
we provide all levels for both the geometrical centerlines or the elevation band
flowlines:

- `L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/centerlines/qc3/pcp2.5/no_match/>`_ for CRU + centerlines
- `L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match/>`_ for CRU + elevation bands
- `L3-L5_files/ERA5/centerlines/qc3/pcp1.6/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/centerlines/qc3/pcp1.6/no_match/>`_ for ERA5 + centerlines
- `L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/no_match/>`_ for ERA5 + elevation bands

D. Further set-ups
^^^^^^^^^^^^^^^^^^

Here is the current list of available configurations at the time of writing (explore the server for more!):

- `L3-L5_files/CERA+ERA5/elev_bands/qc3/pcp1.6/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CERA+ERA5/elev_bands/qc3/pcp1.6/no_match/>`_ for CERA+ERA5 + elevation bands
- `L3-L5_files/CERA+ERA5/elev_bands/qc3/pcp1.6/match_geod <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CERA+ERA5/elev_bands/qc3/pcp1.6/match_geod/>`_ for CERA+ERA5 + elevation bands + matched on regional geodetic mass-balances
- `L3-L5_files/CRU/elev_bands/qc3/pcp2.5/match_geod <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc3/pcp2.5/match_geod/>`_ for CRU + elevation bands + matched on regional geodetic mass-balances
- `L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/match_geod <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/match_geod/>`_ for ERA5 + elevation bands + matched on regional geodetic mass-balances
- `L3-L5_files/ERA5/elev_bands/qc3/pcp1.8/match_geod <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.8/match_geod/>`_ for ERA5 + elevation bands flowlines + matched on regional geodetic mass-balances + precipitation factor 1.8
- `L3-L5_files/CRU/elev_bands/qc0/pcp2.5/match_geod <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc0/pcp2.5/match_geod/>`_ for CRU + elevation bands flowlines + matched on regional geodetic mass-balances + no climate quality check
- `L3-L5_files/CRU/elev_bands/qc0/pcp2.5/no_match <https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc0/pcp2.5/no_match/>`_ for CRU + elevation bands flowlines + no climate quality check

Note: the additional set-ups might not always have all map sizes available. Please
get in touch if you have interest in a specific set-up.

RGI-TOPO
--------

The `RGI-TOPO <https://rgitools.readthedocs.io/en/latest/dems.html>`_ dataset
provides a local topography map for each single glacier in the RGI. It was
generated with OGGM, and can be used very easily from the OGGM-Shop (visit
our `tutorials`_ if you are interested!).

    .. figure:: _static/malaspina_topo.png
        :width: 100%

        Example of the various RGI-TOPO products at Malaspina glacier

.. _tutorials: https://oggm.org/tutorials

ITS_LIVE
--------

The `ITS_LIVE <https://its-live.jpl.nasa.gov/>`_ ice velocity products
can be downloaded and reprojected to the glacier directory
(visit our `tutorials`_ if you are interested!).

    .. figure:: _static/malaspina_itslive.png
        :width: 80%

        Example of the reprojected ITS_LIVE products at Malaspina glacier

The data source used is https://its-live.jpl.nasa.gov/#data
Currently the only data downloaded is the 120m composite for both
(u, v) and their uncertainty. The composite is computed from the
1985 to 2018 average.

If you want more velocity products, feel free to open a new topic
on the OGGM issue tracker!

Ice thickness
-------------

The `Farinotti et al., 2019 <https://www.nature.com/articles/s41561-019-0300-3>`_
ice thickness products can be downloaded and reprojected to the glacier directory
(visit our `tutorials`_ if you are interested!).

    .. figure:: _static/malaspina_thick.png
        :width: 80%

        Example of the reprojected ice thickness products at Malaspina glacier

Raw data sources
----------------

These data are used to create the pre-processed directories explained above.
If you want to run your own workflow from A to Z, or if you would like
to know which data are used in OGGM, read further!

.. _outlines:

Glacier outlines and intersects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Glacier outlines are obtained from the `Randolph Glacier Inventory (RGI)`_.
We recommend to download them right away by opening a python interpreter
and type:

.. code-block:: python

    from oggm import cfg, utils
    cfg.initialize()
    utils.get_rgi_intersects_dir()
    utils.get_rgi_dir()

The RGI folders should now contain the glacier outlines in the
`shapefile format <https://en.wikipedia.org/wiki/Shapefile>`_, a format widely
used in GIS applications. These files can be read by several softwares
(e.g. `qgis <https://www.qgis.org/en/site/>`_), and OGGM can read them too.

The "RGI Intersects" shapefiles contain the locations of the ice divides
(intersections between neighboring glaciers). OGGM can make use of them to
determine which bed shape should be used (rectangular or parabolic). See the
`rgi tools <https://rgitools.readthedocs.io/en/latest/tools.html#glacier-intersects>`_
documentation for more information about the intersects.

The following table summarizes the RGI attributes used by OGGM. It
can be useful to refer to this list if you use your own glacier outlines
with OGGM.

==================  ===========================  ======================
RGI attribute       Equivalent OGGM variable     Comments
==================  ===========================  ======================
RGIId               ``gdir.rgi_id``              [#f1]_
GLIMSId             ``gdir.glims_id``            not used
CenLon              ``gdir.cenlon``              [#f2]_
CenLat              ``gdir.cenlat``              [#f2]_
O1Region            ``gdir.rgi_region``          not used
O2Region            ``gdir.rgi_subregion``       not used
Name                ``gdir.name``                used for graphics only
BgnDate             ``gdir.rgi_date``            [#f3]_
Form                ``gdir.glacier_type``        [#f4]_
TermType            ``gdir.terminus_type``       [#f5]_
Status              ``gdir.status``              [#f6]_
Area                ``gdir.rgi_area_km2``        [#f7]_
Zmin                ``glacier_statistics.csv``   recomputed by OGGM
Zmax                ``glacier_statistics.csv``   recomputed by OGGM
Zmed                ``glacier_statistics.csv``   recomputed by OGGM
Slope               ``glacier_statistics.csv``   recomputed by OGGM
Aspect              ``glacier_statistics.csv``   recomputed by OGGM
Lmax                ``glacier_statistics.csv``   recomputed by OGGM
Connect             not included
Surging             not included
Linkages            not included
EndDate             not included
==================  ===========================  ======================

For Greenland and Antarctica peripheral glaciers, OGGM does not take into account the
connectivity level between the Glaciers and the Ice sheets.
We recommend to the users to think about this before they
run the task: ``workflow.init_glacier_directories``.

.. _Randolph Glacier Inventory (RGI): https://www.glims.org/RGI/

.. rubric:: Comments

.. [#f1] The RGI id needs to be unique for each entity. It should resemble the
         RGI, but can have longer ids. Here are example of valid IDs:
         ``RGI60-11.00897``, ``RGI60-11.00897a``, ``RGI60-11.00897_d01``.
.. [#f2] ``CenLon`` and ``CenLat`` are used to center the glacier local map and DEM.
.. [#f3] The date is the acquisition year, stored as an integer.
.. [#f4] Glacier type: ``'Glacier'``, ``'Ice cap'``, ``'Perennial snowfield'``,
         ``'Seasonal snowfield'``, ``'Not assigned'``. Ice caps are treated
         differently than glaciers in OGGM: we force use a single flowline
         instead of multiple ones.
.. [#f5] Terminus type: ``'Land-terminating'``, ``'Marine-terminating'``,
         ``'Lake-terminating'``, ``'Dry calving'``, ``'Regenerated'``,
         ``'Shelf-terminating'``, ``'Not assigned'``. Marine and Lake
         terminating are classified as "tidewater" in OGGM and cannot advance
         - they "calve" instead, using a very simple parameterisation.
.. [#f6] Glacier status: ``'Glacier or ice cap'``, ``'Glacier complex'``,
         ``'Nominal glacier'``, ``'Not assigned'``. Nominal glaciers fail at
         the "Glacier Mask" processing step in OGGM.
.. [#f7] The area of OGGM's flowline glaciers is corrected to the one provided
         by the RGI, for area conservation and inter-comparison reasons. If
         you do not want to use the RGI area but the one computed from the
         shape geometry in the local OGGM map projection instead, set
         ``cfg.PARAMS['use_rgi_area']`` to ``False``. This is useful when
         using homemade inventories.


Topography data
~~~~~~~~~~~~~~~

When creating a :ref:`glacierdir`, a suitable topographical data source is
chosen automatically, depending on the glacier's location. OGGM supports
a large number of datasets (almost all of the freely available ones, we
hope). They are listed on the
`RGI-TOPO <https://rgitools.readthedocs.io/en/latest/dems.html>`_ website.

The current default is to use the following datasets:

- NASADEM: 60°S-60°N
- COPDEM: Global, with missing regions (islands, etc.)
- GIMP, REMA: Regional datasets
- TANDEM: Global, with artefacts / missing data
- MAPZEN: Global, when all other things failed

These data are chosen in the provided order. If a dataset is not available,
the next on the list will be tested: if the tested dataset covers
75% of the glacier area, it is selected. In practice, NASADEM and COPDEM
are sufficient for all but about 300 of the world's glaciers.

These data are downloaded only when needed (i.e. during an OGGM run)
and they are stored in the ``dl_cache_dir``
directory. The gridded topography is then reprojected and resampled to the local
glacier map. The local grid is defined on a Transverse Mercator projection centered over
the glacier, and has a spatial resolution depending on the glacier size. The
default in OGGM is to use the following rule:

.. math::

    \Delta x = d_1 \sqrt{S} + d_2

where :math:`\Delta x` is the grid spatial resolution (in m), :math:`S` the
glacier area (in km\ :math:`^{2}`) and :math:`d_1`, :math:`d_2` some parameters (set to 14 and 10,
respectively). If the chosen spatial resolution is larger than 200 m
(:math:`S \ge` 185 km\ :math:`^{2}`) we clip it to this value.


.. ipython:: python
   :suppress:

    import json
    from oggm.utils import get_demo_file
    with open(get_demo_file('dem_sources.json'), 'r') as fr:
        DEM_SOURCE_INFO = json.loads(fr.read())
    # for k, v in DEM_SOURCE_INFO.items():
    #   print(v)

**Important:** when using these data sources for your OGGM runs, please refer
to the original data provider of the data! OGGM adds a ``dem_source.txt``
file in each glacier directory specifying how to cite these data. We
reproduce this information
`here <https://github.com/OGGM/oggm/blob/master/oggm/data/dem_sources.txt>`_.

.. warning::

    A number of glaciers will still suffer from poor topographic information.
    Either the errors are large or obvious (in which case the model won't run),
    or they are left unnoticed. The importance of reliable topographic data for
    global glacier modelling cannot be emphasized enough, and it is a pity
    that no consistent, global DEM is yet available for scientific use.
    Visit `rgitools <https://rgitools.readthedocs.io/en/latest/dems.html>`_
    for a discussion about our current efforts to find "the best" DEMs.

.. note::

    `In this blogpost <https://oggm.org/2019/10/08/dems/>`_ we talk about which
    requirements a DEM must fulfill to be helpful to OGGM. And we also explain
    why and how we preprocess some DEMs before we make them available to the
    OGGM workflow.

Climate data
~~~~~~~~~~~~

The mass-balance model implemented in OGGM needs monthly time series of temperature and
precipitation. The current default is to download and use the `CRU TS`_
data provided by the Climatic Research Unit of the University of East Anglia.

.. _CRU TS: https://crudata.uea.ac.uk/cru/data/hrg/


**‣ CRU (default)**

If not specified otherwise, OGGM will automatically download and unpack the
latest dataset from the CRU servers. To download them you can use the
following convenience functions:

.. code-block:: python

    from oggm.shop import cru
    cru.get_cl_file()
    cru.get_cru_file(var='tmp')
    cru.get_cru_file(var='pre')


.. warning::

    While each downloaded zip file is ~200mb in size, they are ~2.9Gb large
    after decompression!

The raw, coarse dataset (CRU TS v4.04 at 0.5° resolution) is then downscaled to
a higher resolution grid (CRU CL v2.0 at 10' resolution) following the anomaly mapping approach
described by Tim Mitchell in his `CRU faq`_ (Q25). Note that we don't expect
this downscaling to add any new information than already available at the
original resolution, but this allows us to have an elevation-dependent dataset
based on a presumably better climatology. The monthly anomalies are computed
following Harris et al., (2010): we use standard anomalies for temperature and
scaled (fractional) anomalies for precipitation. At the locations where the
monthly precipitation climatology is 0 we fall back to the standard anomalies.

**When using these data, please refer to the original provider:**

Harris, I., Jones, P. D., Osborn, T. J., & Lister, D. H. (2014). Updated
high-resolution grids of monthly climatic observations - the CRU TS3.10 Dataset.
International Journal of Climatology, 34(3), 623–642. https://doi.org/10.1002/joc.3711

.. _CRU faq: https://crudata.uea.ac.uk/~timm/grid/faq.html

**‣ ECMWF (ERA5, CERA, ERA5-Land)**

The data from ECMWF are used "as is", i.e. without any further downscaling.
We propose several datasets (see :py:func:`oggm.shop.ecmwf.process_ecmwf_data`)
and, with the task :py:func:`oggm.tasks.historical_delta_method`, also
allow for combinations of them.

**When using these data, please refer to the original provider:**

For example for ERA5:

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A.,
Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I.,
Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2019):
ERA5 monthly averaged data on single levels from 1979 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
(Accessed on < 01-12-2020 >), 10.24381/cds.f17050d7


**‣ User-provided dataset**

You can provide any other dataset to OGGM by setting the ``climate_file``
parameter in ``params.cfg``. See the HISTALP data file in the `sample-data`_
folder for an example.

.. _sample-data: https://github.com/OGGM/oggm-sample-data/tree/master/test-workflow

**‣ GCM data**

OGGM can also use climate model output to drive the mass-balance model. In
this case we still rely on gridded observations (CRU) for the baseline
climatology and apply the GCM anomalies computed from a preselected reference
period. This method is sometimes called the
`delta method <http://www.ciesin.org/documents/Downscaling_CLEARED_000.pdf>`_.

Currently we can process data from the
`CESM Last Millenium Ensemble <http://www.cesm.ucar.edu/projects/community-projects/LME/>`_
project (see :py:func:`tasks.process_cesm_data`), and CMIP5/CMIP6
(:py:func:`tasks.process_cmip_data`).


Mass-balance data
~~~~~~~~~~~~~~~~~

In-situ mass-balance data are used by OGGM to calibrate and validate the
mass-balance model. We rely on mass-balance observations provided by the
World Glacier Monitoring Service (`WGMS`_).
The `Fluctuations of Glaciers (FoG)`_ database contains annual mass-balance
values for several hundreds of glaciers worldwide. We exclude water-terminating
glaciers and the time series with less than five years of
data.
Since 2017, the WGMS provides a lookup table
linking the RGI and the WGMS databases. We updated this list for version 6 of
the RGI, leaving us with 268 mass balance time series. These are not equally
reparted over the globe:

.. figure:: _static/wgms_rgi_map.png
    :width: 100%

    Map of the RGI regions; the red dots indicate the glacier locations
    and the blue circles the location of the 254 reference WGMS
    glaciers used by the OGGM calibration. From our `GMD paper`_.

These data are shipped automatically with OGGM. All reference glaciers
have access to the timeseries through the glacier directory:


.. ipython:: python

    gdir = workflow.init_glacier_directories('RGI60-11.00897',
                                             from_prepro_level=3,
                                             prepro_border=10)[0]
    mb = gdir.get_ref_mb_data()
    @savefig plot_ref_mbdata.png width=100%
    mb[['ANNUAL_BALANCE']].plot(title='WGMS data: Hintereisferner')


.. _WGMS: https://wgms.ch
.. _Fluctuations of Glaciers (FoG): https://wgms.ch/data_databaseversions/
.. _GMD Paper: https://www.geosci-model-dev.net/12/909/2019/
