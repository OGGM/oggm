.. _shop:

.. currentmodule:: oggm

OGGM Shop
=========

    .. figure:: _static/logos/logo_shop.png
        :width: 100%

OGGM needs various data files to run. **We rely exclusively on
open-access data that can be downloaded automatically for the user**. We like
to see this service as a "shop", allowing users to define a "shopping list"
of data that users can add to their :ref:`glacierdir`.

This page describes the various products you will find ind this shop.

.. important::

    Don't forget to set-up or check your :ref:`system-settings` before
    downloading new data! (you'll need to
    do this only once per computer)

.. _preprodir:

Pre-processed directories
-------------------------

The simplest way to run OGGM is to rely on :ref:`glacierdir` which have been
prepared for you by the OGGM developers. Depending on your use case,
you can start from various stages in the processing chain, various map sizes,
and various model set-ups.

The default directories have been generated with the default parameters
of the current stable OGGM version and combinations hereof. If you want to
change some of these parameters, you *may* have to start a run from a lower
processing level. Whether or not this is necessary depends on the
stage of the workflow you'd like your computations to diverge from the
defaults (we provide example use cases below).

To start from a pre-processed state, simply use the
:py:func:`workflow.init_glacier_directories` function with the
``from_prepro_level`` and ``prepro_border`` keyword arguments set to the
values of your choice. This will fetch the default directories: there are
more options to that, which we explain below.

Processing levels
~~~~~~~~~~~~~~~~~

Currently, there are five available levels of pre-processing:

- **Level 0**: the lowest level, with directories containing the glacier
  outlines only.
- **Level 1**: directories now contain the glacier topography data as well.
- **Level 2**: at this stage, the flowlines and their downstream lines are
  computed and ready to be used.
- **Level 3**: adding the baseline climate timeseries (CRU or ERA5, see below)
  to the directories. Adding all necessary pre-processing tasks
  for a dynamical run, including the mass-balance calibration, bed inversion,
  etc. up to the `init_present_time_glacier` (included). These directories
  still contain all gridded data, i.e. they are the largest in size but the
  most flexible.
- **Level 4**: same as level 3 but with all intermediate ouptut files removed.
  The strong advantage of level 4 files is that their size is considerably
  reduced, at the cost that certain operations (like plotting on maps or
  running the bed inversion algorithm again) are not possible anymore.
- **Level 5**: on top of level 4, an additional historical simulation is run
  from the RGI date to the last possible date of the baseline climate file
  (for example, CRU in the Alps mean a 2003-2019 simulation for most glaciers).
  The state of the glacier as month 01, year 2020 can then be used for
  future projections.

In practice, most users are going to use level 2, level 3 or level 5 files. Here
are some example use cases:

1. *Running OGGM with the default settings and with GCM / RCM data*: start at level 5
2. *Using OGGM's flowlines but running your own flavor of the baseline climate,
   mass-balance or ice thickness inversion models*: start at level 2 (and maybe
   use OGGM's workflow again for the ice dynamics?).
3. *Run sensitivity experiments for the ice thickness inversion*: start at level
   3 (with climate data available) and re-run the inversion steps.
4. *Implement and test a new calving parameterisation*: start at level 3


Map size
~~~~~~~~

The size of the local glacier map is given in number of grid points *outside*
the glacier boundaries. The larger the map, the largest the glacier can
become. Therefore, user should choose the map border parameter depending
on the expected glacier growth in their simulations: for most cases,
a border value of 80 or 160 should be enough.

Here is an example with the Hintereisferner in the Alps:

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


For runs into the Little Ice Age, a border value of 160 is more than enough.
For simulations into the 21st century, a border value of 80 is
sufficient.

Users should be aware that the amount of data to download isn't small,
especially for full directories at levels 3. Here is an indicative
table for the total amount of data for all 18 RGI regions
(excluding Antarctica):

======  =====  =====  =====  =====
Level   B  10  B  80  B 160  B 250
======  =====  =====  =====  =====
**L1**  2.4G   11G    29G    63G
**L2**  5.1G   14G    32G    65G
**L3**  13G    44G    115G   244G
**L4**  4.2G   4.5G   4.8G   5.3G
======  =====  =====  =====  =====

Certain regions are much smaller than others of course. As an indication,
with prepro level 3 and a map border of 160, the Alps are 2G large, Greenland
11G, and Iceland 334M.

Therefore, it is recommended to always pick the smallest border value suitable
for your research question, and to start your runs from level 4 if possible.

.. note::

  The data download of the preprocessed directories will occur one single time
  only: after the first download, the data will be cached in OGGM's
  ``dl_cache_dir`` folder (see above).


Available pre-processed configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: **New in version 1.4!**

    OGGM now has several configurations and directories to choose from,
    and the list is getting larger. Don't hestitate to ask us if you think
    we should add more!



.. _rawdata:

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

The following table summarizes the attributes from the RGI used by OGGM. It
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

For Greenland and Antarctica, OGGM does not take into account the
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

When creating a :ref:`glacierdir` a suitable topographical data source is
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

These data are downloaded only when needed (i.e.: during an OGGM run)
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
    why and how we preprocess certain DEMs before we make them available to the
    OGGM workflow.

Climate data
~~~~~~~~~~~~

The MB model implemented in OGGM needs monthly time series of temperature and
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

The raw, coarse (0.5°) dataset is then downscaled to a higher resolution grid
(CRU CL v2.0 at 10' resolution) following the anomaly mapping approach
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
project (see :py:func:`tasks.process_cesm_data`), but adding other models
should be relatively easy.


Mass-balance data
~~~~~~~~~~~~~~~~~

In-situ mass-balance data is used by OGGM to calibrate and validate the
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
