.. currentmodule:: oggm

Other data sources
==================

If you want to run your own workflow from A to Z, or if you would like
to know which data are used in OGGM before being available in the
pre-processed directories, read further for some more details!

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
used in GIS applications. These files can be read by several software
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
         - they "calve" instead, using a very simple parameterization.
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

The current default is to use COPDEM90 and COPDEM30. In practice, COPDEM
is sufficient for all but about 300 of the world's glaciers. The we use
some other datasets available in RGITOPO.

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
    global glacier modelling cannot be emphasized enough.

.. note::

    `In this blogpost <https://oggm.org/2019/10/08/dems/>`_ we talk about which
    requirements a DEM must fulfill to be helpful to OGGM. And we also explain
    why and how we preprocess some DEMs before we make them available to the
    OGGM workflow.
