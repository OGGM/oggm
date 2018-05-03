rgitools
========


`rgitools <https://github.com/OGGM/rgitools>`_ provides several tools to
(pre-)process and analyse the glacier outlines provided by the
`Randolph Glacier Inventory <https://www.glims.org/RGI/>`_ (RGI).

It is currently in development, but our goal is to provide the following
sevices to the RGI working group:

- **automated data quality check**: see :ref:`rgitools.qc`.
- **ID link list between RGI versions**: the RGI glacier identifiers might
  change from one version to another. If the geometry didn't change, it is
  possible to automatically identify the new ID of a glacier in the new RGI
  version.
- **Geometry of the ice divides and listing or glacier clusters**:
  see :ref:`rgitools.intersects`.
- **Merging of glacier clusters**: see :ref:`rgitools.merge`.
- **Glacier hypsometry**: continuing the work done by Matthias Huss for
  previous RGI version, rgitools relies on OGGM to provide the glacier
  hypsometries and topography data for each single glacier in the RGI.


Package structure
-----------------

The package is structured in two modules:

- **rgitools.funcs**, which contains standalone functions acting on one single
  shapefile, and
- **rgitools.scripts**, which contains scripts acting on an RGI directory and
  calling ``rgitools.funcs`` in a multi-processing loop.

The examples below make only use of ``rgitools.funcs``, applying them
to a subset of the RGI region 06, Iceland.

.. _rgitools.qc:

Data quality check
------------------

``rgitools.funcs.check_geometries`` loops over each glacier and makes to
data quality checks:

MultiPolygons
~~~~~~~~~~~~~

A glacier geometry should be of type `Polygon <https://toblerity.org/shapely/manual.html#polygons>`_.
If not, it is usually  a `MultiPolygon <https://toblerity.org/shapely/manual.html#MultiPolygon>`_.
In this case the tool makes a new check:

- a MultiPolygon could be a list of the outline and one or more nunataks.
  In this case, the geometry is corrected to a valid Polygon with
  interior geometries. This was the case for all glaciers in RGIV4 but does
  not seem to happen since RGIV5
- a MultiPolygon results from self intersecting glacier geometries (e.g. a
  geometry in the form of the number "8". In theory, this should result in
  two separate glacier entities. In practice, this happens for few glaciers,
  and often one part of the "8" is very small: the tool will cut these out
  and raise an error if the geometry that was cut is too large.

Invalid geometries
~~~~~~~~~~~~~~~~~~

Some geometries might be invalid in the sense of standards defined by the
`Open Geospatial Consortium <http://www.opengeospatial.org/standards/sfa>`_.
When possible, the tool corrects these geometries using the
`geom.buffer(0) <https://toblerity.org/shapely/manual.html#object.buffer>`_
method provided by shapely.


The function is simple to use. The output shapefile contains a new column
(``check_geom``) which contains information on the provided check. If empty,
it means that no correctino was applied. If not, a message tells you what
happened:

.. ipython:: python

    import geopandas as gpd
    from rgitools.funcs import get_demo_file, check_geometries

    df = gpd.read_file(get_demo_file('RGI6_icecap.shp'))
    df = check_geometries(df)
    df.loc[df.check_geom != ''][['RGIId', 'check_geom']]

``WARN:WasInvalid`` means that the geometry was invalid but is now corrected.


.. _rgitools.intersects:

Glacier intersects
------------------

The RGI doesn't provide information on **where** a glacier entity has
neighbors and **which** glaciers are connected. This is what the
``rgitools.funcs.compute_intersects`` function is for:

.. ipython:: python

    import matplotlib.pyplot as plt
    from rgitools.funcs import compute_intersects
    dfi = compute_intersects(df)

    f, ax = plt.subplots(figsize=(6, 4))
    df.plot(ax=ax, edgecolor='k');
    @savefig plot_intersects.png width=100%
    dfi.plot(ax=ax, edgecolor='C3');

The intersects shapefile contains the divide geometries (in red on the plot)
and the ID of the two glaciers it is the divide for:


.. ipython:: python

    dfi.iloc[:5][['RGIId_1', 'RGIId_2']]


This information is then used by the ``rgitools.funcs.find_clusters`` function
to detect the connected entities:

.. ipython:: python

    from rgitools.funcs import find_clusters
    clusters = find_clusters(dfi)

    for i, (k, c) in enumerate(clusters.items()):
        df.loc[df.RGIId.isin(c), 'cluster_id'] = i+1

    f, ax = plt.subplots(figsize=(6, 4))
    @savefig plot_clusters.png width=100%
    df.plot(ax=ax, column='cluster_id', edgecolor='k', cmap='Set2');

This function returns a dictionary containg the list of identifiers for
each cluster.

.. _rgitools.merge:

Merging glacier clusters
------------------------

If needed, you can merge each of these clusters into a single polygon:

.. ipython:: python

    from rgitools.funcs import merge_clusters
    df_merged = merge_clusters(df, dfi)

    f, ax = plt.subplots(figsize=(6, 4))
    @savefig plot_merged.png width=100%
    df_merged.plot(ax=ax, column='cluster_id', edgecolor='k', cmap='Set2');


More tools
----------

More tools are coming soon! Stay tuned...
