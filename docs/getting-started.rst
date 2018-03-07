.. currentmodule:: oggm

.. _getting-started:

Getting started
===============

.. ipython:: python
   :suppress:

    import os
    import numpy as np
    np.set_printoptions(threshold=10)

The ultimate goal of OGGM will be to hide the python workflow behind the model
entirely, and run it only using configuration files and scripts. We are not
there yet, and if you want to use and participate to the development of OGGM
you'll have to get your hands dirty. We hope however that the workflow is
structured enough so that it is possible to jump in without having to
understand all of its internals.

The few examples below are meant to illustrate the general design of OGGM,
without going into the details of the implementation.

Imports
-------

The following imports are necessary for all of the examples:

.. ipython:: python

    import geopandas as gpd
    import oggm
    from oggm import cfg, tasks, graphics
    from oggm.utils import get_demo_file

Initialisation and GlacierDirectories
-------------------------------------

The first thing to do when running OGGM is to initialise it. This function
will read the `default configuration file <https://github.com/OGGM/oggm/blob/master/oggm/params.cfg>`_
which contains all user defined parameters:

.. ipython:: python

    cfg.initialize()

These parameters are now accessible to all OGGM routines. For example, the
``cfg.PARAMS`` dict contains some runtime parameters, while ``cfg.PATHS`` stores
the paths to the input files and the working directory (where the model output
will be written):

.. ipython:: python

    cfg.PARAMS['topo_interp']
    cfg.PARAMS['temp_default_gradient']
    cfg.PATHS

We'll use some demo files
(`shipped with OGGM <https://github.com/OGGM/oggm-sample-data>`_) for the basic
input:

.. ipython:: python

    cfg.PATHS['working_dir'] = oggm.gettempdir('oggm_wd')  # working directory
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')  # topography
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))  # intersects

The starting point of a run is always a valid `RGI <http://www.glims.org/RGI/>`_ file.
In this case we use a very small subset of the RGI, the outlines of the
`Hinereisferner <http://acinn.uibk.ac.at/research/ice-and-climate/projects/hef>`_
(HEF) in the Austrian Alps:

.. ipython:: python

    entity = gpd.GeoDataFrame.from_file(get_demo_file('HEF_MajDivide.shp')).iloc[0]
    entity

This information is enough to define HEF's :py:class:`GlacierDirectory`:

.. ipython:: python

    gdir = oggm.GlacierDirectory(entity)

:ref:`glacierdir` have two major purposes:

  - carry information about the glacier attributes
  - handle I/O and filepaths operations

For example, it will tell OGGM where to write the data for this glacier or
its terminus type:

.. ipython:: python

    gdir.dir
    gdir.terminus_type

GlacierDirectories are the input to most OGGM functions. In fact, they are the
only required input to all :ref:`apientitytasks`. These entity tasks are
processes which can run on one glacier at a time (the vast majority of OGGM
tasks are entity tasks). The first task to apply to an empty GlacierDirectory
is :py:func:`tasks.define_glacier_region`, which sets the local glacier map
and topography, and :py:func:`tasks.glacier_masks`, which prepares gridded
topography data:

.. ipython:: python

    tasks.define_glacier_region(gdir, entity=entity)
    tasks.glacier_masks(gdir)
    os.listdir(gdir.dir)

The directory is now filled with data. Other tasks can build upon these, for
example the plotting functions:

.. ipython:: python

    @savefig plot_domain.png width=80%
    graphics.plot_domain(gdir)

What next?
----------

This documentation is growing step by step. In the meantime, a good place
to start is the ``oggm/docs/notebooks`` directory.

You will find several notebooks:

- ``getting_started.ipynb``, which set-ups an entire OGGM run
  in the Ötztal region.
- ``flowline_model.ipynb``, which describes the usage of the flowline model
  for idealized glaciers.
- ``flowline_with_known_bedrock.ipynb``, which describes the usage of the
  flowline model with custom boundary conditions.
- ``specmb_vs_ela.ipynb``, which was used to make the analyses presented in
  `this blog post <http://oggm.org/2017/10/01/specmb-ela/>`_
- ``dynamics_and_length_changes.ipynb``, which was used to make the analyses
  presented in `this blog post <http://oggm.org/2017/10/23/length-changes/>`_
