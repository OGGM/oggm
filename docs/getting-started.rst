.. currentmodule:: oggm

.. _getting-started:

Getting started
===============


.. ipython:: python
   :suppress:

    import numpy as np
    np.set_printoptions(threshold=10)

    # try download a couple of times
    from oggm.utils import get_demo_file
    try:
        get_demo_file('Hintereisferner.shp')
    except:
        pass
    try:
        get_demo_file('Hintereisferner.shp')
    except:
        pass
    try:
        get_demo_file('Hintereisferner.shp')
    except:
        pass

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
which contains all user defined parameters.

.. ipython:: python

    cfg.initialize()

These parameters are now accessible to all OGGM routines. For example, the
`cfg.PARAMS` dict contains some runtime parameters, while `cfg.PATHS` stores
the paths to the input files and the working directory (where the model output
will be written):

.. ipython:: python

    cfg.PARAMS['topo_interp']
    cfg.PARAMS['temp_default_gradient']
    cfg.PARAMS['border'] = 20  # set this parameter
    cfg.PATHS

We'll use some demo files
(`shipped with OGGM <https://github.com/OGGM/oggm-sample-data>`_) for the basic
input:

.. ipython:: python

    cfg.set_divides_db(get_demo_file('divides_workflow.shp')) # glacier divides
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif') # topography

The starting point of a run is always a valid `RGI <http://www.glims.org/RGI/>`_ file.
In this case we use a very small subset of the RGI, the outlines of the
`Hinereisferner <http://acinn.uibk.ac.at/research/ice-and-climate/projects/hef>`_
(HEF) in the Austrian Alps:

.. ipython:: python

    entity = gpd.GeoDataFrame.from_file(get_demo_file('Hintereisferner.shp')).iloc[0]
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


.. ipython:: python

    tasks.define_glacier_region(gdir, entity=entity)
    tasks.glacier_masks(gdir)

And this makes a plot:

.. ipython:: python

    @savefig plot_domain.png
    graphics.plot_domain(gdir)

Do it yourself
--------------

A good place to start is the ``oggm/docs/notebooks`` directory. You will find
two notebooks:

- ``getting_started.ipynb``, which set-ups an entire OGGM run
  in the Ötztal region.
- ``flowline_model.ipynb``, which describes the usage of the flowline model
  for idealized test cases.
