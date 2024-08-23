.. currentmodule:: oggm

Getting started
===============

.. ipython:: python
   :suppress:

    import os
    import numpy as np
    np.set_printoptions(threshold=10)

Although largely automated, OGGM model still requires some python
scripting to prepare and run a simulation. This section will guide you
through several examples to get you started.

.. important::

   Did you know that you can try OGGM in your browser before installing it
   on your computer? Visit :doc:`cloud` for more information.

   Did you know that we provide standard projections for all glaciers in the
   world, and that you may not have to run OGGM for your use case?
   Visit :doc:`download-projections` for more information.

.. _system-settings:

First step: system settings for input data
------------------------------------------

OGGM needs various input data files to run. Currently, **we rely exclusively on
open-access data that are all downloaded automatically for the user**.
OGGM implements a bunch of tools to make access to input data as painless
as possible for you, including the automated download of all the required files.
This requires you to tell OGGM where to store these data.
Let's start by opening a python interpreter and type:

.. ipython:: python

    from oggm import cfg
    cfg.initialize()

At your very first import, this will do two things:

1. It will download a small subset of data used for testing and calibration.
   Those data are located in your home directory, in a hidden folder
   called ``.oggm``.
2. It will create a configuration file in your home folder, where you can
   indicate where you want to store further input data. This configuration
   file is also located in your home directory under the name ``.oggm_config``.

To locate this config file, you can type:

.. ipython:: python

    cfg.CONFIG_FILE

.. important::

    The default settings will probably work for you, but we recommend to have
    a look at this file and set the paths to a directory
    where enough space is available: a minimum of 8 Gb for all climate data
    and glacier outlines is necessary. Pre-processed glacier directories can
    quickly grow to several tens of Gb as well, even for regional runs.


Calibration data and testing: the ``~/.oggm`` directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the first import, OGGM will create a cached ``.oggm`` directory in your
``$HOME`` folder. This directory contains all data obtained from the
`oggm sample data`_ repository. It contains several files needed only for
testing, but also some important files needed for calibration and validation
(e.g. the `reference mass balance data`_ from WGMS with
`links to the respective RGI polygons`_).

.. _oggm sample data: https://github.com/OGGM/oggm-sample-data
.. _reference mass balance data: https://github.com/OGGM/oggm-sample-data/tree/master/wgms
.. _links to the respective RGI polygons: http://fabienmaussion.info/2017/02/19/wgms-rgi-links/

The ``~/.oggm`` directory should be updated automatically when you update OGGM,
but if you encounter any problems with it, simply delete the directory (it will
be re-downloaded automatically at the next import).

.. _oggm-config:

All other data: auto-downloads and the ``~/.oggm_config`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike runtime parameters (such as physical constants or working directories),
input data are shared across runs and even across computers if you want
to. Therefore, the paths to previously downloaded data are stored in a
configuration file that you'll find in your ``$HOME`` folder:
the ``~/.oggm_config`` file.

The file should look like::

    dl_cache_dir = /path/to/download_cache
    dl_cache_readonly = False
    tmp_dir = /path/to/tmp_dir
    rgi_dir = /path/to/rgi_dir
    test_dir = /path/to/test_dir
    has_internet = True

Some explanations:

- ``dl_cache_dir`` is a path to a directory where *all* the files you
  downloaded will be cached for later use. Most of the users won't need to
  explore this folder (it is organized as a list of urls) but you have to make
  sure to set this path to a folder with sufficient disk space available. This
  folder can be shared across compute nodes if needed (it is even recommended
  for HPC setups). Once a file is stored in this cache folder (e.g. a specific
  DEM tile), OGGM won't download it again.
- ``dl_cache_readonly`` indicates if writing is allowed in this folder (this is
  the default). Setting this to ``True`` will prevent any further download in
  this directory (useful for cluster environments, where this data might be
  available on a readonly folder): in this case, OGGM will use a fall back
  directory in your current working directory.
- ``tmp_dir`` is a path to OGGM's temporary directory. Most of the
  files used by OGGM are downloaded and cached in a compressed format (zip,
  bz, gz...).
  These files are extracted in ``tmp_dir`` before use. OGGM will never allow more
  than 100 ``.tif`` (or 100 ``.nc``) files to exist in this directory by
  deleting the oldest ones
  following the rule of the `Least Recently Used (LRU)`_ item. Nevertheless,
  this directory might still grow to quite a large size. Simply delete it
  if you want to get this space back.
- ``rgi_dir`` is the location where the RGI shapefiles are extracted.
- ``test_dir`` is the location where OGGM will write some of its output during
  tests. It can be set to ``tmp_dir`` if you want to, but it can also be
  another directory (for example a fast SSD disk). This folder shouldn't take
  too much disk space but here again, don't hesitate to delete it if you need to.

.. note::

  For advanced users or cluster configuration: the user's
  ``tmp_dir`` and ``rgi_dir`` settings can be overridden and set to a
  specific directory by defining an environment variable ``OGGM_EXTRACT_DIR``
  to a directory path. Similarly, the environment variables
  ``OGGM_DOWNLOAD_CACHE`` and ``OGGM_DOWNLOAD_CACHE_RO`` override the
  ``dl_cache_dir`` and ``dl_cache_readonly`` settings.

.. _Least Recently Used (LRU): https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)


OGGM tutorials
--------------

Refer to our `tutorials`_ for real-world applications!

.. _tutorials: https://tutorials.oggm.org
