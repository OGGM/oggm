Input data
==========

OGGM needs various data files to run. To date, we rely exclusively on
open-access data that are all downloaded automatically for the user. This
page explains the various ways OGGM uses to get the the data it needs.


Calibration data and testing: the ``~/.oggm`` directory
-------------------------------------------------------

At the first import, OGGM will create a cached ``.oggm`` directory in your
``$HOME`` folder. This directory contains all data obtained from the
`oggm sample data`_ repository. It contains various files needed only for
testing, but also some important files needed for calibration and validation.
For example:

- The CRU `baseline climatology`_ (CL v2.0, obtained from
  `crudata.uea.ac.uk/ <https://crudata.uea.ac.uk/cru/data/hrg/>`_ and prepared
  for OGGM),
- The `reference mass-balance data`_ from WGMS with
  `links to the respective RGI polygons`_,
- The `reference ice thickness data`_ from WGMS (`GlaThiDa`_ database).

.. _oggm sample data: https://github.com/OGGM/oggm-sample-data
.. _baseline climatology: https://github.com/OGGM/oggm-sample-data/tree/master/cru
.. _reference mass-balance data: https://github.com/OGGM/oggm-sample-data/tree/master/wgms
.. _links to the respective RGI polygons: http://fabienmaussion.info/2017/02/19/wgms-rgi-links/
.. _reference ice thickness data: https://github.com/OGGM/oggm-sample-data/tree/master/glathida
.. _GlaThiDa: http://www.gtn-g.ch/data_catalogue_glathida/

This directory should be updated automatically when new files are available
on GitHub, but if you encounter any problems simply delete it, it will be
re-downloaded automatically.


All other data: auto-downloads and the ``~/.oggm_config`` file
--------------------------------------------------------------

OGGM implements a bunch of logic to make access to the input data as painless
as possible, including the automated download of all the required files.

Unlike runtime parameters (such as physical constants or working directories),
the input data is shared accross runs and even computers. Therefore, the
paths to previously downloaded data are stored in a simple configuration file
that you'll find in your ``$HOME`` folder: the ``~/.oggm_config`` file.

The file should look like::

    dl_cache_dir = /path/to/download_cache
    dl_cache_readonly = False
    tmp_dir = /path/to/tmp_dir
    cru_dir = /path/to/cru_dir
    rgi_dir = /path/to/rgi_dir
    test_dir = /path/to/test_dir
    has_internet = True

With:

- ``dl_cache_dir`` is a path to a directory where *all* the files you
  downloaded will be cached for later use. Most of the users won't need to
  explore this folder (it is organized as a list of urls) but you have to make
  sure to set this path to a folder with sufficient disk space available.
- ``dl_cache_readonly`` indicates if writing is allowed in this folder (this is
  the default). Setting this to ``True`` will prevent any further download.
- ``tmp_dir`` is a path to OGGM's temporary directory. Most of the topography
  files used by OGGM are downloaded and cached in a compressed format. They
  will be extracted in ``tmp_dir`` before use. OGGM will never allow more than
  100 ``.tiff`` files to exist in this directory be deleting the oldest ones
  following the rule of the `Least Recently Used (LRU)`_ item. Nevertheless,
  this directory might still grow to quite a large size. Simply delete it
  if you want to get this space back.
- ``cru_dir`` is the location where the CRU climate files are extracted. They
  are quite large! (approx. 6Gb)
- ``rgi_dir`` is the location where the RGI shapefiles are extracted.
- ``test_dir`` is the location where OGGM will write some of its output during
  tests. It can be set to ``tmp_dir`` if you want to, but it can also be
  another directory (for example a fast SSD disk -- the data written for
  testing is much smaller than the input files).

.. _Least Recently Used (LRU): https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_.28LRU.29


More about the topography data
------------------------------

TODO