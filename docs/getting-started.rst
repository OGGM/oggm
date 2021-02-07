.. currentmodule:: oggm

.. _getting-started:

Getting started
===============

.. ipython:: python
   :suppress:

    import os
    import numpy as np
    np.set_printoptions(threshold=10)

Although largely automatised, the OGGM model still requires some python
scripting to prepare and run a simulation. The tutorials will guide you
through several examples to get you started.

.. important::

   Did you know that you can try OGGM in your browser before installing it
   on your computer? Visit :ref:`cloud` for more information.

.. _system-settings:

First step: system settings for input data
------------------------------------------

OGGM needs various data files to run. Currently, **we rely exclusively on
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
   This data is located in your home directory, in a hidden folder
   called `.oggm`.
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
(e.g. the `reference mass-balance data`_ from WGMS with
`links to the respective RGI polygons`_).

.. _oggm sample data: https://github.com/OGGM/oggm-sample-data
.. _reference mass-balance data: https://github.com/OGGM/oggm-sample-data/tree/master/wgms
.. _links to the respective RGI polygons: http://fabienmaussion.info/2017/02/19/wgms-rgi-links/

The ``~/.oggm`` directory should be updated automatically when you update OGGM,
but if you encounter any problems with it, simply delete the directory (it will
be re-downloaded automatically at the next import).

.. _oggm-config:

All other data: auto-downloads and the ``~/.oggm_config`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike runtime parameters (such as physical constants or working directories),
the input data is shared across runs and even across computers if you want
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

.. _Least Recently Used (LRU): https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_.28LRU.29


OGGM workflow
-------------

For a step by step tutorial of the entire OGGM workflow, download and run
the
:download:`getting started <https://raw.githubusercontent.com/OGGM/oggm-edu-notebooks/master/oggm-tuto/getting_started.ipynb>`
jupyter notebook (right-click -> "Save link as").

Alternatively, you can try OGGM directly in your browser without having
to install anything! Click on the button below:

.. image:: https://img.shields.io/badge/Launch-OGGM%20tutorials-579ACA.svg?style=popout&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAlCAYAAAAjt+tHAAAACXBIWXMAABcSAAAXEgFnn9JSAAAAB3RJTUUH4wENDyoWA+0MpQAAAAZiS0dEAP8A/wD/oL2nkwAACE5JREFUWMO9WAtU1FUaH1BTQVJJKx+4BxDEgWEGFIzIVUMzPVBauYng8Jr3AxxAHObBvP6MinIUJdLwrTwqzXzkWVMSLW3N7bTrtmvpno7l6WEb7snMB6DffvfOzJ87A5a27t5zvjP/x/1/v9/9Xve7IxA84BFXYBMIi+zBIoUrOCLbxD9PVLgE/9MRtdhKfycW2gfGFzkMCFgXV2CPEStdAyQqLui/BhiXU3lP8xJkzkclSu77SapqSEYRyZ2bE+TO0b8JdGKRozeRRZWDcHXDEuWuEQkyx8gkJTcirtA2VCh3DvJYwJGT7AUngu9PDJ9nGH5/yM9oBU+X1fK3sXlVQyQKVyyu5lkELcUVviZRcHvECtc+BNiNz+vFSq5cWGifm6Sq/oghcE2s4GggRC+23Bv2hHwbfz1eankIFachkBsB/8mu7F4EyZyNzrNGUMsU2H4dfMxCI2v+cAQuRyWX+lSu5HrkbgSU3GcxeVWpgujZQd74uDs4+pS/jpZaxiD45kCFaHpIlDspaKp2JaQV10CavgYma5aDGJ/jN/RdAImvULc2Jt8WRnEIiQWGAPSZCr8oxiBrYRWRa6J8qqEW5tkbIXdlExSteQPkdbtR3oSC2lbIXr4DMq0bIb1kNU+SIXIdSdTE5FlHEoz4woDgFslc3mLhHIRA9X6rRuAUzQqY79gM2oa3wbTjCNib2/3E0eL5Xbb1MKjr98JLrq0wRbeCkmbioUskc64dm22iGRHPZ9gslSf4pLZ+yGwBTr7DghMzS1c1g2n7UbAhSFXTMbDueq+XmHYcpe9szcfAjNfEOjPK1lJr8AtSVneK5a5KksrelBUIAIASiFhUORx9fIE1+xPo37zVLRTgbsBEzDveg8bDH+Nvm3euZ77+1f0wa9l6PxJoiX9jZmX6V68iZ3/0kZI1/WS1GxZw234VvBIts+/05/CvH38G7vXjYGHeke+0DftgWukaak2fblI/hIW2CJ5AssqNvuc+7TE9BxkV66hPfwncsrMN1h04Dddu3gIyzpz/hhKyBpAoqH0dJuGCkhjrYkF7zlNac02C2AJbPGMiTLEVkLNyF9gxuHgwFDv6lyVEwM5c+BLu3LlDCXR2dcOu9rM0HlgCS7f8EeZaNvgFJV6vmVhkHyaIlzmCRDKHnvU9MVlp4ztg84L5zNr21y+g4dAZMOPKHc3vQ1atC56tk0P37dvgGx1Xr4OztR2t02MFkiEkkNnURIufwuyLInkfjOmxiSXwjLEeU+s4r8C47Qi0nvgb3Ojsgj99dgncb7wPFdvfgdHlT8MAlRDaPz/NE+jsvg0HPzoPRsYVJHs0mJ5PLanlSWAgdmDPIBZg5PdDafcRIL4ixcbZesIT4bjalbs/gPNf/0ABiLGb2/8B05eXwrDiFBisEYG+xcUT6OruggOfnAR9416o2uWxILHkktcO0rjyBWOSkkoaBmB1v2RmByNllRQSnwXI6vd+eI6u3je++O4KJNiyYIhOAqEoydw8/t2Nzptg318PT7qKqZt8cVC26RDMNr4SmA3TBNg49EM5xRJ40ckQ2P4unDx3EQKHvsUJ4UtSIEyfBAM1CXDpyrf0+c+3roN0SwWEl6SDdlMr2JuOUwKljYeoa1kCmG2/JyUxOKHI0cLWAFLTiQts+LFswxbYcOwt+P7qDxhs3TyBC5cvwnjzLBiCBEJ1YnAdbKDPf7zxEyS75kOoVgypDhkSOEFjoHjDfphRXkdT3BdrSGYK1n8uGCPSwgZhxtJ1NIrNO4/AVK4YQvUiyKjNg8N//4BPOTLmvaKBocWTqBUilk2Dn25eg8tXOyipEF0ijCqbDvkNG4FrPQnKdXvozskHocL1DTYyIkGU1Bo0ocCWxhJ4smQVqNe/DbKNm2FMeQYM1opAII+FREcWtJ37kCeg2lkFw0omUwIkFox7VsPWk3sgWBFHn4Xpk2GKU0FjgdQVP/8ruSPYK47z7APZxhB8cJHPBJUb5pjrYYa7DAZphVTZw6gsSDEBptbkwLZTb8HBs8dAZM/0AnlkiF4C0aaZNDjDvFaINM6F3LpGDMCGwEJkw2YlxLsNc/2xHuj9GhCNE6JKFlHz+wAICZL3jxhSYUTpFB6IJ4D3IdpEhpAYRi5Jh6QyA6RqatgN6Sa6fZZ/B1xgexzN/2kPCTfEq5fBY7rZqIgo7QEjQUeEBe8tnvmjtFkgUlqoPqazasbq+5jnQJHr6VYlai4Id8RMLA6drCsSkMQoXSZVSFb0y6A9riAyWvcciNRm1LOc7a6uYPBl+a1+TuV6z8a0sHIATihmXUFIiFVWiNLmQ7g+nbok0CKsycn7ofpUiNRKQay2+oN7fL9iXI5psKcDr/L1hMqe3kDuHIwTDaQksySSVE60hhGiNIXwuG4OgqQgWAJKPISgEPBHdNNhnHYhCNVL6fxJKlYHXf1ezDh6Stp0oC2gK1Y42XPeQDTTy+irgJacEHHhyqrQtCYkVAFCTSlKGd5XQqLaAhKVw8/fjOkPSZTVkT6Msdl9HPUmMt3qw/PLgnCrFmIPtw3j4lbvvt8dAOTuE9gbdK9G5pjC+zr89BqhmSUCac0Wpk13vIAKLt/vqchb6/+Mi5odmq3lT8dohfs4I05X98fVr2LjAQvWUVR8GEl1BAKSediAnsccr4/Nt6YTFRmla3l1v1tkur8zKnYsKQj0lx4/Vt9C8Kf4CZNzQ4c+b4gam22Mf2iuLkIQ8/wA9nvZqq140FX/9v8E0P+5GDy3EbybEMA60RSHBYu+TDL0/dFM1QP4uyPDd1QLIxtVKuZuE66+QyznXhb8v0bkYrPf/ag/VIwYLzWHsdXzQYz/ABScQI1BUjcgAAAAAElFTkSuQmCC
  :target: https://mybinder.org/v2/gh/OGGM/binder/master?urlpath=git-pull?repo=https://github.com/OGGM/oggm-edu-notebooks%26amp%3Bbranch=master%26amp%3Burlpath=lab/tree/oggm-edu-notebooks/oggm-tuto/welcome.ipynb%3Fautodecode


OGGM tutorials
----------------

Refer to our :ref:`tutorials` for real-world applications.
