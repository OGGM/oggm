Installing OGGM
===============

.. important::

   Did you know that you can try OGGM in your browser before installing it
   on your computer? Visit :doc:`cloud` for more information.

   Did you know that we provide standard projections for all glaciers in the
   world, and that you may not have to run OGGM for your use case?
   Visit :doc:`download-projections` for more information.

OGGM itself is a pure python package, but some dependencies are not trivial to install.
The instructions below should provide all the required details.
See :ref:`install-troubleshooting` if something goes wrong.

OGGM is `fully tested`_ with Python versions 3.11 to 3.14 on Linux.
MacOS is not automatically tested, but should still work.

.. warning::

    **OGGM does not work natively on Windows, but there is a workaround.**
    If you are using Windows 10 or above, install the free `Windows subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ (WSL), then install and run OGGM from there.
    Within WSL, the installation instructions are identical to the instructions for Linux.

OGGM now supports installation with ``pip``, ``conda``, and ``uv``.
For most users we recommend installing Python and package dependencies with the :ref:`conda package manager <conda-install>`, in particular with ``mamba`` and ``conda-forge``.

.. _fully tested: https://github.com/OGGM/oggm/actions/workflows/run-tests.yml
.. _conda: https://conda.io/projects/conda/en/latest/user-guide/index.html
.. _pip: https://docs.python.org/3/installing/
.. _mamba: https://mamba.readthedocs.io

OGGM can be installed:

- as a library, if you don't want to modify its source code. This is recommended for most users.
    - **stable**: this is the latest official release and has a fixed version number (e.g. v1.6.3).
    - **dev**: this is the development version.
      It may contain new features and bug fixes, but will continue changing until its release.

- as an **editable**: if you want to make changes, or develop the model. This is recommended for developers.

Don't forget to :ref:`test-oggm` before using it!

Dependencies
------------

A full installation of OGGM requires GDAL.
The easiest way to get GDAL is:

.. code-block:: bash

    sudo apt-get install gdal-bin libgdal-dev  # Linux (Debian distros)
    brew install gdal  # MacOS

OGGM supports installation with ``conda``, ``pip``, and ``uv``.
We recommend using ``conda`` or ``mamba`` for most users, and ``uv`` for OGGM contributors.

.. note::

    If you are not familiar with Python and its
    `way too many package management systems <https://xkcd.com/1987/>`_,
    you might find all of this quite confusing and overwhelming. Be patient,
    `read the docs <https://docs.conda.io>`_ and stay hydrated.

You should have a recent version of the `conda`_ package manager.
Our recommendation is `mambaforge`_. If you are completely new to these things, check out
`this page <https://fabienmaussion.info/intro_to_programming/week_01/01-Installation.html>`_
which explains how to install `mambaforge`_ and
`this one <https://fabienmaussion.info/intro_to_programming/week_05/01-install-packages.html>`_
for installing packages.

.. warning::

    Do not install mambaforge on top of an existing conda installation! See
    `this issue <https://github.com/OGGM/oggm/issues/1571>`_ for context.
    If you have conda installed and want to switch to mamba + conda-forge,
    follow the instructions on the respective platforms.

OGGM now also supports installation with `uv`_, an ultra-fast package manager which allows you to quickly switch between different installations of OGGM, and set up reproducible environments for running simulations.
We would strongly recommend this for OGGM contributors, and users who require a stable Python environment.

.. _miniconda: http://conda.pydata.org/miniconda.html
.. _mambaforge: https://github.com/conda-forge/miniforge#mambaforge
.. _uv: https://docs.astral.sh/uv/getting-started/features/

.. _conda-install:

The recommended way: ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For users who don't want to use ``uv`` at all.

Create and activate a conda environment:

.. code-block:: bash

    conda env create --n oggm python=3.13 oggm -c conda-forge -c oggm
    conda activate oggm

.. _uv-install:

The fastest way: ``uv``
~~~~~~~~~~~~~~~~~~~~~~~

For users who just want to run a notebook and don't need a dedicated environment.
This includes users who want to test OGGM before installing it, and teaching staff who want to use OGGM in a classroom.

If you are new to Python and want to get started with OGGM as quickly as possible, we recommend using `uv`_.
You can install and run OGGM on a Jupyter server without polluting your base system or conda environment.

Install `uv`:

.. code-block:: bash

    wget -qO- https://astral.sh/uv/install.sh | sh

Launch a Jupyter notebook with OGGM fully installed:

.. code-block:: bash

    uvx --with oggm jupyter lab

This installs a minimal version of OGGM into an isolated environment and runs a Jupyter server.

.. note:: This will not pollute your base system or conda environment.

.. _pip-install:

The extensible way: ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate your python environment (conda or virtualenv).

Install the latest **stable** release:

.. code-block:: bash

    pip install oggm  # for a minimal installation
    pip install oggm[full]  # for a full installation
    pip install oggm[dev]  # for developers, including documentation dependencies
    uv pip install oggm  # if you have uv installed

Or install the latest **development** version:

.. code-block:: bash

    pip install --upgrade git+https://github.com/OGGM/oggm.git

The editable way: ``pip`` or ``uv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is recommended for contributors or if you want to make changes to the OGGM source code.
It installs OGGM in "editable" mode, so any changes you make to the source code are immediately applied when you import OGGM.

You will need `git`_ installed on your system.
Clone the latest repository version:

.. code-block:: bash

    git clone https://github.com/OGGM/oggm.git
    cd oggm
    # Activate your python environment (conda or venv) here if you haven't already
    pip install -e .[full]  # if using pip
    uv sync --extra full  # if using uv

.. _git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

.. note::

    You can also update OGGM with a simple `git pull`_ from the root of the cloned repository.

.. _git pull: https://git-scm.com/docs/git-pull

.. _test-oggm:

Test OGGM
~~~~~~~~~

Activate your python environment, and test your OGGM installation:

.. code-block:: bash

    pytest.oggm  --disable-warnings

The tests should run for 5 to 10 minutes. If successful, you should see something like::

    =================================== test session starts ====================================
    platform linux -- Python 3.10.6, pytest-7.1.3, pluggy-1.0.0
    Matplotlib: 3.5.3
    Freetype: 2.12.1
    rootdir: /home/mowglie/disk/Dropbox/HomeDocs/git/oggm-fork, configfile: pytest.ini
    plugins: anyio-3.6.1, mpl-0.150.0
    collected 373 items

    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_benchmarks.py ...s.ss            [  1%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_graphics.py ..........s...s....s [  7%]
    ss                                                                                   [  7%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_minimal.py ...                   [  8%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_models.py ...................... [ 14%]
    .........ssss.......ssss.ss.ss..ssssssssss..ssssssssssssssssssssssssssssssss.s       [ 35%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_numerics.py sssss.ss.sssss.s.sss [ 40%]
    ss.sss.sss.s                                                                         [ 43%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_prepro.py ..................s... [ 49%]
    ...........s....ss...ss..s.....ss.....sssssss..sssss....s.......                     [ 67%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_shop.py ss..........s.           [ 70%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_utils.py ....................... [ 76%]
    ....sss.......sss...s....................ss.s..ssss.ssssss..sss...s.ss.ss.ss....     [ 98%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_workflow.py ssssss               [100%]

    ======================= 224 passed, 149 skipped in 217.03s (0:03:37) ======================

.. important::

   The tests (without the ``--run-slow`` option) should run in 5 to 15 minutes.
   If it takes longer, this may indicate something's wrong.

If you want to run the *entire* test suite (including graphics and slow running tests):

.. code-block:: bash

    pytest.oggm --run-slow --mpl

**Congratulations**, you are now set-up for the :doc:`getting-started` section!

.. _install-troubleshooting:

Installation troubleshooting
----------------------------

Please get in touch with us `on github <https://github.com/OGGM/oggm/issues>`_ if you encounter issues installing OGGM.
Installations can be tricky, and typical errors are often related to the `pyproj`, `fiona`` and `GDAL`` packages, which are heavy, change often, and are prone to platform specific errors.
It may help to diagnose which package is causing the error.
Errors like `segmentation fault` or `Proj Error` usually point to errors in upstream packages, rarely in OGGM itself.

.. _virtualenv-install:

Install with pyenv (Linux)
--------------------------

.. note::

   We recommend our users to use ``conda`` instead of ``pip``, because
   of the ease of installation with ``conda``. If you are familiar with ``pip`` and
   ``pyenv``, the instructions below work as well: as of Sept 2022 (and thanks
   to pip wheels), a pyenv installation is possible without major issue
   on Debian/Ubuntu/Mint systems.

Linux packages
~~~~~~~~~~~~~~

Run the following commands to install the required linux packages.

For building python and stuff::

    $ sudo apt-get install --no-install-recommends make build-essential git \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget \
        curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev

For NetCDF and HDF::

    $ sudo apt-get install netcdf-bin ncview hdf5-tools libhdf5-dev


Pyenv and pyenv-virtualenv
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If you are not familiar with pyenv, you can visit
    `their documentation <https://realpython.com/intro-to-pyenv/>`_
    (especially the installing pyenv section).

Install `pyenv <https://github.com/pyenv/pyenv>`_ and create a new virtual environment
with a recent python version (3.7+) using `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_.

Python packages
~~~~~~~~~~~~~~~

Be sure to be on the working environment::

    $ pyenv activate oggm_env

Update pip (**important!**)::

    $ pip install --upgrade pip

Install some packages one by one::

   $ pip install numpy scipy pandas shapely matplotlib pyproj \
       rasterio Pillow geopandas netcdf4 scikit-image configobj joblib \
       xarray progressbar2 pytest motionless dask bottleneck toolz \
       tables rioxarray pytables

A pinning of the NetCDF4 package to 1.3.1 might be necessary on some systems
(`related issue <https://github.com/Unidata/netcdf4-python/issues/962>`_).

Finally, install the pytest-mpl OGGM fork and salem libraries::

    $ pip install git+https://github.com/OGGM/pytest-mpl.git
    $ pip install salem

Install OGGM and run the tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to `Install OGGM itself`_ above.