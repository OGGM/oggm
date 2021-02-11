.. _installing.oggm:

Installing OGGM
===============

.. important::

   Did you know that you can try OGGM in your browser before installing it
   on your computer? Visit :ref:`cloud` for more information.

OGGM itself is a pure Python package, but it has several dependencies which
are not trivial to install. The instructions below provide all the required
details and should work on Linux and Mac OS. See :ref:`install-troubleshooting`
if something goes wrong.

OGGM is fully `tested`_ with Python version 3.6, 3.7 and 3.8 on Linux.
OGGM doesn't work with Python 2. We do not test OGGM automatically on
Mac OSX, but it should probably run fine there as well.

.. warning::

    OGGM does not work on Windows. If you are using Windows 10,
    we recommend to install the free
    `Windows subsytem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_
    and install and run OGGM from there.

For most users we recommend to
install Python and the package dependencies with the :ref:`conda package manager <conda-install>`.
Linux users with experience with `pip`_ can follow
:ref:`these instructions <virtualenv-install>` to install OGGM in a pyenv environment with pip.

.. _tested: https://travis-ci.org/OGGM/oggm
.. _conda: https://conda.io/projects/conda/en/latest/user-guide/index.html
.. _pip: https://docs.python.org/3/installing/
.. _strongly recommend: http://python3statement.github.io/


Dependencies
------------

Here is a list of *all* dependencies of the OGGM model. If you want to use
OGGM's numerical models only (i.e. no GIS or preprocessing tools), refer to
`Install a minimal OGGM environment`_ below.

Standard SciPy stack:
    - numpy
    - scipy
    - scikit-image
    - pillow
    - matplotlib
    - pandas
    - xarray
    - dask
    - joblib

Configuration file parsing tool:
    - configobj

I/O:
    - netcdf4
    - pytables

GIS tools:
    - shapely
    - pyproj
    - rasterio
    - geopandas

Testing:
    - pytest
    - pytest-mpl (`OGGM fork <https://github.com/OGGM/pytest-mpl>`_ required)

Other libraries:
    - `salem <https://github.com/fmaussion/salem>`_
    - `motionless <https://github.com/ryancox/motionless/>`_

Optional:
    - progressbar2 (displays the download progress)
    - bottleneck (might speed up some xarray operations)
    - `python-colorspace <https://github.com/retostauffer/python-colorspace>`_
      (applies HCL-based color palettes to some graphics)

.. _conda-install:

Install with conda (all platforms)
----------------------------------

This is the recommended way to install OGGM for most users.

.. note::

    If you are not familiar with Python and its
    `way too many package management systems <https://xkcd.com/1987/>`_, you might find all
    of this quite confusing and overwhelming. Be patient,
    `read the docs <https://docs.conda.io>`_ and stay hydrated.

Prerequisites
~~~~~~~~~~~~~

You should have a recent version of the `conda`_ package manager.
You can get `conda`_ by installing `miniconda`_ (the package manager alone -
recommended)  or `anaconda`_ (the full suite - with many packages you won't
need).


.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: http://docs.continuum.io/anaconda/install

Conda environment
~~~~~~~~~~~~~~~~~

We recommend to create a specific `environment`_ for OGGM. In a terminal
window, type::

    conda create --name oggm_env python=3.X


where ``3.X`` is the Python version shipped with conda (currently 3.8).
You can of course use any other name for your environment.

Don't forget to activate it before going on::

    source activate oggm_env


.. _environment: https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html


Feeling adventurous? Try mamba (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conda package manager has recently been criticized for being slow (it *is*
quite slow to be honest). A new, faster tool is now available to replace conda: `mamba <https://mamba.readthedocs.io>`_.
Mamba is a drop-in replacement for all conda commands.
If you feel like it, install mamba in your conda environment (``conda install -c conda-forge mamba``)
and replace all occurrences of ``conda`` with ``mamba`` in the instructions below.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

Install all OGGM dependencies from the ``conda-forge`` and ``oggm`` conda channels::

    conda install -c oggm -c conda-forge oggm-deps

The ``oggm-deps`` package is a "meta package". It does not contain any code but
will install all the packages OGGM needs automatically.

.. important::

    The `conda-forge`_ channel ensures that the complex package dependencies are
    handled correctly. Subsequent installations or upgrades from the default
    conda channel might brake the chain. We strongly
    recommend to **always** use the the `conda-forge`_ channel for your
    installation.

You might consider setting `conda-forge`_  as your
default channel::

    conda config --add channels conda-forge

No scientific Python installation is complete without installing a good
testing framework, as well as `IPython`_ and `Jupyter`_::

    conda install -c conda-forge pytest ipython jupyter

.. _conda-forge: https://conda-forge.github.io/
.. _IPython: https://ipython.org/
.. _Jupyter: https://jupyter.org/


Install OGGM itself
~~~~~~~~~~~~~~~~~~~

First, choose which version of OGGM you would like to install:

- **stable**: this is the latest version officially released and has a fixed
  version number (e.g. v1.1). As of today (Feb 2021), we do *not* recommend to
  install the stable version which is quite outdated. We are in the process
  of releasing a new stable version soon(ish).
- **dev**: this is the development version. It might contain new
  features and bug fixes, but is also likely to continue to change until a
  new release is made. This is the recommended way if you want to use the
  latest changes to the code.
- **dev+code**: this is the recommended way if you plan to explore the OGGM
  codebase, contribute to the model, and/or if you want to use the most
  recent model updates.

**‣ install the stable version:**

If you are using conda, you can install stable OGGM as a normal conda package::

    conda install -c oggm oggm

If you are using pip, you can install OGGM from `PyPI <https://pypi.python.org/pypi/oggm>`_::

    pip install oggm

**‣ install the dev version:**

For this to work you'll need to have the `git`_ software installed on your
system. In your conda environment, simply do::

    pip install --upgrade git+https://github.com/OGGM/oggm.git

With this command you can also update an already installed OGGM version
to the latest version.


**‣ install the dev version + get access to the OGGM code:**

For this to work you'll need to have the `git`_ software installed on your
system. Then, clone the latest repository version::

    git clone https://github.com/OGGM/oggm.git

.. _git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Then go to the project root directory::

    cd oggm

And install OGGM in development mode (this is valid for both  **pip** and
**conda** environments)::

    pip install -e .


.. note::

    Installing OGGM in development mode means that subsequent changes to this
    code repository will be taken into account the next time you will
    ``import oggm``. You can also update OGGM with a simple `git pull`_ from
    the root of the cloned repository.

.. _git pull: https://git-scm.com/docs/git-pull


Testing OGGM
~~~~~~~~~~~~

You can test your OGGM installation by running the following command from
anywhere (don't forget to activate your environment first)::

    pytest --pyargs oggm

The tests can run for about 10 minutes (`we are trying to reduce this <https://github.com/OGGM/oggm/issues/1063>`_).
If everything worked fine, you should see something like::

    ================================ test session starts ================================
    platform linux -- Python 3.8.5, pytest-6.0.2, py-1.9.0, pluggy-0.13.1
    Matplotlib: 3.3.2
    Freetype: 2.6.1
    rootdir: /home/mowglie/disk/Dropbox/HomeDocs/git/oggm-fork, configfile: pytest.ini
    plugins: mpl-0.122
    collected 297 items

    oggm/tests/test_benchmarks.py .......                                         [  2%]
    oggm/tests/test_graphics.py ...................X                              [  9%]
    oggm/tests/test_minimal.py ...                                                [ 10%]
    oggm/tests/test_models.py ..........................sss.......ssss..s.ss..sss [ 27%]
    sss..sss                                                                      [ 29%]
    oggm/tests/test_numerics.py .sssssssssss.ssss...s..ss.s                       [ 39%]
    oggm/tests/test_prepro.py .................s........................s........ [ 56%]
    ........s....s............                                                    [ 64%]
    oggm/tests/test_shop.py .......                                               [ 67%]
    oggm/tests/test_utils.py .................................................... [ 84%]
    ss.ss..sssss.ssssss..sss...s.ss.ss.ss..                                       [ 97%]
    oggm/tests/test_workflow.py ssssss                                            [100%]

    ================================= warnings summary ==================================
    (warnings are mostly ok)
    ======== 223 passed, 73 skipped, 1 xpassed, 9 warnings in 771.11s (0:12:51) =========


You can safely ignore deprecation warnings and other messages (if any),
as long as the tests end without errors.

This runs a minimal suite of tests. If you want to run the entire test suite
(including graphics and slow running tests), type::

    pytest --pyargs oggm --run-slow --mpl

**Congrats**, you are now set-up for the :ref:`getting-started` section!



.. _install-troubleshooting:

Installation troubleshooting
----------------------------

We try to do our best to avoid issues, but experience shows that the installation
of the necessary packages can be difficult. Typical errors are often
related to the pyproj, fiona and GDAL packages, which are heavy and (for pyproj)
have changed a lot in the recent past and are prone to platform specific errors.

If the tests don't pass, a diagnostic of which package creates the errors
might be necessary. Errors like ``segmentation fault`` or ``Proj Error``
are frequent and point to errors in upstream packages, rarely in OGGM itself.

If you are having troubles, installing the packages manually from a fresh
environment might help. At the time of writing (20.01.2021), creating an
environment from the following ``environment.yml`` file used to work::

    name: oggm_env
    channels:
      - conda-forge
    dependencies:
      - python=3.8
      - jupyter
      - jupyterlab
      - numpy
      - scipy
      - pandas
      - shapely
      - matplotlib
      - Pillow
      - netcdf4
      - scikit-image
      - scikit-learn
      - configobj
      - xarray
      - pytest
      - dask
      - bottleneck
      - pyproj
      - cartopy
      - geopandas
      - rasterio
      - descartes
      - seaborn
      - pytables
      - pip
      - pip:
        - joblib
        - progressbar2
        - motionless
        - git+https://github.com/fmaussion/salem.git
        - git+https://github.com/retostauffer/python-colorspace
        - git+https://github.com/OGGM/pytest-mpl
        - git+https://github.com/OGGM/oggm


See the
`conda docs <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_
for more information about how to create an environment from a ``yml`` file, OR
you can do what I usually do (much faster): install `mamba`_
first, then run ``mamba env create -f environment.yml``.


.. _virtualenv-install:

Install with pyenv (Linux)
--------------------------

.. note::

   We recommend our users to use `conda` instead of `pip`, because
   of the ease of installation with `conda`. If you are familiar with `pip` and
   `pyenv`, the instructions below work as well: as of Sept 2020 (and thanks
   to pip wheels), a pyenv
   installation is possible without major issue on Debian/Ubuntu/Mint
   systems.

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
       xarray progressbar2 pytest motionless dask bottleneck toolz descartes \
       tables

A pinning of the NetCDF4 package to 1.3.1 might be necessary on some systems
(`related issue <https://github.com/Unidata/netcdf4-python/issues/962>`_).

Finally, install the pytest-mpl OGGM fork, salem and python-colorspace libraries::

    $ pip install git+https://github.com/OGGM/pytest-mpl.git
    $ pip install git+https://github.com/fmaussion/salem.git
    $ pip install git+https://github.com/retostauffer/python-colorspace.git

Install OGGM and run tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to `Install OGGM itself`_ above.

Install a minimal OGGM environment
----------------------------------

If you plan to use only the numerical core of OGGM (that is, for idealized
simulations or teaching), you can skip many dependencies and only
install this shorter list:

- numpy
- scipy
- pandas
- matplotlib
- shapely
- requests
- configobj
- netcdf4
- xarray
- pytables

Installing them with pip or conda should be much easier.
`Install OGGM itself`_ then as above.

Running the tests in this minimal environment works the same. Simply run
from a terminal::

    pytest --pyargs oggm

The number of tests will be much smaller!
