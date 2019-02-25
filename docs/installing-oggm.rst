.. _installing.oggm:

Installing OGGM
===============

OGGM itself is a pure Python package, but it has several dependencies which
are not trivial to install. The instructions below provide all the required
detail and should work on any platform.

OGGM is fully `tested`_ with Python version 3.6 on Linux, and all versions
above 3.4 should work as well. OGGM doesn't work with Python 2.7.

OGGM itself should also work on Mac OS and Windows platforms, but we make no
guarantee that our dependencies do.

.. note::

   Complete beginners should get familiar with Python and its packaging
   ecosystem before trying to install and run OGGM.

For most users we recommend following :ref:`the steps here<conda-install>` to
install Python and the package dependencies with the conda_ package manager.
Linux or Debian users and people with experience with `pip`_ can
:ref:`follow the specific instructions here<virtualenv-install>` to install
with virtualenv.

.. _tested: https://travis-ci.org/OGGM/oggm
.. _conda: http://conda.pydata.org/docs/using/index.html
.. _pip: https://docs.python.org/3/installing/
.. _strongly recommend: http://python3statement.github.io/


Dependencies
------------

Standard SciPy stack:
    - numpy
    - scipy
    - scikit-image
    - pillow
    - matplotlib
    - pandas
    - xarray
    - joblib

Configuration file parsing tool:
    - configobj

I/O:
    - netcdf4

GIS tools:
    - gdal
    - shapely
    - pyproj
    - rasterio
    - geopandas

Testing:
    - pytest

Other libraries:
    - boto3
    - `salem <https://github.com/fmaussion/salem>`_
    - `motionless <https://github.com/ryancox/motionless/>`_

Optional:
    - progressbar2 (displays the download progress)
    - bottleneck (speeds up xarray operations)
    - dask (works nicely with xarray)
    - `python-colorspace <https://github.com/retostauffer/python-colorspace>`_
      (applies HCL-based color palettes to some graphics)

.. _conda-install:

Install with conda (all platforms)
----------------------------------

This is the recommended way to install OGGM.

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


where ``3.X`` is the Python version shipped with conda (currently 3.6).
You can of course use any other name for your environment.

Don't forget to activate it before going on::

    source activate oggm_env

(on Windows: ``activate oggm_env``)

.. _environment: http://conda.pydata.org/docs/using/envs.html
.. _this problem: https://github.com/conda-forge/geopandas-feedstock/issues/9


Dependencies
~~~~~~~~~~~~

Install all OGGM dependencies from the ``conda-forge`` and ``oggm`` conda channels::

    conda install -c oggm -c conda-forge oggm-deps

The ``oggm-deps`` package is a "meta package". It does not contain any code but
will install all the packages OGGM needs automatically.

.. warning::

    The `conda-forge`_ channel ensures that the complex package dependencies are
    handled correctly. Subsequent installations or upgrades from the default
    conda channel might brake the chain. We strongly
    recommend to **always** use the the `conda-forge`_ channel for your
    installation.

You might consider setting `conda-forge`_ (and ``oggm``) as your 
default channels::

    conda config --add channels conda-forge
    conda config --add channels oggm

No scientific Python installation is complete without installing
`IPython`_ and `Jupyter`_::

    conda install -c conda-forge ipython jupyter


.. _conda-forge: https://conda-forge.github.io/
.. _IPython: https://ipython.org/
.. _Jupyter: https://jupyter.org/


Install OGGM itself
~~~~~~~~~~~~~~~~~~~

First, choose which version of OGGM you would like to install:

- **stable**: this is the latest version officially released and has a fixed
  version number (e.g. v1.1).
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
system. In your conda environmnent, simply do::

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

The tests can run for a couple of minutes. If everything worked fine, you
should see something like::

    =============================== test session starts ===============================
    platform linux -- Python 3.5.2, pytest-3.3.1, py-1.5.2, pluggy-0.6.0
    Matplotlib: 2.1.1
    Freetype: 2.6.1
    rootdir:
    plugins: mpl-0.9
    collected 164 items

    oggm/tests/test_benchmarks.py ...                                           [  1%]
    oggm/tests/test_graphics.py ...................                             [ 13%]
    oggm/tests/test_models.py ................sss.ss.....sssssss                [ 34%]
    oggm/tests/test_numerics.py .ssssssssssssssss                               [ 44%]
    oggm/tests/test_prepro.py .......s........................s..s.......       [ 70%]
    oggm/tests/test_utils.py .....................sss.s.sss.sssss..ss.          [ 95%]
    oggm/tests/test_workflow.py sssssss                                         [100%]

    ==================== 112 passed, 52 skipped in 187.35 seconds =====================


You can safely ignore deprecation warnings and other messages (if any),
as long as the tests end without errors.

This runs a minimal suite of tests. If you want to run the entire test suite
(including graphics and slow running tests), type::

    pytest --pyargs oggm --run-slow --mpl

**Congrats**, you are now set-up for the :ref:`getting-started` section!


.. _virtualenv-install:

Install with virtualenv (Linux/Debian)
--------------------------------------

.. note::

   The installation with virtualenv and pip requires a few more steps than with
   conda. Unless you have a good reason to install by this route,
   :ref:`installing with conda <conda-install>` is probably what you want to do.


The instructions below have been tested on Debian / Ubuntu / Mint systems only!

Linux packages
~~~~~~~~~~~~~~

Run the following commands to install required packages.

For the build::

    $ sudo apt-get install build-essential python-pip liblapack-dev gfortran libproj-dev python-setuptools

For matplolib::

    $ sudo apt-get install tk-dev python3-tk python3-dev

For GDAL::

    $ sudo apt-get install gdal-bin libgdal-dev python-gdal

For NetCDF::

    $ sudo apt-get install netcdf-bin ncview python-netcdf4


Virtual environment
~~~~~~~~~~~~~~~~~~~

Next follow these steps to set up a virtual environment.

Install extensions to virtualenv::

    $ sudo apt-get install virtualenvwrapper

Reload your profile::

    $ source /etc/profile

Make a new environment, for example called ``oggm_env``, with **Python 3**::

    $ mkvirtualenv oggm_env -p /usr/bin/python3

(further details can be found for example in
`this tutorial <http://simononsoftware.com/virtualenv-tutorial-part-2/>`_)


Python packages
~~~~~~~~~~~~~~~

Be sure to be on the working environment::

    $ workon oggm_env

Update pip (important!)::

    $ pip install --upgrade pip

Install some packages one by one::

   $ pip install numpy scipy pandas shapely matplotlib

Installing **GDAL** is not so straightforward. First, check which version of
GDAL is installed on your Linux system::

    $ gdal-config --version

The package version (e.g. ``2.2.0``, ``2.3.1``, ...) should match
that of the Python package you want to install. For example, if the Linux
GDAL version is ``2.2.0``, install the latest corresponding Python version.
The following command works on any system and automatically gets the right version::

    $ pip install gdal=="$(gdal-config --version)" --install-option="build_ext" --install-option="$(gdal-config --cflags | sed 's/-I/--include-dirs=/')"

Fiona also builds upon GDAL, so let's compile it the same way::

    $ pip install fiona --install-option="build_ext" --install-option="$(gdal-config --cflags | sed 's/-I/--include-dirs=/')"

(Details can be found in `this blog post <http://tylerickson.blogspot.co.at/2011/09/installing-gdal-in-python-virtual.html>`_.)

Now install further dependencies::

    $ pip install pyproj rasterio Pillow geopandas netcdf4 scikit-image configobj joblib xarray boto3 progressbar2 pytest motionless dask bottleneck

Finally, install the salem and python-colorspace libraries::

    $ pip install git+https://github.com/fmaussion/salem.git
    $ pip install git+https://github.com/retostauffer/python-colorspace.git

OGGM and tests
~~~~~~~~~~~~~~

Refer to `Install OGGM itself`_ above.
