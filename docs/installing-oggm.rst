Installing OGGM
===============

OGGM itself is a pure python package, but it has several dependencies wich
are not trivial to install. The instructions below are
self-explanatory and should work on any platform.

OGGM is `tested`_ with the python `versions`_ 2.7, 3.4 and 3.5. We
`strongly recommend`_ to use python 3.5.

.. note::

   Complete beginners should get familiar with python and its packaging
   ecosystem before trying to install and run OGGM.

For most users we recommend to install python and the package dependencies
withs the conda_ package manager:
`Install with conda (all platforms)`_. Linux users and people
with experience with `pip`_ can follow the specific instructions
`Install with virtualenv (linux/debian)`_.


.. _tested: https://travis-ci.org/OGGM/oggm
.. _versions: https://wiki.python.org/moin/Python2orPython3
.. _conda: http://conda.pydata.org/docs/using/index.html
.. _pip: https://docs.python.org/3/installing/
.. _strongly recommend: http://python3statement.github.io/


Dependencies
------------

Standard SciPy track:
    - numpy
    - scipy
    - scikit-image
    - pillow
    - matplotlib
    - pandas
    - xarray
    - joblib

Python 2 support:
    - six

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
    - nose

Other libraries:
    - `salem <https://github.com/fmaussion/salem>`_
    - `cleo <https://github.com/fmaussion/cleo>`_
    - `motionless (py3) <https://github.com/fmaussion/motionless>`_

Optional:
    - progressbar2 (Display download progress)


Install with conda (all platforms)
----------------------------------

Prerequisites
~~~~~~~~~~~~~

You should have a recent version of `git`_ and of the `conda`_ package manager.
You can get `conda`_ by installing `miniconda`_ (the package manager alone -
recommended)  or `anaconda`_ (the full suite - with many packages you wont
need).


**Linux** users should install a couple of packages (not all of them are
required but it's good to have them anyway)::

    $ sudo apt-get install build-essential liblapack-dev gfortran libproj-dev git gdal-bin libgdal-dev netcdf-bin ncview python-netcdf ttf-bitstream-vera

.. _git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: http://docs.continuum.io/anaconda/install


Conda environment
~~~~~~~~~~~~~~~~~

We recommend to create a specific `environment`_ for OGGM. In a terminal
window, type::

    conda create --name oggm_env python=3.5

You can of course use any other name for your environment.
Don't forget to activate it before going on::

    source activate oggm_env

(on windows: `activate oggm_env`)

.. _environment: http://conda.pydata.org/docs/using/envs.html


Packages
~~~~~~~~

Install the packages from the `conda-forge`_ channel::

    conda install -c conda-forge geopandas matplotlib Pillow joblib netCDF4 scikit-image configobj nose pyproj numpy krb5 rasterio xarray

.. warning::

    The `conda-forge`_ channel ensures that the complex package dependencies are
    handled correctly. Subsequent installations or upgrades from the default
    conda channel might brake the chain (see an example `here`_). We strongly
    recommend to **always** use the the `conda-forge`_ channel for your
    installation.

You might consider setting `conda-forge`_ per default, as suggested on their
documentation page::

    conda config --add channels conda-forge
    conda install <package-name>

If you want progress reports when downloading data files you can optionally
install progressbar2::

    pip install progressbar2

No scientific python installation is complete without installing
`ipython`_ and `jupyter`_::

    conda install -c conda-forge ipython jupyter

After success, install the following packages from Fabien's github::

    pip install git+https://github.com/fmaussion/motionless.git
    pip install git+https://github.com/fmaussion/salem.git
    pip install git+https://github.com/fmaussion/cleo.git


.. _conda-forge: https://conda-forge.github.io/
.. _here: https://github.com/ioos/conda-recipes/issues/623
.. _ipython: https://ipython.org/
.. _jupyter: https://jupyter.org/

OGGM
~~~~

We recommend to clone the git repository (or a fork if you want
to participate to the development)::

   git clone https://github.com/OGGM/oggm.git

Then go to the project root directory::

    cd oggm

And install OGGM in development mode::

    pip install -e .


.. note::

    Installing OGGM in development mode means that subsequent changes to this
    code repository will be taken into account the next time you will
    ``import oggm``. This means that you are going to
    be able to update OGGM with a simple `git pull`_ from the head of the
    cloned repository (but also that if you make changes in this repository,
    this might brake things).

.. _git pull: https://git-scm.com/docs/git-pull


Testing
~~~~~~~

You are almost there! The last step is to check if everything works as
expected. From the oggm directory, type::

    nosetests .

The tests can run for several minutes. If everything worked fine, you
should see something like::

    ...............S.S..................S......................SSS..SSS.SSSSS.SSS
    ----------------------------------------------------------------------
    Ran 77 tests in 401.080s

    OK (SKIP=17)

You can safely ignore deprecation warnings and other DLL messages as long as
the tests end with ``OK``.

**Congrats**, you are now set-up for the :ref:`getting-started` section!


Install with virtualenv (linux/debian)
--------------------------------------

.. note::

   The installation with pip requires to compile the packages one by one: it
   can take a long time. Unless you have a good reason to be here,
   `Install with conda (all platforms)`_ is probably what you want do do.

The instructions below are for Debian / Ubuntu / Mint systems only!

Linux packages
~~~~~~~~~~~~~~

For building stuffs::

    $ sudo apt-get install build-essential python-pip liblapack-dev gfortran libproj-dev

For matplolib to work on **python 2**::

    $ sudo apt-get install python-gtk2-dev

And on **python 3**::

    $ sudo apt-get install tk-dev python3-tk python3-dev

For GDAL::

    $ sudo apt-get install gdal-bin libgdal-dev python-gdal

For NETCDF::

    $ sudo apt-get install netcdf-bin ncview python-netcdf


Virtual environment
~~~~~~~~~~~~~~~~~~~

Install::

    $ sudo pip install virtualenvwrapper

Create the directory where the virtual environments will be created::

    $ mkdir ~/.pyvirtualenvs

Add these three lines to the files: ~/.profile and ~/.bashrc::

    # Virtual environment options
    export WORKON_HOME=$HOME/.pyvirtualenvs
    source /usr/local/bin/virtualenvwrapper_lazy.sh

Reset your profile::

    $ . ~/.profile

Make a new environment with **python 2**::

    $ mkvirtualenv oggm_env -p /usr/bin/python

Or **python 3**::

    $ mkvirtualenv oggm_env -p /usr/bin/python3

(Details: http://simononsoftware.com/virtualenv-tutorial-part-2/ )


Python Packages
~~~~~~~~~~~~~~~

Be sure to be on the working environment::

    $ workon oggm_env

Install one by one the easy stuff::

   $ pip install numpy scipy pandas shapely

For Matplotlib and **python 2** we need to link the libs in the virtual env::

    $ ln -sf /usr/lib/python2.7/dist-packages/{glib,gobject,cairo,gtk-2.0,pygtk.py,pygtk.pth} $VIRTUAL_ENV/lib/python2.7/site-packages
    $ pip install matplotlib

(Details: http://www.stevenmaude.co.uk/2013/09/installing-matplotlib-in-virtualenv.html )

For Matplotlib and **python 3** it doesn't seem to be necessary::

    $ pip install matplotlib

Check if plotting works by running these three lines in python::

    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1,2,3])
    >>> plt.show()

If nothing shows-up, something got wrong.

For **GDAL**, it's also not straight forward. First, check which version of
GDAL is installed::

    $ dpkg -s libgdal-dev

The version (10, 11, ...) should match that of the python package. Install
using the system binaries::

    $ pip install gdal==1.10.0 --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"
    $ pip install fiona --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"

(Details: http://tylerickson.blogspot.co.at/2011/09/installing-gdal-in-python-virtual.html )

Install further stuffs::

    $ pip install pyproj rasterio Pillow geopandas netcdf4 scikit-image configobj joblib xarray progressbar2

And the external libraries::

    $ pip install git+https://github.com/fmaussion/motionless.git
    $ pip install git+https://github.com/fmaussion/salem.git
    $ pip install git+https://github.com/fmaussion/cleo.git

OGGM and tests
~~~~~~~~~~~~~~

Refer to `OGGM`_ above.
