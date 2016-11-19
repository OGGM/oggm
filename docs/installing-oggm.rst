.. _installing.oggm:

Installing OGGM
===============

OGGM itself is a pure python package, but it has several dependencies wich
are not trivial to install. The instructions below are
self-explanatory and should work on any platform.

OGGM is fully `tested`_ with the python `version`_ 3.5 on linux and partially
tested with python 3.4 on `windows`_ (for development purposes only). OGGM
might work with python version 2.7, but it isn't tested any more and we
`strongly recommend`_ to use python 3+.

.. note::

   Complete beginners should get familiar with python and its packaging
   ecosystem before trying to install and run OGGM.

For most users we recommend to install python and the package dependencies
withs the conda_ package manager:
`Install with conda (all platforms)`_. Linux users and people
with experience with `pip`_ can follow the specific instructions
`Install with virtualenv (linux/debian)`_.


.. _tested: https://travis-ci.org/OGGM/oggm
.. _windows: https://ci.appveyor.com/project/fmaussion/oggm
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
    - filelock
    - `salem <https://github.com/fmaussion/salem>`_
    - `motionless <https://github.com/ryancox/motionless/>`_

Optional:
    - progressbar2 (displays the download progress)


Install with conda (all platforms)
----------------------------------

This is the recommended way to install OGGM.

Prerequisites
~~~~~~~~~~~~~

You should have a recent version of `git`_ and of the `conda`_ package manager.
You can get `conda`_ by installing `miniconda`_ (the package manager alone -
recommended)  or `anaconda`_ (the full suite - with many packages you wont
need).


**Linux** users should install a couple of linux packages (not all of them are
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

.. warning::

    The windows build of one of OGGM dependencies is not working properly
    with python 3.5. The only solution until `this problem`_ is resolved is
    to install OGGM in a python 3.4 environment. That is: type ``python=3.4``
    instead of ``python=3.5`` in the command above. All the rest should
    hopefully work the same.


Don't forget to activate it before going on::

    source activate oggm_env

(on windows: ``activate oggm_env``)

.. _environment: http://conda.pydata.org/docs/using/envs.html
.. _this problem: https://github.com/conda-forge/geopandas-feedstock/issues/9


Packages
~~~~~~~~

Install the packages from the `conda-forge`_ and oggm channels::

    conda install -c oggm -c conda-forge oggm-deps

The oggm-deps package is a "meta package". It does not contain any code but
will insall all the packages oggm needs automatically.

.. warning::

    The `conda-forge`_ channel ensures that the complex package dependencies are
    handled correctly. Subsequent installations or upgrades from the default
    conda channel might brake the chain (see an example `here`_). We strongly
    recommend to **always** use the the `conda-forge`_ channel for your
    installation.

You might consider setting `conda-forge`_ (and oggm) per default, as suggested on their
documentation page::

    conda config --add channels conda-forge
    conda config --add channels oggm
    conda install <package-name>

No scientific python installation is complete without installing
`ipython`_ and `jupyter`_::

    conda install -c conda-forge ipython jupyter


.. _conda-forge: https://conda-forge.github.io/
.. _here: https://github.com/ioos/conda-recipes/issues/623
.. _ipython: https://ipython.org/
.. _jupyter: https://jupyter.org/

OGGM
~~~~

You can install OGGM as a normal python package (in that case you will be able
to use the model but not change its code)::

    conda install -c oggm -c conda-forge oggm

We recommend to clone the git repository (or a fork if you want
to participate to the development, see also :ref:`contributing`)::

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
    cloned repository.

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

    $ pip install pyproj rasterio Pillow geopandas netcdf4 scikit-image configobj joblib xarray filelock progressbar2

And the external libraries::

    $ pip install git+https://github.com/fmaussion/motionless.git
    $ pip install git+https://github.com/fmaussion/salem.git

OGGM and tests
~~~~~~~~~~~~~~

Refer to `OGGM`_ above.
