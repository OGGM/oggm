OGGM install checklist with conda
=================================


Prerequisites
-------------

You should have a recent version of `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
and of `conda <http://conda.pydata.org/docs/using/index.html>`_ (either by
installing
`miniconda <http://conda.pydata.org/miniconda.html>`_ or `anaconda
<http://docs.continuum.io/anaconda/install>`_).


**Linux** users should install a couple of packages (not all of them are
required but just to be sure)::

    $ sudo apt-get install build-essential liblapack-dev gfortran libproj-dev gdal-bin libgdal-dev netcdf-bin ncview python-netcdf


Conda environment
-----------------

We recommend to create a specific `environment <http://conda.pydata
.org/docs/using/envs.html>`_ for OGGM. In a terminal window, type::

    conda create --name oggm python=3.4

The OGGM conda installation has been tested with Python 2.7 and 3.4. You can
of course use any other name for your environment. Don't forget to activate
it::

    source activate oggm

(on windows: `activate oggm`)


Packages
--------

We install the packages from the ioos channel:

    conda install -c ioos geopandas matplotlib Pillow joblib netCDF4 scikit-image configobj nose pyproj numpy

After success, install the following packages from Fabien Maussion's github::

    pip install git+https://github.com/fmaussion/motionless.git
    pip install git+https://github.com/fmaussion/salem.git
    pip install git+https://github.com/fmaussion/cleo.git


OGGM
----

We recommend to clone the git repository (or a fork if you want
to participate to the development)::

   git clone git@github.com:OGGM/oggm.git

Then go to the root directory::

    cd oggm

And install OGGM in development mode::

    pip install -e .


Testing
-------

From the oggm root directory, type::

    nosetests .

