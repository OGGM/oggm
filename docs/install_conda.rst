OGGM install checklist with conda
=================================

As of today the default conda version of Fiona is broken
(see: https://github.com/ioos/conda-recipes/issues/623 ). The
workaround described below should work fine but it is a temporary solution
as it relies on older versions of the packages.

Prerequisites
-------------

You should have a recent version of `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
and of `conda <http://conda.pydata.org/docs/using/index.html>`_, either by installing
`miniconda <http://conda.pydata.org/miniconda.html>`_ or `anaconda
<http://docs.continuum.io/anaconda/install>`_.


**Linux** users should install a couple of packages (just to be sure)::

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


Packages
--------

When installing the packages the versions of GDAL and numpy have to be set
to 1.11.2 and 1.9.3, respectively. The other packages will follow::

    conda install -c https://conda.anaconda.org/ioos geopandas pandas matplotlib Pillow joblib netCDF4 rasterio scikit-image configobj nose pyproj numpy=1.9.3 gdal=1.11.2

note that we used the *ioos* channel for the installation.

After success, install the following packages from Fabien Maussion's github::

    pip install git+https://github.com/fmaussion/motionless.git
    pip install git+https://github.com/fmaussion/salem.git
    pip install git+https://github.com/fmaussion/cleo.git


OGGM
----


We recommend to clone the git repository (or your fork if you want
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

