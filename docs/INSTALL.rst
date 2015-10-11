Install OGGM on your system
===========================

OGGM relies on several python packages. All of them are easy to install and
do not require external libraries, with the exception of GDAL. 

Due to GDAL problems with conda
(see this `issue <https://github.com/OGGM/oggm/issues/1>`_ ), we currently
recommend to follow the installation with vitualenv
described `here <./install_virtualenv.rst>`_.

OGGM Dependencies
-----------------

Standard SciPy track:
    - numpy
    - scipy
    - scikit-image
    - pillow
    - matplotlib
    - pandas
    - joblib


Python 2 and 3 support:
    - six

Configuration file parsing tool:
    - configobj

I/O:
    - netcdf4

GIS and geometrical tools:
    - geopandas
    - shapely
    - pyproj
    - gdal

Other libraries:
    - `salem <https://github.com/fmaussion/salem>`_
    - `cleo <https://github.com/fmaussion/cleo>`_
    - `motionless (py3) <https://github.com/fmaussion/motionless>`_