Install OGGM on your system
===========================

OGGM relies on several scientific python packages. Most of them are easy to
install and do not require external libraries. GDAL and its python bindings
are usually less easy to install.

For most users we recommend to install the packages using
`conda <http://conda.pydata.org/docs/using/index.html>`_: 

`OGGM Install checklist with conda <./install_conda.rst>`_

Linux users and/or users with experience with *pip* can
follow the `install instructions for linux and virtualenv <./install_virtualenv.rst>`_.


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
    - gdal
    - rasterio
    - shapely
    - pyproj
    - geopandas

Other libraries:
    - `salem <https://github.com/fmaussion/salem>`_
    - `cleo <https://github.com/fmaussion/cleo>`_
    - `motionless (py3) <https://github.com/fmaussion/motionless>`_
