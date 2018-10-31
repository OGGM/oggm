#!/bin/bash
set -e

echo "Installing numpy..."
pip install numpy

echo "Installing scipy..."
pip install scipy

echo "Installing pandas shapely cython matplotlib..."
pip install pandas shapely cython
pip install matplotlib

echo "Installing gdal fiona..."
pip install gdal==1.10.0 --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"
pip install fiona --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"

echo "Installing other packages..."
pip install pyproj rasterio Pillow geopandas netcdf4 scikit-image configobj joblib xarray motionless pytest pytest-mpl

echo "Installing git packages..."
pip install git+https://github.com/fmaussion/salem.git
pip install git+https://github.com/retostauffer/python-colorspace

echo "Done installing pip packages"

