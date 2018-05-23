"""Setup file for the Salem package.

   Adapted from the Python Packaging Authority template."""

from setuptools import setup, find_packages  # Always prefer setuptools
from codecs import open  # To use a consistent encoding
from os import path, walk
import sys, warnings, importlib, re
import versioneer


DISTNAME = 'oggm'
LICENSE = 'GPLv3+'
AUTHOR = 'oggm Developers'
AUTHOR_EMAIL = 'fabien.maussion@uibk.ac.at'
URL = 'http://oggm.org'
CLASSIFIERS = [
        # How mature is this project? Common values are
        # 3 - Alpha  4 - Beta  5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License ' +
        'v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]

DESCRIPTION = 'Open Global Glacier Model'
LONG_DESCRIPTION = """
OGGM builds upon `Marzeion et al., (2012)`_ and intends to become a
global scale, modular, and open source model for glacier dynamics. The model
accounts for glacier geometry (including contributory branches) and includes
a simple (yet explicit) ice dynamics module. It can simulate past and
future mass-balance, volume and geometry of any glacier in a fully
automated workflow. We rely exclusively on publicly available data for
calibration and validation.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html

Links
-----
- Project website: http://oggm.org
- HTML documentation: http://oggm.readthedocs.io
- Source code: http://github.com/oggm/oggm
"""


def check_dependencies(package_names):
    """Check if packages can be imported, if not throw a message."""
    not_met = []
    for n in package_names:
        try:
            _ = importlib.import_module(n)
        except ImportError:
            not_met.append(n)
    if len(not_met) != 0:
        errmsg = "Warning: the following packages could not be found: "
        print(errmsg + ', '.join(not_met))


req_packages = ['numpy',
                'scipy',
                'pyproj',
                'pandas',
                'geopandas',
                'fiona',
                'matplotlib',
                'scikit-image',
                'Pillow',
                'joblib',
                'netCDF4',
                'shapely',
                'rasterio',
                'configobj',
                'pytest',
                'xarray',
                'progressbar2',
                'boto3']
check_dependencies(req_packages)


def file_walk(top, remove=''):
    """
    Returns a generator of files from the top of the tree, removing
    the given prefix from the root/file result.
    """
    top = top.replace('/', path.sep)
    remove = remove.replace('/', path.sep)
    for root, dirs, files in walk(top):
        for file in files:
            yield path.join(root, file).replace(remove, '')

setup(
    # Project info
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # The project's main homepage.
    url=URL,
    # Author details
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # License
    license=LICENSE,
    classifiers=CLASSIFIERS,
    # What does your project relate to?
    keywords=['geosciences', 'glaciers', 'climate', 'gis'],
    # We are a python 3 only shop
    python_requires='>=3.4',
    # Find packages automatically
    packages=find_packages(exclude=['docs']),
    # Decided not to let pip install the dependencies, this is too brutal
    install_requires=[],
    # additional groups of dependencies here (e.g. development dependencies).
    extras_require={},
    # data files that need to be installed
    package_data={'oggm': ['params.cfg']},
    # Old
    data_files=[],
    # Executable scripts
    entry_points={},
)
