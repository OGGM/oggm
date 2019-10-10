"""Setup file for the Salem package.

   Adapted from the Python Packaging Authority template."""

from setuptools import setup, find_packages  # Always prefer setuptools


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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]

DESCRIPTION = 'Open Global Glacier Model'
LONG_DESCRIPTION = """
**OGGM is a modular open source model for glacier dynamics**

The model accounts for glacier geometry (including contributory branches) and
includes an explicit ice dynamics module. It can simulate past and
future mass-balance, volume and geometry of (almost) any glacier in the world
in a fully automated and extensible workflow. We rely exclusively on publicly
available data for calibration and validation.

Links
-----
- Project website: http://oggm.org
- HTML documentation: http://docs.oggm.org
- Source code: http://github.com/oggm/oggm
"""


req_packages = ['numpy',
                'scipy',
                'pandas',
                'matplotlib>=2.0.0',
                'shapely',
                'requests',
                'configobj',
                'netcdf4',
                'xarray',
                ]


setup(
    # Project info
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # Version info
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version=True,
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
    python_requires='>=3.5',
    # Find packages automatically
    packages=find_packages(exclude=['docs']),
    # Include package data
    include_package_data=True,
    # Install dependencies
    install_requires=req_packages,
    # additional groups of dependencies here (e.g. development dependencies).
    extras_require={},
    # Executable scripts
    entry_points={
        'pytest11': ['pytest_oggm = oggm.pytest_plugin'],
        'console_scripts': [
            'oggm_prepro = oggm.cli.prepro_levels:main',
            'oggm_benchmark = oggm.cli.benchmark:main',
            'oggm_tdmdem90_login = oggm.cli.netrc_credentials:tandemx',
            'oggm_nasa_earthdata_login = oggm.cli.netrc_credentials:earthdata',
        ],
    },
)
