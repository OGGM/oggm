"""Setup file for the Salem package.

   Adapted from the Python Packaging Authority template."""

from setuptools import setup, find_packages  # Always prefer setuptools
from codecs import open  # To use a consistent encoding
from os import path, walk
import importlib

# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def check_dependencies(package_names):
    """Check if packages can be imported, if not throw a message."""
    not_met = []
    for n in package_names:
        try:
            _ = importlib.import_module(n)
        except ImportError:
            not_met.append(n)
    if len(not_met) != 0:
        errmsg = "Following packages could not be found: "
        raise ImportError(errmsg + ', '.join(not_met))

req_packages = ['numpy',
                'scipy',
                'six',
                'pyproj',
                'pandas',
                'matplotlib',
                'joblib']

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
    name='OGGM',
    version='0.0.1',
    description='Open Global Glacier Model',
    long_description=long_description,
    # The project's main homepage.
    url='https://github.com/OGGM/oggm',
    # Author details
    author='OGGM developers',
    author_email='fabien.maussion@uibk.ac.at',
    # License
    license='GPLv3+',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha  4 - Beta  5 - Production/Stable
        'Development Status :: 4 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License '
        'v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    # What does your project relate to?
    keywords=['glacier', 'climate'],
    # Find packages automatically
    packages=find_packages(exclude=['docs']),
    # Decided not to let pip install the dependencies, this is too brutal
    install_requires=[],
    # additional groups of dependencies here (e.g. development dependencies).
    extras_require={},
    # data files that need to be installed
    package_data={'oggm.tests': list(file_walk('oggm/tests/baseline_images',
                                               remove='oggm/tests/')),
                  'oggm': ['params.cfg']},

    # Old
    data_files=[],
    # Executable scripts
    entry_points={},
)
