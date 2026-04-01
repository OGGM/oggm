Installing OGGM
===============

.. note::

   Did you know that you can try OGGM in your browser before installing it
   on your computer? Visit :doc:`cloud` for more information.

   Did you know that we provide standard projections for all glaciers in the
   world, and that you may not have to run OGGM for your use case?
   Visit :doc:`download-projections` for more information.

OGGM is `fully tested <https://github.com/OGGM/oggm/actions?query=event%3Apush>`_ with Python 3.11 to 3.14 on Linux.
macOS is not automatically tested but it should work on Mac just the same.

.. warning::

    **OGGM does not run natively on Windows.**
    If you are using Windows 10 or later, install the free
    `Windows subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_
    (WSL) and run OGGM from there. :doc:`installing-oggm-windows` provides specific
    instructions.

    .. toctree::
        :maxdepth: 1
        :hidden:

        installing-oggm-windows.rst


Dependencies
------------

OGGM itself is a pure Python package, but it relies on several other packages
which are not (e.g. GDAL, PROJ, etc.).
See OGGM's `recommended_env.yml <https://raw.githubusercontent.com/OGGM/oggm/master/docs/recommended_env.yml>`_ or
`pyproject.toml <https://github.com/OGGM/oggm/blob/master/pyproject.toml>`_
for a full list of dependencies.

Choose an installation method
-----------------------------

If you have never set up a Python environment before, you will need to
install a Python environment manager first.
OGGM dependencies can be installed using ``conda/mamba`` or ``uv``.
We recommend ``conda/mamba`` for most users, and ``uv`` for users
already familiar with it.

*If you are not familiar with Python and its*
`way too many package management systems`_,
*you might find all of this confusing. Be patient, read the docs,
and stay hydrated.*

.. _way too many package management systems: https://xkcd.com/1987/

.. tab-set::

   .. tab-item::  mamba/miniforge (recommended)

        .. include:: rst/install-mamba.rst

   .. tab-item::  uv (the new cool kid)

        .. include:: rst/install-uv.rst

   .. tab-item:: conda/anaconda

        .. include:: rst/install-conda.rst

Install OGGM and its dependencies
---------------------------------

.. tab-set::

    .. tab-item::  mamba

        Create and activate a mamba environment:

        .. code-block:: console

            mamba env create -n oggm_env python=3.13 oggm[full] -c conda-forge -c oggm
            conda activate oggm

        You are now ready to :ref:`test-oggm`!

    .. tab-item::  uv

        You may need to install GDAL separately first:

        .. code-block:: console

            sudo apt-get install gdal-bin libgdal-dev  # Linux (Debian distros)
            brew install gdal  # MacOS

        Launch Jupyter lab with OGGM fully installed:

        .. code-block:: console

            uvx --with oggm jupyter lab

        This installs a minimal version of OGGM into an isolated environment
        and runs a Jupyter server without polluting your base system or
        conda environment.

        Or you can install OGGM into a virtual environment:

        .. code-block:: console

            uv init  # run if the current directory has no pyproject.toml file
            uv venv  # create a virtual environment
            source .venv/bin/activate
            uv add oggm[full]
            uv sync

    .. tab-item:: conda

        Create and activate a conda environment:

        .. code-block:: console

            conda env create -n oggm_env python=3.13 oggm[full] -c conda-forge -c oggm
            conda activate oggm

        You are now ready to :ref:`test-oggm`!

.. _test-oggm:

Test OGGM
~~~~~~~~~

We **strongly** recommend testing OGGM prior to running it.
To test OGGM, activate your python environment, and run the tests:

.. code-block:: console

    pytest.oggm  --disable-warnings

The tests should run for 5 to 10 minutes (not much more).
If successful, you should see something like:

.. code-block:: console

    =================================== test session starts ====================================
    platform linux -- Python 3.10.6, pytest-7.1.3, pluggy-1.0.0
    Matplotlib: 3.5.3
    Freetype: 2.12.1
    rootdir: /home/mowglie/disk/Dropbox/HomeDocs/git/oggm-fork, configfile: pytest.ini
    plugins: anyio-3.6.1, mpl-0.150.0
    collected 373 items

    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_benchmarks.py ...s.ss            [  1%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_graphics.py ..........s...s....s [  7%]
    ss                                                                                   [  7%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_minimal.py ...                   [  8%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_models.py ...................... [ 14%]
    .........ssss.......ssss.ss.ss..ssssssssss..ssssssssssssssssssssssssssssssss.s       [ 35%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_numerics.py sssss.ss.sssss.s.sss [ 40%]
    ss.sss.sss.s                                                                         [ 43%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_prepro.py ..................s... [ 49%]
    ...........s....ss...ss..s.....ss.....sssssss..sssss....s.......                     [ 67%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_shop.py ss..........s.           [ 70%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_utils.py ....................... [ 76%]
    ....sss.......sss...s....................ss.s..ssss.ssssss..sss...s.ss.ss.ss....     [ 98%]
    disk/Dropbox/HomeDocs/git/oggm-fork/oggm/tests/test_workflow.py ssssss               [100%]

    ======================= 224 passed, 149 skipped in 217.03s (0:03:37) ======================


**Congratulations**, you are now set-up for the :doc:`getting-started` section!

The tests (without the ``--run-slow`` option) should run in 5 to 15 minutes.
If it takes longer, this may indicate something's wrong.

If you want to run the *entire* test suite (including graphics and slow running tests):

.. code-block:: console

    pytest.oggm --run-slow --mpl


.. _install-troubleshooting:

Installation troubleshooting
----------------------------

Please get in touch with us `on github <https://github.com/OGGM/oggm/issues>`_
if you encounter issues installing OGGM. Before doing so, a method that
has proven effective in the past is to use a conda environment file.

Download the conda environment file
`here <https://raw.githubusercontent.com/OGGM/oggm/master/docs/recommended_env.yml>`_
(right-click -> “save link as”) and install OGGM and its dependencies with:

.. code-block:: console

    mamba env create -f environment.yml


Install the latest OGGM version
-------------------------------

OGGM can be installed in two ways:

- as a **library** (recommended for most users):
    - **stable**: the latest stable release (e.g. v1.6.3)
    - **dev**: the development version, which may contain new features and bug fixes

- in **editable mode**: recommended if you want to modify or develop the model

In all cases, don't forget to :ref:`test-oggm` after each installation.


From github (library)
~~~~~~~~~~~~~~~~~~~~~

To install the latest **development** version (from github) as a **library**,
activate your conda environment first, **and uninstall oggm if it is already installed
(important)**:

.. code-block:: console

    conda/mamba uninstall oggm

Then, install the latest version from github:

.. code-block:: console

    pip install --upgrade git+https://github.com/OGGM/oggm.git

From source (editable)
~~~~~~~~~~~~~~~~~~~~~~

This is recommended for contributors or if you want to make changes to the OGGM source code.
It installs OGGM in "editable" mode, so any changes you make to the source code are
immediately applied when you import OGGM.

You will need `git`_ installed on your system.
Clone the latest repository version:

.. code-block:: console

    git clone https://github.com/OGGM/oggm.git
    cd oggm
    # Activate your python environment (conda or venv) here if you haven't already
    pip install -e .[full]  # if using pip
    uv sync --extra full  # if using uv

.. note::

    You can also update OGGM with a simple `git pull`_ from the root of the cloned repository.

.. _git pull: https://git-scm.com/docs/git-pull

.. _git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
