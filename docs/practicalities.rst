.. currentmodule:: oggm

Performance, cluster environments and reproducibility
=====================================================

If you plan to run OGGM on more than a handful of glaciers, you might
be interested in using all processors available to you, whether you
are working on your laptop or on a cluster: see `Parallel computations`_
for how to do this.

For regional or global computations you will need to run
OGGM in `Cluster environments`_. Here we provide a couple of guidelines based
on our own experience with operational runs.

In `Reproducibility with OGGM`_, we discuss certain aspects of scientific
reproducibility with OGGM, and how we try to ensure that our results are
reproducible (it's not easy).


Parallel computations
---------------------

OGGM is designed to use the available resources as well as possible. For single
nodes machines but with more than one processor (e.g. for personal
computers) OGGM ships with a multiprocessing approach which is fairly simple to
use. For cluster environments with more than one machine, you can use `MPI`_.


Multiprocessing
~~~~~~~~~~~~~~~

Most OGGM computations are `embarrassingly parallel`_: they
are standalone operations to be realized on one single glacier entity and
therefore independent from each other
(they are called **entity tasks**, as opposed to the non-parallelizable
**global tasks**).

.. _embarrassingly parallel: https://en.wikipedia.org/wiki/Embarrassingly_parallel

When given a list of :ref:`glacierdir` on which to apply a given task,
the :py:func:`workflow.execute_entity_task` will distribute the operations on
the available processors using Python's `multiprocessing`_ module.
You can control this behavior with the ``use_multiprocessing`` config
parameter and the number of processors with ``mp_processes``.
The default in OGGM is:

.. ipython:: python

    from oggm import cfg
    cfg.initialize()
    cfg.PARAMS['use_multiprocessing']  # whether to use multiprocessing
    cfg.PARAMS['mp_processes']  # number of processors to use

``-1`` means that all available processors will be used.

.. _multiprocessing: https://docs.python.org/3.6/library/multiprocessing.html

MPI
~~~

OGGM can be run in a cluster environment, using standard mpi features.

.. note::

    In our own cluster deployment (see below), we chose *not* to use MPI, for
    simplicity. Therefore, our MPI support is currently untested: it should
    work, but let us know if you encounter any issue.

OGGM depends on mpi4py in that case, which can be installed either via conda::

    conda install -c conda-forge mpi4py

or pip::

    pip install mpi4py


``mpi4py`` itself depends on a working mpi environment, which is usually
supplied by the maintainers of your cluster.
On conda, it comes with its own copy of ``mpich``, which is nice and easy for
quick testing, but maybe undesirable for the performance of actual runs.

For an actual run, invoke any script using oggm via ``mpiexec``, and pass the
``--mpi`` parameter to the script itself::

    mpiexec -n 10 python ./run_rgi_region.py --mpi

Be aware that the first process with rank 0 is the manager process, that by
itself does not do any calculations and is only used to distribute tasks.
So the actual number of working processes is one lower than the number passed
to mpiexec/your clusters scheduler.

Cluster environments
--------------------

Here we describe some of the ways to use OGGM in a cluster environment. We
provide examples of our own set-up,
but your use case might vary depending on the
cluster type you are working with, who is administrating the cluster, etc.

Installation
~~~~~~~~~~~~

The installation procedure explained in :ref:`installing.oggm` should also
work in cluster environments. If you don't have admin rights,
installing with conda in your ``$HOME`` probably is the easiest option.
Once OGGM is installed, you can use your scripts (like the ones provided in
:ref:`run-set-up`). But you probably want to check if the tests pass and our
`Data storage`_ section below first!


If you are lucky, your cluster might support
`singularity containers <https://www.sylabs.io/>`_,
in which case we highly recommend their usage.


Singularity and docker containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For those not familiar with this concept,
`containers <https://www.docker.com/resources/what-container>`_ can be seen
as a lightweight, downloadable operating system which can run programs for
you. They are highly configurable, and come in many flavors.

.. important::

    Containers may be unfamiliar to some of you, but they are the best way
    to ensure traceable, reproducible results with any numerical model.
    We highly recommend their use.

The OGGM team (mostly: `Timo <https://github.com/TimoRoth>`_) provides,
maintains and updates a `Docker <https://www.docker.com/>`_ container.
You can see all OGGM containers `here <https://hub.docker.com/u/oggm>`_.
Our most important repositories are:

- `untested_base <https://hub.docker.com/r/oggm/untested_base>`_ is a
  container based on Ubuntu 18.04 and shipping with all OGGM dependencies
  installed on it. **OGGM is not guaranteed to run on these**, but we
  use them for our tests on
  `Travis <https://github.com/OGGM/oggm/blob/master/.travis.yml#L36-L51>`_.
- `base <https://hub.docker.com/r/oggm/base>`_ is built upon ``untested_base``,
  but is **pushed online only after the OGGM tests have run successfully
  on it**. Therefore, is provides a a more secure base for the model, although
  we cannot guarantee that past or future version of the model will always
  work on it.
- `oggm <https://hub.docker.com/r/oggm/oggm>`_ is built upon ``base`` each
  time that a new change is made to the OGGM codebase. They have OGGM
  installed, and **are guaranteed to run the OGGM version they ship with**.
  We cannot guarantee that past or future version of the model will always
  work on it.

To ensure reproducibility over time or different machines (and avoid
dependency update problems), **we recommend to use** ``base`` **or** ``oggm``
for your own purposes. Use ``base``
if you want to install your own OGGM version (don't forget to test it
afterwards!), and use ``oggm`` if you know which OGGM version you want.

Here is how we run a given fixed version of OGGM on our cluster, using
singularity to pull from a given fixed base::

    # All commands in the EOF block run inside of the container
    srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec docker://oggm/base:20181123 bash -s <<EOF
      set -e
      # Setup a fake home dir inside of our workdir, so we don't clutter the
      # actual shared homedir with potentially incompatible stuff
      export HOME="$OGGM_WORKDIR/fake_home"
      mkdir "\$HOME"
      # Create a venv that _does_ use system-site-packages, since everything is
      # already installed on the container. We cannot work on the container
      # itself, as the base system is immutable.
      python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env"
      source "$OGGM_WORKDIR/oggm_env/bin/activate"
      # Make sure latest pip is installed
      pip install --upgrade pip setuptools
      # Install a fixed OGGM version (here 20 Jan 2019)
      pip install "git+https://github.com/OGGM/oggm.git@c0c81cb612d6c020647ca7262705349a097b606f"
      # Finally, you can test OGGM with `pytest --pyargs oggm`, or run your script:
      YOUR_RUN_SCRIPT_HERE
    EOF


Some explanations:

- `srun <https://slurm.schedmd.com/srun.html>`_ is the command used by our
  job scheduler, SLURM. We specify the number of nodes and CPU's we'd like to
  use.
- ``singularity exec`` uses `Singularity <https://www.sylabs.io/>`_
  to execute a series of commands in a singularity container, which here
  simply is taken from our Docker container base (singularity
  `can run docker containers <https://www.sylabs.io/guides/3.1/user-guide/singularity_and_docker.html>`_).
  Singularity is preferred over Docker in cluster
  environments, mostly for security and performance reasons.
- we fix the container version we want to use to a certain
  `tag <https://hub.docker.com/r/oggm/base/tags>`_. With this, we are
  guaranteed to always use the same software versions across runs.
- it follows a number of commands to make sure we don't mess around with
  the system settings.
- then we install a **specific OGGM version**, here specified by its
  git hash (you can use a
  `git tag <https://stackoverflow.com/questions/13685920/install-specific-git-commit-with-pip>`_
  as well). Again, this is to ensure reproducibility and **document** which
  dependency and OGGM versions where used for a specific run.

We recommend to keep these scripts alongside our code and data, so that you
can trace them.

Data storage
~~~~~~~~~~~~

**‣ Input**

OGGM needs a certain amount of data to run (see :ref:`input-data`). Regardless
if you are using pre-processed directories or raw data, you will need to have
access to them from your environment. The default in OGGM is to download
the data and store it in a folder, specified in the ``$HOME/.oggm_config``
file (see ``dl_cache_dir`` in :ref:`system-settings`).

The structure of this folder is following the URLs from which the data
is obtained. You can either let OGGM fill it up at run time by downloading the
data (recommended if you do regional runs, i.e. don't need the entire data
set), but you might also want to pre-download everything using ``wget`` or
equivalent. OGGM will use the data as long as the url structure is OK.

System administrators can mark this folder as being "read only", in which
case OGGM will run only if the data is already there and exit with an error
otherwise.

**‣ Output**

.. warning::

    An OGGM run can write a significant amount of data. In particular, it
    writes a **very large number of folder and files**. This makes certain
    operations like copying or even deleting working directory folders quite
    slow.

Therefore, there are two ways to reduce the amount of data (and data files)
you have to deal with:

- the easiest way is to simply delete the glacier directories after a run
  and keep only the aggregated statistics files generated with the ``compile_``
  tasks (see :ref:`api-workflow`). A typical workflow would be to start from
  pre-processed directories, do the run, aggregate the results, copy the
  aggregated files for long term storage, and delete the working directory.
- the method above does not allow to go back to a single glacier
  for plotting or restarting a run, or to have a more detailed look at the
  glacier geometry evolution. If you want to do these things, you'll need to
  store the glacier directories as well. In order to reduce the number of files
  you'll have to deal with in this case, you can use the
  :py:func:`utils.gdir_to_tar` and :py:func:`utils.base_dir_to_tar` functions
  to create compressed, aggregated files of your directories. You can
  later initialize new directories from these tar files with the `from_tar`
  keyword argument in :py:func:`workflow.init_glacier_regions`.


Run per RGI region, not globally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance and data handling reasons, **we recommend to run the model on
single RGI regions independently** (or smaller regional entities). This is
a good compromise between performance (parallelism) and output file size as
well as other workflow considerations.

On our cluster, we use the following parallelization strategy: we use an
array of jobs to submit as many jobs as RGI regions (or experiments, if you
are running experiments on a single region for example), and each job is
run on one node only. This way, we avoid using MPI and do not require
communication between nodes, while still using our cluster at near 100%.


Reproducibility with OGGM
-------------------------

`Reproducibility <https://en.wikipedia.org/wiki/Reproducibility>`_ has become
an important topic recently, and we scientists have to do our best to make
sure that our research findings are "findable, accessible, interoperable, and
reusable" (`FAIR <https://www.nature.com/articles/sdata201618>`_).

Within OGGM, we do our best to follow the FAIR principles.

Source code and version control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source code of OGGM is located on `GitHub <https://github.com/OGGM/oggm>`_.
All the history of the codebase (and the tests and documentation) are
documented in the form of git commits.

When certain development milestones are reached, we release a new version
of the model using a so-called "tag" (version number). We will try to follow
our own `semantic versioning <https://semver.org/>`_ convention for
release numbers. We use MAJOR.MINOR.PATCH, with:

1. PATCH version number increase when the changes to the codebase are small
   increments or harmless bug fixes, and when we are confident that **the
   model output is not affected by these changes**.
2. MINOR version number increase when we add functionality or bug fixes which
   are not affecting the model behavior in a significant way. However,
   **it is possible that the model results are affected in some unpredictable
   ways, that we estimated to be "small enough"** to justify a minor release
   instead of major one. Unlike the original convention, we cannot always
   guarantee backwards compatibility in the OGGM syntax yet, because it is
   too costly. We'll try not to brake things at each release, though
3. MAJOR version number increase when we significantly change the OGGM syntax
   and/or the model results, for example by relying on a new default
   parametrization.

The current OGGM model version is:

.. ipython:: python

    import oggm
    oggm.__version__

We document the changes we make to the model on GitHub, and in the
:ref:`whats-new`.

Dependencies
~~~~~~~~~~~~

OGGM relies on a large number of external python packages (dependencies).
Many of them have complex dependencies themselves, often compiled binaries
(for example rasterio, which relies on a C package: GDAL).

The complexity of this dependency tree as well as the permanent updates of
both OGGM and its dependencies has lead to several unfortunate situations
in the past: this involved a lot of maintenance work for the OGGM developers
that had little or nothing to do with the model itself.

Furthermore, while the vast majority of the dependency updates are without
consequences, some might change the model results. As an example, updates in
the interpolation routines of GDAL/rasterio can change the glacier
topography in a non-traceable way for OGGM. This is an obstacle to
reproducible science, and we should try to avoid these situations.

Therefore, we have written a
"`roadmap <https://github.com/OGGM/oggm/blob/master/DEPENDENCIES_ROADMAP.md>`_"
as a tool to guide our decision regarding software dependencies in OGGM.
This document also lists some example situations affecting model users and
developers.

.. important::

    **The short answer is: use our docker/singularity containers for the
    most reproducible workflows.** Refer to `Singularity and docker containers`_
    for how to do that.


Dependence on hardware and input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The OGGM model will always be dependant on the input data (topography, climate,
outlines...). Be aware that while certain results are robust (like interannual
variability of surface mass-balance), other results are highly sensitive
to small changes in the boundary conditions. Some examples include:

- the ice thickness inversion at a specific location is highly sensitive to the
  local slope
- the equilibrium volume of a glacier under a constant climate is highly
  sensitive to small changes in the ELA or the bed topography
- more generally: growing large glaciers on longer periods are "more
  sensitive" to boundary conditions than shrinking small glaciers on shorter
  periods.

We haven't really tested the dependency of OGGM on hardware, but we expect
it to be low, as glaciers are not chaotic systems like the atmosphere.

Tools to monitor OGGM results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have developed a series of checks to monitor the changes in OGGM. They are
not perfect, but we constantly seek to improve them:

.. image:: https://coveralls.io/repos/github/OGGM/oggm/badge.svg?branch=master
        :target: https://coveralls.io/github/OGGM/oggm?branch=master
        :alt: Code coverage

.. image:: https://travis-ci.org/OGGM/oggm.svg?branch=master
    :target: https://travis-ci.org/OGGM/oggm
    :alt: Linux build status

.. image:: https://img.shields.io/badge/Cross-validation-blue.svg
    :target: https://cluster.klima.uni-bremen.de/~github/crossval/
    :alt: Mass-balance cross validation

.. image:: https://readthedocs.org/projects/oggm/badge/?version=latest
    :target: http://docs.oggm.org/en/latest
    :alt: Documentation status

.. image:: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
    :target: https://cluster.klima.uni-bremen.de/~github/asv/
    :alt: Benchmark status
