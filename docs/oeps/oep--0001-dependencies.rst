==========================================================
OEP-0001: Package dependencies and reproducibility in OGGM
==========================================================

:Authors: Fabien Maussion, Timo Rothenpieler, Alex Jarosch
:Status: Largely implemented
:Created: 11.11.2018


Abstract
--------

OGGM relies on a large number of external python packages (dependencies).
Many of them have complex dependencies themselves, often compiled binaries
(for example `rasterio`, which relies on a C package: `GDAL`).

The complexity of this dependency tree as well as the permanent updates of
both OGGM and its dependencies has lead to several unfortunate situations
in the past: this involved a lot of maintenance work for the OGGM
developers that had little or nothing to do with the model itself.

Furthermore, while the vast majority of the dependency updates are
without consequences, some might change the model results. As an example,
updates in the interpolation routines of GDAL/rasterio can change the
glacier topography in a non-traceable way for OGGM. This is an obstacle
to `reproducible science <https://en.wikipedia.org/wiki/Reproducibility>`_,
and we should try to avoid these situations.

OGGM, as a high level "top of the dependency pyramid" software, does not have
the same requirements in terms of being always up-to-date as, say, general
purpose libraries like pandas or matplotlib. With this document, the OGGM
developers attempt to define a clear policy for handling package dependencies
in the OGGM project. This policy can be seen as a set of "rules" or
"guidelines" that we will try to follow as much as possible.


Example situations
------------------

Here are some example of situations that happened in the past or might happen
in the future:

**Situation 1**
   A new user installs OGGM with `conda install oggm-deps` and the installation 
   fails because one of one of the dependencies. This 
   is a problem in conda (or conda-forge) and has nothing to do with OGGM.
   
**Situation 2**
   A new user installs OGGM with `conda install oggm-deps`: the installation 
   succeeds, but the OGGM tests fail to pass. One of our dependencies 
   renamed a functionality or deprecated it, and we didn't update OGGM in time.
   This requires action in OGGM itself, but we don't always have time to 
   fix it quickly enough, leading to an overall bad user experience.
   
**Situation 3**
   A developer writes a new feature and sends a pull-request: the tests pass
   on her machine but not on Travis. The failing tests are unrelated to the PR:
   it is one of our dependency update that broke something in OGGM. 
   This a bad experience for new developers, and it is the job of an OGGM core 
   developer to solve the problem. 
   
**Situation 4**
   A developer writes a quantitative test of the type:
   "the model simulation should yield a volume of xx.xxx km3 ice": the test
   passes locally but fails on Travis. Indeed, the glacier topography is 
   slightly different on Travis because of difference in the installed GDAL
   version, yielding non-negligible differences in model results.
   
**Situation 5**
   Worst case scenario: someone tries to replicate the results of a simulation
   published in a paper, and her results are off by 10% to the published 
   ones. A reason could be that the system and/or 
   package dependencies are different for the new simulation environment. But
   which results are the correct ones? 
  
These situations are frequent and apply to any project with complex 
dependencies (not only OGGM). **These problems currently do not have a simple,
"out-of-the box" solution**. This document attempts to prioritize some of the 
issues and provide guidelines for how to handle software dependencies 
within the OGGM framework.


Goals
-----

Here is a set of goals that we should always try to follow:

1. Stability is more important than dependency updates.
2. A *standard dependency list* of the names and fixed version number of the 
   major OGGM dependencies should be defined and documented. 
   The standard dependency list has 
   its own version number which is decoupled from the OGGM one.
3. Updates of the standard dependency list should be rare, well justified and
   documented. Examples for updates include: a new feature we 
   want to use, a performance increase, or regular updates to keep track with 
   the scientific python stack (e.g. twice a year).
4. It should be possible to use a python environment fixed to the standard
   dependency list on all these platforms: the Bremen cluster, on Travis, 
   on a local linux machine and on a university cluster with singularity.
5. The latest OGGM code should always run error-free on the latest standard 
   dependency list. Older OGGM versions should have a standard dependency list 
   version number attached to them, and scientific publications as well.
6. As far as possible, OGGM should run on the latest version of all packages
   as well. This rule is less important than Rule 5 and should not require
   urgent handling by the OGGM developers.
7. It should be possible for a new user to create a working environment based 
   on the standard dependency list from scratch, either with a meta-package
   (i.e. `conda install oggm-deps`) or a file 
   (`conda install oggm-deps.yml` or `pip install requirements.txt`).
8. If 7 fails, it should be possible for a user to create a working environment 
   based on the latest dependencies from scratch, either with a meta-package
   (i.e. `conda install oggm-deps-latest`) or a file 
   (`conda install oggm-deps-latest.yml` or 
   `pip install requirements-latest.txt`).
9. We recommend users to define a fixed environment for OGGM. We do not take 
   responsibility if users update packages on their own afterwards.
10. OGGM should provide tools to quickly and easily test a user installation.
11. OGGM should always work on Linux. Because of these dependency problems,
    we make no guarantee that OGGM will work on other operating systems 
    (Mac OSX, Windows).


Standard dependency list and updates (goals 1, 2, 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest and all previous standard dependency lists can be found on the
github repository: `<https://github.com/OGGM/OGGM-dependency-list>`_

Discussions about whether the standard dependency list should be updated or not
will take place on this repository or the main OGGM repository.


Docker and Singularity containers (goal 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to guarantee consistency accross platforms and over longer
periods of time are `containers <https://www.docker.com/resources/what-container>`_.
A container image is a lightweight, standalone, executable package of software 
that includes everything needed to run an application: code, runtime, system 
tools, system libraries and settings.

The most widely used container platform is `Docker <https://www.docker.com>`_.
For security and preformance reasons, HPC centers prefer to use 
`Singularity <https://www.sylabs.io>`_. Fortunately, Singularity containers
can be built from Docker images.

OGGM maintains an Ubuntu container image that users can download and use 
for free, and convert it to a singularity image for use on HPC.

The images can be found at: `<https://hub.docker.com/r/oggm/oggm>`_

The build scripts can be found at `<https://github.com/OGGM/OGGM-Docker>`_

**OGGM releases will point to a specific version of the Docker image
for reproducible results over time**.


Travis CI (goals 5, 6 and 11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Travis CI <https://travis-ci.org/OGGM/oggm>`_ is the tool we use for continuous
testing of the OGGM software. **The tests should run on the stable
Docker image built with the standard dependency list**. Optionally,
we will monitor the tests on the latest image as well, but the tests are 
not guaranteed to pass ("allowed failures").

**The main OGGM reporistory will not test on other platforms than Linux**. 
We might run the tests for other platforms as well, but this is without 
guarantee and should happen on a separate repository (e.g. on 
`<https://github.com/OGGM/OGGM-dependency-list>`_).


pip and conda (goals 7 and 8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Docker and Singularity containers are the most secure and consistent way to 
run OGGM. However, they require some knowledge about containers and the 
command line, and they still do not belong to the standard set of tools 
of most scientists.

Therefore, we should help and support users in installing OGGM dependencies
"the standard way", i.e. using pip or conda. We can do this by maintaining 
and testing the standard and latest dependency lists on various platforms
as often as possible. When problems arise, we can attempt to fix them but
make no guarantee for the problems generated upstream, i.e. problems which are 
unrelated to OGGM itself.


Check install (goal 10)
~~~~~~~~~~~~~~~~~~~~~~~

The user will have two complementary ways to test the correct installation
of OGGM:

- `pytest --pyargs oggm --run-slow --mpl` will run the test suite. This test
  suite does not contain quantitative tests, i.e. it does not guarantee
  consistency of model results accross platforms
- `oggm.check_install()` will be a top level function performing quantitative 
  tests to see if user results are consistent with the benchmark container.
