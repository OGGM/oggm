======================================================
OEP-0002: OGGM on the cloud with JupyterHub and Pangeo
======================================================

:Authors: Fabien Maussion
:Status: Partly implemented (proof of concept for hub.oggm.org)
:Created: 03.07.2019


Abstract
--------

We plan to **set-up and deploy the Open Global Glacier Model in a scalable
cloud environment, and make this computing environment available for anyyone**.

We envision an online platform where people can log-in and get access to a
fully functional computing environment where they can run the OGGM model.
This environment will scale according to resources demand. It will be
personalized and persistent, so that a user can prepare some computations,
start them, log out, then log back in and still find the computing environment
he or she left earlier. The advantages for the user will be important:
scalable computing resources, no installation burden, no data download,
no version issues, user-friendly development environment, all that in a web
browser.

Users will be able to log-in and benefit from a containerized and persistent
environment where OGGM can run and where its output can be analyzed.
This environment will allow  exploratory experimenting and teaching as well as
"serious work" and analyses.


Glossary
--------

There is quite a few tools and concepts scattered around this proposal, let's
list the most important ones:

- `Jupyter Notebooks`_ are an open-source web application that allows you to
  create and share documents that contain live computer code, equations,
  visualizations and narrative text.
- `JupyterLab`_ is the web development environment where the notebooks can be
  run and edited.
- `JupyterHub`_ is the server which spawns JupyterLab in the containerized
  environment. It also handles user authentification and many other things.
- `MyBinder`_ is a versatile server which automatises the process of
  providing *any kind of environment* in a JupyterHub. MyBinder is a free to
  use deployment of the open-source Binder tool - but anyone can deploy a
  Binder server (e.g. Pangeo).
- `repo2docker`_ is used by MyBinder to specify the environments which need
  to be "containerized". It relies on environment files (pip, conda, apt, etc.)
  and is run as a command line tool.
- `docker`_ is the technology we use to create the environments where OGGM
  can run. Think of it as "software capsules" that you can download an run.
- `kubernetes`_ is the tool which does the job of scaling in the background.
  it needs to be installed on the cluster where JupyterHub is hosted, and
  somehow it works: when JupyterHub users come in, the cluster grows.
  kubernetes is designed for cloud, I wonder if it's useful on HPC or not.
- `Helm`_ is the tool which makes it easier to instal things on the kubernetes
  cluster. It's not too relevant here.
- `Dask`_ is an ecosystem of tools providing parallism tools to python. It is
  very cool, and sometimes a bit too fancy even.
- `Pangeo`_ is a a community of people who build scripts and tools that make
  is possible to work with all the tools above and do real computations (i.e.
  with big data and many processors).

.. _JupyterHub: https://jupyter.org/hub
.. _Pangeo: http://pangeo.io/
.. _JupyterLab: https://jupyterlab.readthedocs.io/en/stable/
.. _MyBinder: https://mybinder.org
.. _repo2docker: https://github.com/jupyter/repo2docker
.. _kubernetes: https://kubernetes.io
.. _Helm: https://helm.sh
.. _docker: https://www.docker.com/
.. _Dask: https://dask.org/
.. _Jupyter Notebooks: https://jupyter.org/

Motivation
----------

There is a general trend towards cloud computing in science, and we won't
repeat everything here. Let's discuss the main use cases here, from the
perspective of OGGM:

Big Data
  This is the main motivation behind proprietary platforms like
  `Google Earth Engine <https://earthengine.google.com/>`_ or the
  `Copernicus Data Store <https://cds.climate.copernicus.eu>`_. Huge amounts
  of data are available on some storage (cloud or HPC), and data providers
  want the users to work on these data locally (via their browser) rather
  then download them all. This is also the leitmotiv of the open
  source project `Pangeo`_.
  OGGM is not really a "big data problem". We do however rely on and provide
  large amount of data, and we could imagine a cloud-workflow where all these
  data exist on cloud or HPC and are available via JupyterHub.

Reproducible Science
  Making results reproducible is not only sharing code, it's also sharing the
  computing environment that was used to create those.
  TODO

Collaborative Workflows
  TODO

Ease of use
  TODO

Version control
  TODO


Status (03 Jul 2019)
--------------------

- We have two sorts of docker images for OGGM:
  `OGGM-Docker <https://github.com/OGGM/OGGM-Docker>`_, which are standard
  dockerfiles using ``pip install``. They are ligweight and tested, we use
  them on HPC with `Singularity <https://sylabs.io/docs/>`_. We also have
  repo2docker images (`repository <https://github.com/OGGM/oggm-edu-r2d>`_)
  which we need for MyBinder and JupyterHub (both need images created by
  JupyterHub). They are not lightweight because of conda.
- We have a 15k$ grant from Google (about 13k€) for cloud resources, to be used
  until June 2020. We plan to use this time to see if this endeavor is
  worth pursuing.
- MyBinder is what we use currently for `OGGM-Edu`_  It works well:
  anyone can run the tutorials and the educational material. We will keep this
  going. The only drawbacks are performance and persistence (MyBinder envs are
  temporary). The persistence problem is already a burden for multi-hour
  or multi-day workshops.
- Following the zero2jupyterhub instructions and with some learning by doing,
  we now have our own JupyterHub server running on google cloud:
  `hub.oggm.org`_ which is a vanilla zero2jupyterhub setup with our own
  images created with repo2docker. Pangeo and Met Office folks can log-in
  with github as well if you want to play around.
- I've learned that all these things take time. Scattered around several weeks,
  I still estimate to at least 10 full days of work invested from my side
  (lower estimate). I learned a lot but I need some help if we want to get
  this go further.

**This is enough as proof of concept and to allow more users to try the OGGM
model without installation burden** (we documented this possibility
in our docs recently). **It's not enough to do heavy work**. For more advanced
use we need dask, pangeo, and we need to put our data on cloud as well (see
roadmap).

.. _OGGM-Edu: https://edu.oggm.org
.. _hub.oggm.org: https://hub.oggm.org


Big-picture roadmap
-------------------

Assuming that we want to have that (i.e. OGGM running in JupyterHub for our
users, not only for tutorials and playing around), we have two main avenues:

1. **Continue on cloud**. If we do so, we need pangeo and dask, and we need to
   re-engineer parts of OGGM to work with dask multiproc and with cloud
   buckets for the input data.
2. **Continue on HPC**, once we have access to the big computer in Bremen. The
   tools in the backround would be slightly different, but for the users it
   would be exact same: "I log in, I request resources, I work".

Since we have no HPC yet, and have 15K from google, I'd like to try the cloud
idea a bit more.
If we don't want that, we can do a couple more tricks (personalized Hub, etc.)
and then leave it here.


Open questions
--------------

- **Is it possible to do vanilla mutliprocessing in Dask?** I think it is (with
  dask.delayed or dask.future), but it is not very well documented it
  seems (all use cases are more complex than ours). We **need** dask in order
  to use dask.distributed and dask.kubernetes.
- Does pangeo have an interest in our use case? For branding and
  federation it would be good for both projects, but does pangeo need to do
  anything at all?
- Does binder.pangeo.io also have the same time limits as MyBinder? (10 minutes
  of inactivity shuts down the server)

List of things we need to do
----------------------------

Not by order of importance, some things are very small. Bold mean harder.

- make it possible to install OGGM via pip in JupyterHub. This is already
  possible but only temporarily - i.e. install is lost at next login.
  It would be great so that people can use their own development versions to
  do runs.
- make a better splash screen for hub.oggm.org (see how pangeo is doing it)
- decide **how to use dask in order to scale and use OGGM in multiprocessing**
- **decide on the cloud bucket input data structure**. In the first place, I
  would just provide the pre-processed directories on bucket and see it we
  can read from them without having to download locally. On the long run we
  could add raw data as well
- **decide what do users do with output data**. This is the biggest issue in
  terms of costs I think. A first idea would be to delete the raw output
  on the go and only keep post-processed compiled output. This will come
  with its own challenges.

