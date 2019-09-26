======================================================
OEP-0002: OGGM on the cloud with JupyterHub and Pangeo
======================================================

:Authors: Fabien Maussion
:Status: Partly implemented (proof of concept for hub.oggm.org)
:Created: 03.07.2019


Abstract
--------

We plan to **set-up and deploy the Open Global Glacier Model in a scalable
cloud or HPC environment, and make this computing environment available
to our users via a web browser**.

We envision an online platform where people can log-in and get access to a
fully functional computing environment where they can run the OGGM model.
This platform will allow  exploratory experimenting and teaching, as well as
"serious work" and analyses. The resources will scale according to demand.
The user environment will be
personalized and persistent, so that our users can prepare some computations,
start them, log out, then log back in and still find the computing environment
he or she left earlier. The advantages for the users will be important:
scalable computing resources, no installation burden, no data download,
no version issues, a user-friendly development environment, all that in a web
browser.


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
  Making results reproducible is not only sharing code, it also means sharing
  the computing environment that was used to generate these results. This can
  be achieved via containerization (we already do that), but containers are
  still new and scary for non specialists (see "User friendlyness" below).
  There is a growing demand from publishers and society for reproducible
  computing workflows, and start-ups are flourishing on this new market
  (e.g. `code ocean <https://codeocean.com/>`_).
  In OGGM we would like to use open-source tools, of course.

User friendliness and innovation
  This is my main argument for web-based development workflows.
  Traditional HPC environments require to learn certain skills which aren't in
  the portfolio of many researchers: the linux command line, old-fashioned text
  editors, job schedulers, environment variables... This slows-down
  interactive, trial-and-error development processes and therefore innovation.
  Scientists are used to the notebooks and interactive python environments,
  where most of the development process takes place. If we had a way to combine
  the friendliness of JupyterLab with the power of HPC/Cloud, it would be a
  real win for both users and developers.

Environment control
  This is closely related with "Reproducible Science", but from a model
  developer and HPC provider perspective: with JupyterHub and containers, we
  have full control on the environment that users are using. This makes
  debugging and trouble shooting much easier, since we can exclude the
  usual "flawed set-up" from the user side.

Collaborative Workflows
  The self-documenting notebooks are easy to share accross users, encouraging
  team work. Another (rather vaue at this point) goal would be to allow
  "mulitple users" environments where people can develop scripts and notebooks
  collaboratively, but this is not top priority right now.


Status (03 Jul 2019)
--------------------

- We have two sorts of docker images for OGGM:
  (1) `OGGM-Docker <https://github.com/OGGM/OGGM-Docker>`_, which is a standard
  docker image generated from scratch using ``pip install``. These images
  are ligweight and well tested, we use
  them on HPC with `Singularity <https://sylabs.io/docs/>`_. (2)
  `repo2docker images <https://github.com/OGGM/r2d>`_,
  which we generate for MyBinder and JupyterHub (both need a certain user
  set-up which is best generated from repo2doker). They are not lightweight
  at all (because of conda).
- We have a 15k$ grant from Google (about 13k€) for cloud resources, to be used
  until June 2020. We plan to use this one year period as test phase to see if
  this endeavor isworth pursuing.
- We use MyBinder for `OGGM-Edu`_'s educational material and tutorials. It works
  very well. The only drawbacks are performance and the temporary nature of
  MyBinder environments. This is a real problem for multi-day workshops or
  classes.
- Thanks to the zero2jupyterhub instructions and with some trial-and-error,
  we now have our own JupyterHub server running on google cloud:
  `hub.oggm.org`_ which is a vanilla zero2jupyterhub setup with our own
  images created with repo2docker. The Pangeo organisation folks can log-in
  as well if you want to play with it.
- I've learned that all these things take time. Scattered around several weeks,
  I still estimate to at least 10 full days of work invested from my side
  (lower estimate). I learned a lot but I need some help if we want to have
  this go further.

**We have a proof of concept**, which allows new users to try the
OGGM model without installation burden.

**It is not enough to do heavy work**. For more advanced use cases we need
dask, pangeo, and we need to put our data on cloud as well, or we need to
set-up Jupyterhub on HPC (see roadmap).

.. _OGGM-Edu: https://edu.oggm.org
.. _hub.oggm.org: https://hub.oggm.org


Big-picture roadmap
-------------------

Assuming that we want to achieve this goal (a running instance of OGGM
in a JupyterHub server for reasearch applications), we can follow two main
strategies:

1. **Continue on cloud**. If we do so, we need pangeo and dask, and we need to
   re-engineer parts of OGGM to work with dask multiproc and with cloud
   buckets for the input data.
2. **Continue on HPC**, once we have access to the big computer in Bremen. The
   tools in the backround would be slightly different, but for the users it
   would be exact same: "I log in, I request resources, I work".

The two strategies have many similarities, and are worth discussing.
Since we have no HPC yet (and received 15K from google), I'd like to follow-up
on the cloud idea a little more.


Detailed roadmap
----------------

**Scaling**. This is relatively independant of cloud or HPC and should be done
anyway.

- **refactor the multiprocessing workflow of OGGM to use dask**. Once OGGM can
  run in the dask ecosystem, we will have access to all the nice tools that
  come with it, such as the task scheduler, the jupyterlab extension, and
  (most importantly) dask.distributed for automated scaling on both HPC and
  cloud/kubernetes.
- **build our docker images from pangeo-base instead**. This will come with
  dask pre-installed and allow a typical `pip isntall` workfklow, i.e. we
  can build upon our dockerfiles.
- **make hub.oggm.org point to these new images** -

**Data management and I/O**. This is the hardest part and the one which
will be most different whether we use cloud or HPC resources.

- **Input on cloud**: we need to put the input data on a read-only bucket. In a
  first step, we will make only pre-processed directories available. Ideally,
  OGGM will be able to start from and extract from bucket without downloading
  the data locally, i.e. the buckets look like a mounted disk and OGGM can
  read from them. The performance aspect is going to be interesting.
- **Output on cloud**: probably the biggest issue on cloud, not easy to solve.
  Disk space is quite expensive and users can easily generate huge amounts
  of data with OGGM (we are not really optimizing for data volume currently).
  I.e. we would have to provide tools to reduce the ouptut data amount, force
  the users to store their data elsewhere, etc. All that is not really
  attractive currently.
- **Input/ouput on HPC**: I imagine something not so different from what we
  have on HPC already.

**User environment**. Some things which are nice to have.

- make it possible to install OGGM via pip in JupyterHub. This is already
  possible but only temporarily - i.e. install is lost at next login.
  It would be great so that people can use their own development versions to
  do runs.
- make a better splash screen for hub.oggm.org (see how pangeo is doing it or
  use the pangeo one)
- documentation: use cases, examples, etc.
