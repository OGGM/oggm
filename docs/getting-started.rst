.. currentmodule:: oggm

.. _getting-started:

Getting started
===============

.. ipython:: python
   :suppress:

    import os
    import numpy as np
    np.set_printoptions(threshold=10)

Although largely automatised, the OGGM model still requires some python
scripting to prepare and run a simulation. This documentation will guide you
through several examples to get you started.

First step: system settings for input data
------------------------------------------

OGGM will automatically download all the data it needs for a simulation at
run time. You can specify where on your computer these files should be stored
for later use. Let's start by opening a python interpreter and type in:

.. ipython:: python

    from oggm import cfg
    cfg.initialize()

At your very first import, this will do two things:

1. It will download a small subset of data used for testing and calibration.
   This data is located in your home directory, in a hidden folder
   called `.oggm`.
2. It will create a configuration file in your home folder, where you can
   indicate where you want to store further input data. This configuration
   file is also located in your home directory under the name ``.oggm_config``.


To locate this config file, you can type:

.. ipython:: python

    cfg.CONFIG_FILE

See :ref:`input-data` for an explanation of these entries.

.. important::

    The default settings will probably work for you, but we recommend to have
    a look at this file and set the paths to a directory
    where enough space is available: a minimum of 8 Gb for all climate data
    and glacier outlines is necessary. Topography data can quickly grow
    to several Gb as well, even for regional runs.


OGGM workflow
-------------

For a step by step tutorial of the entire OGGM workflow, download and run
the
:download:`getting started <https://raw.githubusercontent.com/OGGM/oggm/master/docs/notebooks/getting_started.ipynb>`
(right-click -> "Save link as") jupyter notebook.


OGGM run scripts
----------------

Refer to :ref:`run-set-up` for real-world applications.


OGGM Edu
--------

