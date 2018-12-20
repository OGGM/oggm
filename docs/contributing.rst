.. _contributing:

********************
Contributing to OGGM
********************

All contributions, bug reports, bug fixes, documentation improvements,
enhancements and ideas are welcome! OGGM is still in an early development phase,
so most things are not written in stone and can probably be
enhanced/corrected/meliorated by anyone!

You can report issues or discuss OGGM on the
`issue tracker <https://github.com/OGGM/oggm/issues>`_.

**Copyright note**: this page is a shorter version of the excellent
`pandas <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_
documentation.

Working with the code
=====================

Before you contribute, you will need to learn how to work with
GitHub and the OGGM code base.

Version control, Git, and GitHub
--------------------------------

The code is hosted on `GitHub <https://github.com/OGGM/oggm>`_. To
contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitHub help pages <http://help.github.com/>`_.
* the `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`_.
* Matthew Brett's `Pydagogue <http://matthew-brett.github.com/pydagogue/>`_.

Getting started with Git
------------------------

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__ for
installing git, setting up your SSH key, and configuring git.
All these steps need to be completed before you can work seamlessly between
your local repository and GitHub.

Forking
-------

You will need your own fork to work on the code. Go to the `OGGM project
page <https://github.com/OGGM/oggm>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone git@github.com:your-user-name/oggm.git oggm-yourname
    cd oggm-yourname
    git remote add upstream git://github.com/OGGM/oggm.git

This creates the directory `oggm-yourname` and connects your repository to
the upstream (main project) oggm repository.

Creating a branch
-----------------

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch. Try to keep
any changes in this branch specific to one bug or feature.
You can have many shiny-new-features and switch in between them using the git
checkout command.

To update this branch, you need to retrieve the changes from the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the lastest oggm git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``stash`` them prior
to updating.  This will effectively store your changes and they can be reapplied
after updating.

.. _contributing.dev_env:

Creating a development environment
----------------------------------

An easy way to create a OGGM development environment is explained in
:ref:`installing.oggm`.


Contributing to the code base
=============================

Code standards
--------------

OGGM uses the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard.
There are several tools to ensure you abide by this standard,
and some IDE (for example PyCharm) will warn you if you don't follow PEP8.

Test-driven development/code writing
------------------------------------

OGGM is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
Like many packages, OGGM uses the `pytest testing system
<http://doc.pytest.org/en/latest/>`_
and the convenient
extensions in `numpy.testing
<http://docs.scipy.org/doc/numpy/reference/routines.testing.html>`_.


All tests should go into the ``tests`` subdirectory of OGGM.
This folder contains many current examples of tests, and we suggest looking to
these for inspiration.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

The tests can then be run directly inside your Git clone by typing::

    pytest .

The tests can run for several minutes. If everything worked fine, you
should see something like::

    ==== test session starts ====
    platform linux -- Python 3.4.3, pytest-3.0.5, py-1.4.31, pluggy-0.4.0
    rootdir:
    plugins:
    collected 92 items

    oggm/tests/test_graphics.py ..............
    oggm/tests/test_models.py .........s....sssssssssssssssss
    oggm/tests/test_prepro.py ...s................s.s...
    oggm/tests/test_utils.py ...sss..ss.sssss.
    oggm/tests/test_workflow.py ssss

    ===== 57 passed, 35 skipped in 102.50 seconds ====


You can safely ignore deprecation warnings and other DLL messages as long as
the tests end with ``OK``.

Often it is worth running only a subset of tests first around your changes
before running the entire suite.
This is done using one of the following constructs::

    pytest oggm/tests/[test-module].py
    pytest oggm/tests/[test-module].py:[TestClass]
    pytest oggm/tests/[test-module].py:[TestClass].[test_method]


Contributing to the documentation
=================================

Contributing to the documentation is of huge value. Something as simple as
rewriting small passages for clarity is a simple but effective way to
contribute.

About the documentation
-----------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx.pocoo.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://sphinx.pocoo.org/rest.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The OGGM documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``oggm/docs/``.

  The docstrings *should* provide a clear explanation of the usage of the
  individual functions (currently this is not the case everywhere, ufortunately),
  while the documentation in this folder consists of tutorial-like
  overviews per topic together with some other information (what's new,
  installation, etc).

- The docstrings follow the **Numpy Docstring Standard**, which is used widely
  in the Scientific Python community. This standard specifies the format of
  the different sections of the docstring. See `this document
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
  for a detailed explanation, or look at some of the existing functions to
  extend it in a similar manner.

- Some pages make use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build.


How to build the documentation
------------------------------

Requirements
~~~~~~~~~~~~

There are some extra requirements to build the docs: you will need to
have ``sphinx``, ``sphinx_rtd_theme``, ``numpydoc`` and ``ipython`` installed.

If you have a conda environment named ``oggm_env``, you can install the extra
requirements with::

      conda install -n oggm_env sphinx sphinx_rtd_theme ipython numpydoc

If you use pip, activate your python environment and install the requirements
with::

      pip install sphinx sphinx_rtd_theme ipython numpydoc


Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

So how do you build the docs? Navigate to your local
``oggm/docs/`` directory in the console and run::

    make html

Then you can find the HTML output in the folder ``oggm/docs/_build/html/``.

The first time you build the docs, it will take quite a while because it has to
run all the code examples and build all the generated docstring pages.
In subsequent evocations, sphinx will try to only build the pages that have
been modified.

If you want to do a full clean build, do::

    make clean
    make html

Open the following file in a web browser to see the full documentation you
just built::

    oggm/docs/_build/html/index.html

And you'll have the satisfaction of seeing your new and improved documentation!


Contributing your changes
=========================

Committing your code
--------------------

Keep style fixes to a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing::

    git status

If you have created a new file, it is not being tracked by git. Add it by typing::

    git add path/to/file-to-be-added.py

Doing 'git status' again should give something like::

    # On branch shiny-new-feature
    #
    #       modified:   /relative/path/to/file-you-added.py
    #

Finally, commit your changes to your local repository with an explanatory message::

    git commit -a -m 'added shiny feature'

You can make as many commits as you want before submitting your changes to OGGM,
but it is a good idea to keep your commits organised.

Pushing your changes
--------------------

When you want your changes to appear publicly on your GitHub page, push your
forked feature branch's commits::

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  git@github.com:yourname/oggm.git (fetch)
    origin  git@github.com:yourname/oggm.git (push)
    upstream        git://github.com/OGGM/oggm.git (fetch)
    upstream        git://github.com/OGGM/oggm.git (push)

Now your code is on GitHub, but it is not yet a part of the OGGM project.
For that to happen, a pull request needs to be submitted on GitHub.

Review your code
----------------

When you're ready to ask for a code review, file a pull request. Before you do, once
again make sure that you have followed all the guidelines outlined in this document
regarding code style, tests, and documentation. You should also
double check your branch changes against the branch it was based on:

#. Navigate to your repository on GitHub -- https://github.com/your-user-name/oggm
#. Click on ``Branches``
#. Click on the ``Compare`` button for your feature branch
#. Select the ``base`` and ``compare`` branches, if necessary. This will be ``master`` and
   ``shiny-new-feature``, respectively.

Finally, make the pull request
------------------------------

If everything looks good, you are ready to make a pull request.  A pull request is how
code from a local repository becomes available to the GitHub community and can be looked
at and eventually merged into the master version.  This pull request and its associated
changes will eventually be committed to the master branch and available in the next
release.  To submit a pull request:

#. Navigate to your repository on GitHub
#. Click on the ``Pull Request`` button
#. You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
#. Write a description of your changes in the ``Preview Discussion`` tab
#. Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code. If you need to make more changes, you can make them in
your branch, push them to GitHub, and the pull request will be automatically
updated.  Pushing them to GitHub again is done by::

    git push -f origin shiny-new-feature

This will automatically update your pull request with the latest code and restart the
Travis-CI tests.


Delete your merged branch (optional)
------------------------------------

Once your feature branch is accepted into upstream, you'll probably want to get rid of
the branch. First, merge upstream master into your branch so git knows it is safe to
delete your branch::

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can just do::

    git branch -d shiny-new-feature

Make sure you use a lower-case ``-d``, or else git won't warn you if your feature
branch has not actually been merged.

The branch will still exist on GitHub, so to delete it there do::

    git push origin --delete shiny-new-feature
