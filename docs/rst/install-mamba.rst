``mamba`` is a faster drop-in replacement for ``conda``. We recommend it for its
very lightweight installation via miniforge. To install miniforge, follow `these instructions <https://github.com/conda-forge/miniforge?tab=readme-ov-file>`_

If you want to switch to miniforge from Anaconda, be aware of the following:

.. warning::

    Do not install miniforge *on top* of an existing conda installation! See
    `this issue <https://github.com/OGGM/oggm/issues/1571>`_ for context.
    If you have conda installed and want to switch to mamba + conda-forge,
    install miniforge in a different folder or, even better, uninstall Anaconda
    and start from scratch with miniforge.
