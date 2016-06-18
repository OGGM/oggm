.. _mass-balance:

Mass-balance
============

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)
    np.set_printoptions(threshold=10)

Creating a DataArray
~~~~~~~~~~~~~~~~~~~~

.. ipython:: python

    data = np.random.rand(4, 3)
    locs = ['IA', 'IL', 'IN']
    times = pd.date_range('2000-01-01', periods=4)
    foo = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'])
    foo


Only ``data`` is required; all of other arguments will be filled
in with default values: