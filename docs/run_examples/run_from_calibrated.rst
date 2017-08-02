.. currentmodule:: oggm

.. _run-from-calibrated:

Set-up a run with previously calibrated mass-balance
====================================================

This example shows how to run OGGM on a list of glaciers.

We use a previously precalibrated list of tstars for the
run, which means that we don't have to calibrate the mass balance anymore.


Full script
-----------

.. literalinclude:: _code/run_from_calibrated.py

If everything went well, you should see an output similar to::


    2017-08-02 18:09:29: oggm.cfg: Parameter file: /home/mowglie/Documents/git/oggm-official/oggm/params.cfg
    2017-08-02 18:10:11: __main__: Starting OGGM run
    2017-08-02 18:10:11: __main__: Number of glaciers: 4
    2017-08-02 18:10:11: oggm.workflow: Multiprocessing: using all available processors (N=4)
    2017-08-02 18:10:11: oggm.core.preprocessing.gis: RGI50-18.02342: define_glacier_region
    2017-08-02 18:10:11: oggm.core.preprocessing.gis: RGI50-01.10299: define_glacier_region
    (...)
    2017-08-02 18:11:55: oggm.core.models.flowline: RGI50-01.10299: default time stepping was successful!
    2017-08-02 18:13:01: oggm.core.models.flowline: RGI50-18.02342: default time stepping was successful!
    2017-08-02 18:13:01: __main__: Compiling output
    2017-08-02 18:13:01: __main__: OGGM is done! Time needed: 0:03:32

.. note::

   During the ``random_glacier_evolution`` task some numerical warnings might
   occur. These are expected to happen and caught by the solver, which then
   tries a more conservative time stepping scheme.

The ``random_glacier_evolution`` task can be replaced by any climate scenario
built by the user. For this you'll have to develop your own task, which will
be the topic of another example script.


Starting from a preprocessed state
----------------------------------

Now that we've gone through all the preprocessing steps, it is easy to run
new experiments:

.. literalinclude:: _code/run_from_calibrated_and_prepro.py

Which should have a much shorter runtime.

Some analyses
-------------

The output directory contains the compiled output files from the run. The
``glacier_characteristics.csv`` file contains various informations about each
glacier after the preprocessing, either obtained from the RGI directly
(location, name, glacier type...) or from the model itself (hypsometry,
inversion model output...).

Here is an example of how to read the file:

.. code-block:: python

    from os import path
    import pandas as pd
    WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
    df = pd.read_csv(path.join(WORKING_DIR, 'glacier_characteristics.csv'))
    print(df)

Output (reduced for clarity):

==============  ===============  =========  ========  ==============   ==================  ==============  ==============  ===============  ========================  =================  =================
rgi_id          name                cenlon    cenlat    rgi_area_km2   terminus_type         dem_max_elev    dem_min_elev    n_centerlines    longuest_centerline_km    inv_thickness_m    vas_thickness_m
==============  ===============  =========  ========  ==============   ==================  ==============  ==============  ===============  ========================  =================  =================
RGI50-18.02342  Tasman Glacier    170.238   -43.5653          95.216   Land-terminating              3662             715                7                  27.2195            186.38             187.713
RGI50-01.10299  Coxe Glacier     -148.037    61.144           19.282   Marine-terminating            1840               6                8                  11.6243            137.153            103.136
RGI50-11.00897  Hintereisferner    10.7584   46.8003           8.036   Land-terminating              3684            2447                3                   8.99975           107.642             74.2795
RGI50-08.02637  Storglaciaeren     18.5604   67.9042           3.163   Land-terminating              1909            1176                2                   3.54495            63.2469            52.362
==============  ===============  =========  ========  ==============   ==================  ==============  ==============  ===============  ========================  =================  =================


The run outpout is stored in netCDF files, and it can therefore be read with
other tools than python. We are used to python, and use the xarray package to
read the data:

.. code-block:: python

    import xarray as xr
    import matplotlib.pyplot as plt
    from os import path

    WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
    ds1 = xr.open_dataset(path.join(WORKING_DIR, 'run_output_tstar.nc'))
    ds2 = xr.open_dataset(path.join(WORKING_DIR, 'run_output_commitment.nc'))

    v1_km3 = ds1.volume * 1e-9
    v2_km3 = ds2.volume * 1e-9
    l1_km = ds1.length * 1e-3
    l2_km = ds2.length * 1e-3

    f, axs = plt.subplots(2, 4, figsize=(12, 4), sharex=True)

    for i in range(4):
        ax = axs[0, i]
        v1_km3.isel(rgi_id=i).plot(ax=ax, label='t*')
        v2_km3.isel(rgi_id=i).plot(ax=ax, label='Commitment')
        if i == 0:
            ax.set_ylabel('Volume [km3]')
            ax.legend(loc='best')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')

        ax = axs[1, i]

        # Length can need a bit of postprocessing because of some cold years
        # Where seasonal snow is thought to be a glacier...
        for l in [l1_km, l2_km]:
            roll_yrs = 5
            sel = l.isel(rgi_id=i).to_series()
            sel = sel.rolling(roll_yrs).min()
            sel.iloc[0:roll_yrs] = sel.iloc[roll_yrs]
            sel.plot(ax=ax)
        if i == 0:
            ax.set_ylabel('Length [m]')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Years')
        ax.set_title('')

    plt.tight_layout()
    plt.show()

This code snippet should produce the following plot:

.. figure:: ../_static/run_example.png
    :width: 100%

