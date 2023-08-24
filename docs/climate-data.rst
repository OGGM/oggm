Climate data
============

Here are the various climate datasets that OGGM can handle automatically, using the for instance
one of the following functions to pre-process the climate data:

.. code-block:: python

    from oggm.tasks import process_w5e5_data
    from oggm.tasks import import process_cru_data
    from oggm.tasks import import process_histalp_data
    from oggm.tasks import import process_ecmwf_data

.. warning::

.. _climate-w5e5:

W5E5
~~~~

As of v1.6, w5e5 [Lang_et_al_2021]_ is the standard dataset used by OGGM as baseline climate.
It is currently being used for all preprocessed directories. Over land the w5e5 data is
bias corrected ERA5 re-analysis data. The main reasons this climate product is currently being
used as the default baseline climate in OGGM, is that the (ISMIP)[https://www.isimip.org/]
simulations have been bias corrected using this dataset. Normally we always need to bias correct
the climate data when the using another climate (e.g. a GCM simulation) to force OGGM than the
baseline climate, that has been used for the calibration of the mass balance model. We no longer
need to bias correct these ISMIP climate simulations, when calibrating the mass balance model
with w5e5 a baseline climate. This is a big advantage, as we only use a simple bias correction
approach (the delta method) to bias correct climate data in other cases.

**When using these data, please refer to the original providers:**

Lange, S., Menz, C., Gleixner, S., Cucchi, M., Weedon, G. P., Amici, A., Bellouin, N.,
Müller Schmied, H., Hersbach, H., Buontempo, C. & Cagnazzo, C. (2021). WFDE5 over land
merged with ERA5 over the ocean (W5E5 v2.0). ISIMIP Repository.
https://doi.org/10.48364/ISIMIP.342217

.. [Lang_et_al_2021] Lange, S., Menz, C., Gleixner, S., Cucchi, M., Weedon, G. P., Amici,
A., Bellouin, N., Müller Schmied, H., Hersbach, H., Buontempo, C. & Cagnazzo, C. (2021).
WFDE5 over land merged with ERA5 over the ocean (W5E5 v2.0). ISIMIP Repository.
https://doi.org/10.48364/ISIMIP.342217

CRU
~~~

`CRU TS`_
data provided by the Climatic Research Unit of the University of East Anglia.
If asked to do so, OGGM will automatically download and unpack the
latest dataset from the CRU servers.

.. _CRU TS: https://crudata.uea.ac.uk/cru/data/hrg/

To download CRU data you can use the
following convenience functions:

.. code-block:: python

    from oggm.shop import cru
    cru.get_cl_file()
    cru.get_cru_file(var='tmp')
    cru.get_cru_file(var='pre')

.. warning::

    While the downloaded zip files are ~370mb in size, they are ~5.6Gb large
    after decompression!

The raw, coarse (0.5°) dataset is then downscaled to a higher resolution grid
(CRU CL v2.0 at 10' resolution [New_et_al_2002]_) following the anomaly mapping approach
described by Tim Mitchell in his `CRU faq`_ (Q25). Note that we don't expect
this downscaling to add any new information than already available at the
original resolution, but this allows us to have an elevation-dependent dataset
based on a presumably better climatology. The monthly anomalies are computed
following [Harris_et_al_2010]_ : we use standard anomalies for temperature and
scaled (fractional) anomalies for precipitation.

**When using these data, please refer to the original providers:**

Harris, I., Jones, P. D., Osborn, T. J., & Lister, D. H. (2014). Updated
high-resolution grids of monthly climatic observations - the CRU TS3.10 Dataset.
International Journal of Climatology, 34(3), 623–642. https://doi.org/10.1002/joc.3711

New, M., Lister, D., Hulme, M., & Makin, I (2002). A high-resolution data
set of surface climate over global land areas. Climate Research, 21(715), 1–25.
https://doi.org/10.3354/cr021001

.. _CRU faq: https://crudata.uea.ac.uk/~timm/grid/faq.html

.. [Harris_et_al_2010] Harris, I., Jones, P. D., Osborn, T. J., & Lister,
   D. H. (2014). Updated high-resolution grids of monthly climatic observations
   - the CRU TS3.10 Dataset. International Journal of Climatology, 34(3),
   623–642. https://doi.org/10.1002/joc.3711

.. [New_et_al_2002] New, M., Lister, D., Hulme, M., & Makin, I (2002). A high-resolution
   data set of surface climate over global land areas. Climate Research, 21(715),
   1–25. https://doi.org/10.3354/cr021001

ERA5 and CERA-20C
~~~~~~~~~~~~~~~~~

Since OGGM v1.4, users can also use reanalysis data from the ECMWF, the
European Centre for Medium-Range Weather Forecasts based in Reading, UK.
OGGM can use the
`ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_ (1979-2019, 0.25° resolution) and
`CERA-20C <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/cera-20c>`_  (1900-2010, 1.25° resolution)
datasets as baseline. One can also apply a combination of both, for example
by applying the CERA-20C anomalies to the reference ERA5 for example
(useful only in some circumstances).

**When using these data, please refer to the original provider:**

For example for ERA5:

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A.,
Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I.,
Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2019):
ERA5 monthly averaged data on single levels from 1979 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
(Accessed on < 01-12-2020 >), 10.24381/cds.f17050d7

HISTALP
~~~~~~~

OGGM can also automatically download and use the data from the `HISTALP`_
dataset (available only for the European Alps region, more details in [Chimani_et_al_2012]_.
The data is available at 5' resolution (about 0.0833°) from 1801 to 2014.
However, the data is considered spurious before 1850. Therefore, we
recommend to use data from 1850 onwards.

.. _HISTALP: http://www.zamg.ac.at/histalp/

.. [Chimani_et_al_2012] Chimani, B., Matulla, C., Böhm, R., Hofstätter, M.:
   A new high resolution absolute Temperature Grid for the Greater Alpine Region
   back to 1780, Int. J. Climatol., 33(9), 2129–2141, DOI 10.1002/joc.3574, 2012.

.. ipython:: python
   :suppress:

    fpath = "_code/prepare_hef.py"
    with open(fpath) as f:
        code = compile(f.read(), fpath, 'exec')
        exec(code)

.. ipython:: python
   :okwarning:

    @savefig plot_temp_ts.png width=100%

Any other climate dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

It is fairly easy to force OGGM with other datasets too. Recent publications have used
plenty of options, from ERA5-Land to regional reanalyses or more.


GCM data
~~~~~~~~

OGGM can also use climate model output to drive the mass balance model. In
this case we still rely on gridded observations (e.g. W5E5) for the reference
climatology and apply the GCM anomalies computed from a preselected reference
period. This method is often called the
`delta method <http://www.ciesin.org/documents/Downscaling_CLEARED_000.pdf>`_.

Visit our online tutorials to see how this can be done
(`OGGM run with GCM tutorial <https://oggm.org/tutorials/master/notebooks/run_with_gcm.html>`_).