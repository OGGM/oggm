OGGM standard projections
=========================

Since OGGM v1.6.1, we provide what we like to call "standard projections" of OGGM. A detailed description of the experimental-setup, information on the data structure, analysis of the data and a `brief comparison to Rounce et al. (2023) together with the corresponding jupyter notebooks are in the `OGGM/oggm-standard-projections-csv-files repository <https://github.com/OGGM/oggm-standard-projections-csv-files/tree/main>`_.  
These projections use `elevation-band flowlines <https://docs.oggm.org/en/stable/flowlines.html#elevation-bands-flowlines>`_, include the `dynamical spinup <https://docs.oggm.org/en/latest/dynamic-spinup.html>`_, the `new informed 3-step per-glacier geodetic calibration method <https://docs.oggm.org/en/latest/mass-balance-monthly.html>`_, and use the W5E5v2.0 climate dataset `(Lange and others, 2021) <https://doi.org/10.48364/ISIMIP.342217>`_ for calibration and a border of 160. 

We computed all GCMs and scenarios that are currently available at the OGGM cluster until 2100 and where available in a different run again until 2300. We have three different options available, using CMIP6, CMIP6 GCMs from the `bias-corrected CMIP6 ISIMIP3b GCMs <https://www.isimip.org/gettingstarted/isimip3b-bias-adjustment/>`_, and CMIP5 GCMs. An overview of the amount of GCMs per scenario and the resulting global volume changes is given in the figures below. However, you have to choose those scenarios that are suitable and representative for your study. We removed failing glaciers for the aggregation, a summary table is given `here <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/notebooks/missing_glacier_area_stats.png>`_. 

.. figure:: _static/global_glacier_volume_until2100_common_running_2100_oggm_v16.png
   :width: 80%
   :align: left

    Global glacier volume from 2000 to 2100 relative to 2020 (in %) for the different CMIP options using the common running glaciers until 2100. The amount of GCMs per scenario is given in the legend.


.. figure:: _static/global_glacier_volume_oggm_v16_2300.png
   :width: 60%
   :align: center

    Global glacier volume in 2300 relative to 2020 (in %) using all available climate scenarios by using the common running glaciers until 2100 and 2300. The amount of GCMs per scenario is given in the xtick labels.
    Attention: the GCMs until 2300 do not represent very well the ensemble until 2100. For example, the CMIP6 GCMs until 2300 are rather hotter until 2100 compared to the entire CMIP6 GCM ensemble. 


Currently we make these future CMIP forced global glacier simulations available in two different formats, raw and aggregated data.
- Aggregated data is provided for both glacier volume and area evolution in csv-files, aggregated globally and for every RGI region separately. 
    - `README <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/README.md>`_.
    - files available on the `OGGM/oggm-standard-projections-csv-files repository <https://github.com/OGGM/oggm-standard-projections-csv-files/tree/main>`_ which is also linked to citable Zenodo repository. On the OGGM cluster, you can also access the data directly from `https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm-standard-projections-csv-files/ <https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm-standard-projections-csv-files/>`_. 

- raw data is provided per-glacier for all interesting variables on netCDF files with 1000 glaciers each (e.g. monthly or annual runoff components, volume below sea-level, ... ).
    - `extended README <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/README_extended_per_glacier_files.md>`_
    - files available from the OGGM cluster:
        - for OGGM v1.6.1, it is: `https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2023.3/ <https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2023.3/>`_

The following jupyter notebooks give additional informations:
-  analysis of aggregated files is in `this notebook <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/notebooks/analyse_csv_files_1.6.1.ipynb>`_
-  regional or global aggregation workflow and analysis of the common running glaciers that run for all glaciers until 2100 or until 2100 and 2300 is `here <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/notebooks/aggregate_csv_files_1.6.1.ipynb>`_
- comparison to Rounce et al., 2023 is `here <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/notebooks/compare_oggm_1.6.1_to_rounce_et_al_2023.ipynb>`_
- some example analysis of the additional provided data raw oggm-output is `here <https://nbviewer.org/urls/cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/analysis_notebooks/workflow_to_analyse_per_glacier_projection_files.ipynb?flush_cache=true>`_. 




Data usage requirements
-----------------------

When you use the aggregated or the raw per-glacier data, please cite the dataset via:
- TODO: zenodo-link ...

In addition, cite `OGGM <https://doi.org/10.5194/gmd-12-909-2019>`_ and the CMIP option that you are using (references in this `README <https://github.com/OGGM/oggm-standard-projections-csv-files/blob/main/README.md>`_).
