"""OGGM tasks.

This module is simply a shortcut to the core functions
"""
from __future__ import absolute_import, division

# Entity tasks
from oggm.core.preprocessing.gis import define_glacier_region
from oggm.core.preprocessing.gis import glacier_masks
from oggm.core.preprocessing.centerlines import compute_centerlines
from oggm.core.preprocessing.centerlines import compute_downstream_lines
from oggm.core.preprocessing.centerlines import compute_downstream_bedshape
from oggm.core.preprocessing.geometry import catchment_area
from oggm.core.preprocessing.geometry import catchment_intersections
from oggm.core.preprocessing.geometry import initialize_flowlines
from oggm.core.preprocessing.geometry import catchment_width_geom
from oggm.core.preprocessing.geometry import catchment_width_correction
from oggm.core.preprocessing.climate import mu_candidates
from oggm.core.preprocessing.climate import process_cru_data
from oggm.core.preprocessing.climate import process_custom_climate_data
from oggm.core.preprocessing.climate import process_cesm_data
from oggm.core.preprocessing.climate import apparent_mb_from_linear_mb
from oggm.core.preprocessing.inversion import prepare_for_inversion
from oggm.core.preprocessing.inversion import volume_inversion
from oggm.core.preprocessing.inversion import filter_inversion_output
from oggm.core.preprocessing.inversion import distribute_thickness
from oggm.core.models.flowline import init_present_time_glacier
from oggm.core.models.flowline import random_glacier_evolution
from oggm.core.models.flowline import iterative_initial_glacier_search

# Global tasks
from oggm.core.preprocessing.climate import process_histalp_nonparallel
from oggm.core.preprocessing.climate import compute_ref_t_stars
from oggm.core.preprocessing.climate import distribute_t_stars
from oggm.core.preprocessing.climate import crossval_t_stars
from oggm.core.preprocessing.climate import quick_crossval_t_stars
from oggm.core.preprocessing.inversion import optimize_inversion_params
