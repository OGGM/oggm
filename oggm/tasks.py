"""OGGM tasks.

This module is simply a shortcut to the core functions
"""
# Entity tasks
from oggm.core.gis import define_glacier_region
from oggm.core.gis import glacier_masks
from oggm.core.centerlines import compute_centerlines
from oggm.core.centerlines import compute_downstream_line
from oggm.core.centerlines import compute_downstream_bedshape
from oggm.core.centerlines import catchment_area
from oggm.core.centerlines import catchment_intersections
from oggm.core.centerlines import initialize_flowlines
from oggm.core.centerlines import catchment_width_geom
from oggm.core.centerlines import catchment_width_correction
from oggm.core.climate import mu_candidates
from oggm.core.climate import process_cru_data
from oggm.core.climate import process_custom_climate_data
from oggm.core.climate import process_cesm_data
from oggm.core.climate import local_mustar
from oggm.core.climate import apparent_mb
from oggm.core.climate import apparent_mb_from_linear_mb
from oggm.core.inversion import prepare_for_inversion
from oggm.core.inversion import volume_inversion
from oggm.core.inversion import filter_inversion_output
from oggm.core.inversion import distribute_thickness
from oggm.core.flowline import init_present_time_glacier
from oggm.core.flowline import random_glacier_evolution
from oggm.core.flowline import iterative_initial_glacier_search
from oggm.core.flowline import run_from_climate_data
from oggm.core.flowline import run_constant_climate

# Global tasks
from oggm.core.climate import process_histalp_nonparallel
from oggm.core.climate import compute_ref_t_stars
from oggm.core.climate import distribute_t_stars
from oggm.core.climate import crossval_t_stars
from oggm.core.climate import quick_crossval_t_stars
from oggm.core.inversion import optimize_inversion_params
