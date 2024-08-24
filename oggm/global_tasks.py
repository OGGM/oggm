"""OGGM global tasks.

This module is simply a shortcut to the core functions
"""
# flake8: noqa

from oggm.workflow import gis_prepro_tasks
from oggm.workflow import climate_tasks
from oggm.workflow import inversion_tasks
from oggm.workflow import calibrate_inversion_from_consensus
from oggm.workflow import merge_glacier_tasks
from oggm.utils import get_ref_mb_glaciers
from oggm.utils import write_centerlines_to_shape
from oggm.utils import compile_run_output
from oggm.utils import compile_climate_input
from oggm.utils import compile_task_log
from oggm.utils import compile_task_time
from oggm.utils import compile_glacier_statistics
from oggm.utils import compile_fixed_geometry_mass_balance
from oggm.utils import compile_climate_statistics
from oggm.utils import compile_ela
