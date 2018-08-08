# Python imports
from os import path
import oggm

# Module logger
import logging
log = logging.getLogger(__name__)

# Libs
import salem
import matplotlib.pyplot as plt

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.utils import get_demo_file

# For timing the run
import time
start = time.time()

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_spinup_run')
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# Here we override some of the default parameters
# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 100

# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = False

# We use intersects
cfg.set_intersects_db(utils.get_rgi_intersects_region_file('11', version='5'))

# Pre-download other files which will be needed later
utils.get_cru_cl_file()
utils.get_cru_file(var='tmp')
utils.get_cru_file(var='pre')

# Use the Hintereisferner RGI file for the run
rgidf = salem.read_shapefile(get_demo_file('Hintereisferner_RGI5.shp'))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting OGGM run')
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Preprocessing and climate tasks
task_list = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
    tasks.process_cru_data,
    tasks.local_mustar,
    tasks.apparent_mb,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Additional climate file (CESM)
cfg.PATHS['gcm_temp_file'] = get_demo_file('cesm.TREFHT.160001-200512.selection.nc')
cfg.PATHS['gcm_precc_file'] = get_demo_file('cesm.PRECC.160001-200512.selection.nc')
cfg.PATHS['gcm_precl_file'] = get_demo_file('cesm.PRECL.160001-200512.selection.nc')
execute_entity_task(tasks.process_cesm_data, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs)
# We use the default parameters for this run
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
execute_entity_task(tasks.filter_inversion_output, gdirs)

# Final preparation for the run
execute_entity_task(tasks.init_present_time_glacier, gdirs)

# Run the last 200 years with the default starting point (current glacier)
# and CESM data as input
execute_entity_task(tasks.run_from_climate_data, gdirs,
                    climate_filename='cesm_data',
                    ys=1801, ye=2000,
                    output_filesuffix='_no_spinup')

# Run the spinup simulation - t* climate with a cold temperature bias
execute_entity_task(tasks.run_constant_climate, gdirs,
                    nyears=100, bias=0, temperature_bias=-0.5,
                    output_filesuffix='_spinup')
# Run a past climate run based on this spinup
execute_entity_task(tasks.run_from_climate_data, gdirs,
                    climate_filename='cesm_data',
                    ys=1801, ye=2000,
                    init_model_filesuffix='_spinup',
                    output_filesuffix='_with_spinup')

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)
ds1 = utils.compile_run_output(gdirs, filesuffix='_no_spinup')
ds2 = utils.compile_run_output(gdirs, filesuffix='_with_spinup')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))

# Plot
f, ax = plt.subplots(figsize=(9, 4))
(ds1.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax, label='No spinup')
(ds2.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax, label='With spinup')
ax.set_ylabel('Volume (km$^3$)')
ax.set_xlabel('')
ax.set_title('Hintereisferner volume under CESM forcing')
plt.legend()
plt.tight_layout()
plt.show()
