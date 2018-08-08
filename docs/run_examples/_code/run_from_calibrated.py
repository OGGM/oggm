# Python imports
from os import path
import oggm

# Module logger
import logging
log = logging.getLogger(__name__)

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task

# For timing the run
import time
start = time.time()

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
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

# Pre-download other files which will be needed later
utils.get_cru_cl_file()
utils.get_cru_file(var='tmp')
utils.get_cru_file(var='pre')

# Get the RGI glaciers for the run. We use a set of three glaciers here but
# this could be an entire RGI region, or any glacier list you'd like to model
rgi_list = ['RGI60-01.10299', 'RGI60-11.00897', 'RGI60-18.02342']
rgidf = utils.get_rgi_glacier_entities(rgi_list)

# We use intersects
db = utils.get_rgi_intersects_region_file(version='61', rgi_ids=rgi_list)
cfg.set_intersects_db(db)

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

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs)
# We use the default parameters for this run
execute_entity_task(tasks.volume_inversion, gdirs, glen_a=cfg.A, fs=0)
execute_entity_task(tasks.filter_inversion_output, gdirs)

# Final preparation for the run
execute_entity_task(tasks.init_present_time_glacier, gdirs)

# Random climate representative for the tstar climate, without bias
# In an ideal world this would imply that the glaciers remain stable,
# but it doesn't have to be so
execute_entity_task(tasks.run_random_climate, gdirs,
                    nyears=200, bias=0, seed=1,
                    output_filesuffix='_tstar')

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)
utils.compile_run_output(gdirs, filesuffix='_tstar')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
