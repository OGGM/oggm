# Python imports
from os import path
import shutil
import zipfile
import oggm

# Module logger
import logging
log = logging.getLogger(__name__)

# Libs
import salem

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

# We use intersects
# (this is slow, it could be replaced with a subset of the global file)
rgi_dir = utils.get_rgi_intersects_dir(version='5')
cfg.set_intersects_db(path.join(rgi_dir, '00_rgi50_AllRegs',
                                'intersects_rgi50_AllRegs.shp'))

# Pre-download other files which will be needed later
utils.get_cru_cl_file()
utils.get_cru_file(var='tmp')
utils.get_cru_file(var='pre')

# Download the RGI file for the run
# We us a set of four glaciers here but this could be an entire RGI region,
# or any glacier list you'd like to model
dl = 'https://www.dropbox.com/s/6cwi7b4q4zqgh4a/RGI_example_glaciers.zip?dl=1'
with zipfile.ZipFile(utils.file_downloader(dl)) as zf:
    zf.extractall(WORKING_DIR)
rgidf = salem.read_shapefile(path.join(WORKING_DIR, 'RGI_example_glaciers',
                                       'RGI_example_glaciers.shp'))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting OGGM run')
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Preprocessing tasks
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
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks -- only data IO and tstar interpolation!
execute_entity_task(tasks.process_cru_data, gdirs)
tasks.distribute_t_stars(gdirs)
execute_entity_task(tasks.apparent_mb, gdirs)

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
execute_entity_task(tasks.random_glacier_evolution, gdirs,
                    nyears=200, bias=0, seed=1,
                    filesuffix='_tstar')

# Compile output
log.info('Compiling output')
utils.glacier_characteristics(gdirs)
utils.compile_run_output(gdirs, filesuffix='_tstar')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
