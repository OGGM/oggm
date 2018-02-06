# Python imports
from os import path
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

# Initialize OGGM and set up the run parameters
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# Read RGI
rgidf = salem.read_shapefile(path.join(WORKING_DIR, 'RGI_example_glaciers',
                                       'RGI_example_glaciers.shp'))
# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting OGGM run')
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Initialize from existing directories
gdirs = workflow.init_glacier_regions(rgidf)

# We can step directly to a new experiment!
# Random climate representative for the recent climate (1985-2015)
# This is a kinf of "commitment" run
execute_entity_task(tasks.run_random_climate, gdirs,
                    nyears=200, y0=2000, seed=1,
                    output_filesuffix='_commitment')

# Compile output
log.info('Compiling output')
utils.compile_run_output(gdirs, filesuffix='_commitment')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
