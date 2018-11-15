# Python imports
import time
import logging

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task

# Module logger
log = logging.getLogger(__name__)

# For timing the run
start = time.time()

# Initialize OGGM and set up the run parameters
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = utils.gettempdir('OGGM_precalibrated_run')
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# Initialize from existing directories, no need for shapefiles
gdirs = workflow.init_glacier_regions()

log.info('Starting OGGM run')
log.info('Number of glaciers: {}'.format(len(gdirs)))

# We can step directly to a new experiment!
# Random climate representative for the recent climate (1985-2015)
# This is a kind of "commitment" run
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
