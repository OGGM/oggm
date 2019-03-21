# Python imports
import logging

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks

# For timing the run
import time
start = time.time()

# Module logger
log = logging.getLogger(__name__)

# Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='WORKFLOW')

# Here we override some of the default parameters
# How many grid points around the glacier?
# We make it small because we want the model to error because
# of flowing out of the domain
cfg.PARAMS['border'] = 80

# This is the important bit!
# We tell OGGM to continue despite of errors
cfg.PARAMS['continue_on_error'] = True

# Local working directory (where OGGM will write its output)
WORKING_DIR = utils.gettempdir('OGGM_Errors')
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

rgi_ids = ['RGI60-11.00897', 'RGI60-11.01450', 'RGI60-11.03295']

log.workflow('Starting OGGM run')
log.workflow('Number of glaciers: {}'.format(len(rgi_ids)))

# Go - get the pre-processed glacier directories
gdirs = workflow.init_glacier_regions(rgi_ids, from_prepro_level=4)

# We can step directly to the experiment!
# Random climate representative for the recent climate (1985-2015)
# with a negative bias added to the random temperature series
workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                             nyears=150, seed=0,
                             temperature_bias=-1)

# Write the compiled output
utils.compile_glacier_statistics(gdirs)
utils.compile_run_output(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.workflow('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
