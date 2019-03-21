# Python imports
import time
import logging

# Libs
import matplotlib.pyplot as plt

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.utils import get_demo_file

# Module logger
log = logging.getLogger(__name__)

# For timing the run
start = time.time()

# Initialize OGGM and set up the default run parameters
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = utils.gettempdir('OGGM_spinup_run')
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing?
cfg.PARAMS['use_multiprocessing'] = True

# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large
cfg.PARAMS['border'] = 80

# Go - initialize glacier directories
gdirs = workflow.init_glacier_regions(['RGI60-11.00897'], from_prepro_level=4)

# Additional climate file (CESM)
cfg.PATHS['cesm_temp_file'] = get_demo_file('cesm.TREFHT.160001-200512'
                                            '.selection.nc')
cfg.PATHS['cesm_precc_file'] = get_demo_file('cesm.PRECC.160001-200512'
                                             '.selection.nc')
cfg.PATHS['cesm_precl_file'] = get_demo_file('cesm.PRECL.160001-200512'
                                             '.selection.nc')
execute_entity_task(tasks.process_cesm_data, gdirs)

# Run the last 200 years with the default starting point (current glacier)
# and CESM data as input
execute_entity_task(tasks.run_from_climate_data, gdirs,
                    climate_filename='gcm_data',
                    ys=1801, ye=2000,
                    output_filesuffix='_no_spinup')

# Run the spinup simulation - t* climate with a cold temperature bias
execute_entity_task(tasks.run_constant_climate, gdirs,
                    nyears=100, bias=0, temperature_bias=-0.5,
                    output_filesuffix='_spinup')
# Run a past climate run based on this spinup
execute_entity_task(tasks.run_from_climate_data, gdirs,
                    climate_filename='gcm_data',
                    ys=1801, ye=2000,
                    init_model_filesuffix='_spinup',
                    output_filesuffix='_with_spinup')

# Compile output
log.info('Compiling output')
utils.compile_glacier_statistics(gdirs)
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
