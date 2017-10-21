# Python imports
import logging
from os import path

# External modules
import matplotlib.pyplot as plt

# Locals
import oggm.cfg as cfg
from oggm import workflow, graphics
from oggm.core import flowline

# Module logger
log = logging.getLogger(__name__)

# Initialize OGGM
cfg.initialize()

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
cfg.PATHS['working_dir'] = WORKING_DIR

# Initialize from existing directories
# (note that we don't need the RGI file: this is going to be slow sometimes
# but it works)
gdirs = workflow.init_glacier_regions()

# Plot: we will show the state of all four glaciers at the beginning and at
# the end of the commitment simulation
f, axs = plt.subplots(2, 4, figsize=(20, 8))

for i in range(4):
    ax = axs[0, i]
    gdir = gdirs[i]
    # Use the model ouptut file to simulate the glacier evolution
    model = flowline.FileModel(gdir.get_filepath('model_run',
                                                 filesuffix='_commitment'))
    graphics.plot_modeloutput_map(gdirs[i], model=model, ax=ax,
                                  lonlat_contours_kwargs={'interval': 0})
    ax = axs[1, i]
    model.run_until(200)
    graphics.plot_modeloutput_map(gdirs[i], model=model, ax=ax,
                                  lonlat_contours_kwargs={'interval': 0})

plt.subplots_adjust(wspace=0.4, hspace=0.08, top=0.98, bottom=0.02)
plt.show()
