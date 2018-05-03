# Python imports
import os

# Libs
import geopandas as gpd
import shapely.geometry as shpg

# Locals
import oggm
import oggm.cfg as cfg
from oggm import utils, workflow

# For timing the run
import time
start = time.time()

# Initialize OGGM and set up the default run parameters
cfg.initialize()
rgi_version = '61'
rgi_region = '11'  # Alps

# Local working directory (where OGGM will write its output)
WORKING_DIR = os.path.join(os.path.expanduser('~'), 'tmp', 'OGGM_Rofental')
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# We use intersects
path = utils.get_rgi_intersects_region_file(rgi_region, version=rgi_version)
cfg.set_intersects_db(path)

# RGI file
path = utils.get_rgi_region_file(rgi_region, version=rgi_version)
rgidf = gpd.read_file(path)

# Get the Rofental Basin file
path = utils.get_demo_file('rofental_hydrosheds.shp')
basin = gpd.read_file(path)

# Take all glaciers in the Rhone Basin
in_bas = [basin.geometry.contains(shpg.Point(x, y))[0] for
          (x, y) in zip(rgidf.CenLon, rgidf.CenLat)]
rgidf = rgidf.loc[in_bas]
# Store them for later
rgidf.to_file(os.path.join(WORKING_DIR, 'rgi_rofental.shp'))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

print('Starting OGGM run')
print('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf)

# Tasks shortcuts - see the next examples for more details
workflow.gis_prepro_tasks(gdirs)
workflow.climate_tasks(gdirs)
workflow.inversion_tasks(gdirs)

# Compile output
print('Compiling output')
utils.glacier_characteristics(gdirs)
utils.write_centerlines_to_shape(gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
print('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))
