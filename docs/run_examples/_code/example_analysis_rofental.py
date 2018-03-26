# Imports
from os import path
import geopandas as gpd
import matplotlib.pyplot as plt
from oggm.utils import get_demo_file

# Local working directory (where OGGM wrote its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_Rofental')

# Plot: the basin, the outlines and the centerlines
basin = gpd.read_file(get_demo_file('rofental_hydrosheds.shp'))
rgi = gpd.read_file(path.join(WORKING_DIR, 'rgi_rofental.shp'))
centerlines = gpd.read_file(path.join(WORKING_DIR, 'glacier_centerlines.shp'))

f, ax = plt.subplots()
basin.plot(ax=ax, color='k', alpha=0.2)
rgi.plot(ax=ax, color='C0')
centerlines.plot(ax=ax, color='C3')
plt.title('Rofental glaciers and centerlines')
plt.tight_layout()
plt.show()
