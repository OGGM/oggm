import geopandas as gpd
import oggm
from oggm import cfg, tasks
from oggm.utils import get_demo_file, gettempdir

cfg.initialize()
cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')

base_dir = gettempdir('Flowlines_Docs')
entity = gpd.read_file(get_demo_file('HEF_MajDivide.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)

tasks.define_glacier_region(gdir, entity=entity)
tasks.glacier_masks(gdir)
tasks.compute_centerlines(gdir)
tasks.initialize_flowlines(gdir)
tasks.compute_downstream_line(gdir)
