import os
import numpy as np
import oggm
import pandas as pd
import geopandas as gpd

from oggm import tasks
from oggm import cfg, utils
from oggm.utils import get_demo_file
from oggm.tests.funcs import get_test_dir
from oggm.core import climate

testdir = os.path.join(get_test_dir(), 'benchmarks', 'track_hef')


class hef_prepro:

    def setup_cache(self):

        # test directory
        utils.mkdir(testdir, reset=True)

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 120

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=testdir)

        tasks.define_glacier_region(gdir, entity=entity)
        tasks.glacier_masks(gdir)
        tasks.compute_centerlines(gdir)
        tasks.initialize_flowlines(gdir)
        tasks.compute_downstream_line(gdir)
        tasks.compute_downstream_bedshape(gdir)
        tasks.catchment_area(gdir)
        tasks.catchment_intersections(gdir)
        tasks.catchment_width_geom(gdir)
        tasks.catchment_width_correction(gdir)
        tasks.process_custom_climate_data(gdir)
        tasks.mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf)
        tasks.local_mustar(gdir, tstar=res['t_star'][-1],
                           bias=res['bias'][-1],
                           prcp_fac=res['prcp_fac'])
        tasks.apparent_mb(gdir)

        tasks.prepare_for_inversion(gdir)
        tasks.volume_inversion(gdir)

        return gdir

    def track_average_slope(self, gdir):
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['slope_angle'])
        return np.mean(out)

    def track_average_width(self, gdir):
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['width'])
        return np.mean(out)

    def track_rectangular_ratio(self, gdir):
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['is_rectangular'])
        return np.sum(out) / len(out)

    def track_mustar(self, gdir):
        df = pd.read_csv(gdir.get_filepath('local_mustar')).iloc[0]
        return df['mu_star']

    def track_bias(self, gdir):
        df = pd.read_csv(gdir.get_filepath('local_mustar')).iloc[0]
        return df['bias']

    def track_inversion_volume(self, gdir):
        inv_cls = gdir.read_pickle('inversion_output')
        vol = 0
        for cl in inv_cls:
            vol += np.sum(cl['volume'])
        return vol * 1e-9
