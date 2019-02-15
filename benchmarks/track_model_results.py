import os
import numpy as np
import oggm
import geopandas as gpd
import xarray as xr

from oggm import tasks
from oggm import cfg, utils, workflow
from oggm.exceptions import MassBalanceCalibrationError
from oggm.utils import get_demo_file
from oggm.tests.funcs import get_test_dir
from oggm.core import climate, massbalance
from oggm.workflow import execute_entity_task


class hef_prepro:

    testdir = os.path.join(get_test_dir(), 'benchmarks', 'track_hef')

    def cfg_init(self):

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['baseline_climate'] = 'CUSTOM'

    def setup_cache(self):

        utils.mkdir(self.testdir, reset=True)
        self.cfg_init()

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

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
        tasks.glacier_mu_candidates(gdir)
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf)
        tasks.local_t_star(gdir, tstar=res['t_star'],
                           bias=res['bias'])
        tasks.mu_star_calibration(gdir)

        tasks.prepare_for_inversion(gdir)
        tasks.mass_conservation_inversion(gdir)

        return gdir

    def track_average_slope(self, gdir):
        self.cfg_init()
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['slope_angle'])
        return np.mean(out)

    def track_average_width(self, gdir):
        self.cfg_init()
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['width'])
        return np.mean(out)

    def track_rectangular_ratio(self, gdir):
        self.cfg_init()
        cls = gdir.read_pickle('inversion_input')
        out = np.array([])
        for cl in cls:
            out = np.append(out, cl['is_rectangular'])
        return np.sum(out) / len(out)

    def track_mustar(self, gdir):
        self.cfg_init()
        df = gdir.read_json('local_mustar')
        assert df['mu_star_allsame']
        return df['mu_star_glacierwide']

    def track_bias(self, gdir):
        self.cfg_init()
        df = gdir.read_json('local_mustar')
        return df['bias']

    def track_mb_1980_avg(self, gdir):
        self.cfg_init()
        mb = massbalance.PastMassBalance(gdir)
        h, w = gdir.get_inversion_flowline_hw()
        mb_ts = mb.get_specific_mb(heights=h, widths=w,
                                   year=np.arange(31)+1970)
        return np.mean(mb_ts)

    def track_mb_1980_sigma(self, gdir):
        self.cfg_init()
        mb = massbalance.PastMassBalance(gdir)
        h, w = gdir.get_inversion_flowline_hw()
        mb_ts = mb.get_specific_mb(heights=h, widths=w,
                                   year=np.arange(31)+1970)
        return np.std(mb_ts)

    def track_mb_1870_avg(self, gdir):
        self.cfg_init()
        mb = massbalance.PastMassBalance(gdir)
        h, w = gdir.get_inversion_flowline_hw()
        mb_ts = mb.get_specific_mb(heights=h, widths=w,
                                   year=np.arange(31)+1860)
        return np.mean(mb_ts)

    def track_mb_1870_sigma(self, gdir):
        self.cfg_init()
        mb = massbalance.PastMassBalance(gdir)
        h, w = gdir.get_inversion_flowline_hw()
        mb_ts = mb.get_specific_mb(heights=h, widths=w,
                                   year=np.arange(31)+1860)
        return np.std(mb_ts)

    def track_inversion_volume(self, gdir):
        self.cfg_init()
        inv_cls = gdir.read_pickle('inversion_output')
        vol = 0
        for cl in inv_cls:
            vol += np.sum(cl['volume'])
        return vol * 1e-9


class full_workflow:

    testdir = os.path.join(get_test_dir(), 'benchmarks', 'track_wf')

    def cfg_init(self):

        # Initialize OGGM and set up the default run parameters
        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_multiprocessing'] = True
        cfg.PARAMS['border'] = 100
        cfg.PARAMS['continue_on_error'] = False

    def setup_cache(self):

        setattr(full_workflow.setup_cache, "timeout", 360)

        utils.mkdir(self.testdir, reset=True)
        self.cfg_init()

        # Pre-download other files which will be needed later
        utils.get_cru_cl_file()
        utils.get_cru_file(var='tmp')
        utils.get_cru_file(var='pre')

        # Get the RGI glaciers for the run.
        rgi_list = ['RGI60-01.10299', 'RGI60-11.00897', 'RGI60-18.02342']
        rgidf = utils.get_rgi_glacier_entities(rgi_list)

        # We use intersects
        db = utils.get_rgi_intersects_entities(rgi_list, version='61')
        cfg.set_intersects_db(db)

        # Sort for more efficient parallel computing
        rgidf = rgidf.sort_values('Area', ascending=False)

        # Go - initialize glacier directories
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
        execute_entity_task(tasks.local_t_star, gdirs)
        execute_entity_task(tasks.mu_star_calibration, gdirs)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.filter_inversion_output, gdirs)

        # Final preparation for the run
        execute_entity_task(tasks.init_present_time_glacier, gdirs)

        # Random climate representative for the tstar climate, without bias
        # In an ideal world this would imply that the glaciers remain stable,
        # but it doesn't have to be so
        execute_entity_task(tasks.run_constant_climate, gdirs,
                            bias=0, nyears=100,
                            output_filesuffix='_tstar')

        execute_entity_task(tasks.run_constant_climate, gdirs,
                            y0=1990, nyears=100,
                            output_filesuffix='_pd')

        # Compile output
        utils.compile_glacier_statistics(gdirs)
        utils.compile_run_output(gdirs, filesuffix='_tstar')
        utils.compile_run_output(gdirs, filesuffix='_pd')
        utils.compile_climate_input(gdirs)

        return gdirs

    def track_start_volume(self, gdirs):
        self.cfg_init()
        path = os.path.join(cfg.PATHS['working_dir'], 'run_output_tstar.nc')
        ds = xr.open_dataset(path)
        return float(ds.volume.sum(dim='rgi_id').isel(time=0)) * 1e-9

    def track_tstar_run_final_volume(self, gdirs):
        self.cfg_init()
        path = os.path.join(cfg.PATHS['working_dir'], 'run_output_tstar.nc')
        ds = xr.open_dataset(path)
        return float(ds.volume.sum(dim='rgi_id').isel(time=-1)) * 1e-9

    def track_1990_run_final_volume(self, gdirs):
        self.cfg_init()
        path = os.path.join(cfg.PATHS['working_dir'], 'run_output_pd.nc')
        ds = xr.open_dataset(path)
        return float(ds.volume.sum(dim='rgi_id').isel(time=-1)) * 1e-9

    def track_avg_temp_full_period(self, gdirs):
        self.cfg_init()
        path = os.path.join(cfg.PATHS['working_dir'], 'climate_input.nc')
        ds = xr.open_dataset(path)
        return float(ds.temp.mean())

    def track_avg_prcp_full_period(self, gdirs):
        self.cfg_init()
        path = os.path.join(cfg.PATHS['working_dir'], 'climate_input.nc')
        ds = xr.open_dataset(path)
        return float(ds.prcp.mean())


class columbia_calving:

    testdir = os.path.join(get_test_dir(), 'benchmarks', 'track_columbia')

    def cfg_init(self):

        # Initialize OGGM and set up the default run parameters
        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
        cfg.PARAMS['border'] = 10

    def setup_cache(self):

        setattr(full_workflow.setup_cache, "timeout", 360)

        utils.mkdir(self.testdir, reset=True)
        self.cfg_init()

        entity = gpd.read_file(get_demo_file('01_rgi60_Columbia.shp')).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

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
        climate.process_dummy_cru_file(gdir, seed=0)

        # Test default k (it overshoots)
        df1 = utils.find_inversion_calving(gdir)

        # Test with smaller k (it doesn't overshoot)
        cfg.PARAMS['k_calving'] = 0.2
        df2 = utils.find_inversion_calving(gdir)

        return (df1.calving_flux.values, df1.mu_star.values,
                df2.calving_flux.values, df2.mu_star.values)

    def track_over_endloop_calving_flux(self, values):
        return values[0][-1]

    def track_over_startloop_mu_star(self, values):
        return values[1][0]

    def track_over_endloop_mu_star(self, values):
        return values[1][-1]

    def track_under_endloop_calving_flux(self, values):
        return values[2][-1]

    def track_under_startloop_mu_star(self, values):
        return values[3][0]

    def track_under_endloop_mu_star(self, values):
        return values[3][-1]
