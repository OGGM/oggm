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

        rho = cfg.PARAMS['ice_density']
        i = 0
        calving_flux = []
        mu_star = []
        ite = []
        cfg.PARAMS['clip_mu_star'] = False
        cfg.PARAMS['min_mu_star'] = 0  # default is now 1
        while i < 12:

            # Calculates a calving flux from model output
            if i == 0:
                # First call we set to zero (not very necessary,
                # this first loop could be removed)
                f_calving = 0
            elif i == 1:
                # Second call we set a very small positive calving
                f_calving = utils.calving_flux_from_depth(gdir, water_depth=1)
            elif cfg.PARAMS['clip_mu_star']:
                # If we have to clip mu the calving becomes the real flux
                fl = gdir.read_pickle('inversion_flowlines')[-1]
                f_calving = fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 / rho
            else:
                # Otherwise it is parameterized
                f_calving = utils.calving_flux_from_depth(gdir)

            # Give it back to the inversion and recompute
            gdir.inversion_calving_rate = f_calving

            # At this step we might raise a MassBalanceCalibrationError
            mu_is_zero = False
            try:
                climate.local_t_star(gdir)
                df = gdir.read_json('local_mustar')
            except MassBalanceCalibrationError as e:
                assert 'mu* out of specified bounds' in str(e)
                # When this happens we clip mu* to zero and store the
                # bad value (just for plotting)
                cfg.PARAMS['clip_mu_star'] = True
                df = gdir.read_json('local_mustar')
                df['mu_star_glacierwide'] = float(str(e).split(':')[-1])
                climate.local_t_star(gdir)

            climate.mu_star_calibration(gdir)
            tasks.prepare_for_inversion(gdir, add_debug_var=True)
            v_inv, _ = tasks.mass_conservation_inversion(gdir)

            # Store the data
            calving_flux = np.append(calving_flux, f_calving)
            mu_star = np.append(mu_star, df['mu_star_glacierwide'])
            ite = np.append(ite, i)

            # Do we have to do another_loop?
            if i > 0:
                avg_one = np.mean(calving_flux[-4:])
                avg_two = np.mean(calving_flux[-5:-1])
                difference = abs(avg_two - avg_one)
                conv = (difference < 0.05 * avg_two or
                        calving_flux[-1] == 0 or
                        calving_flux[-1] == calving_flux[-2])
                if mu_is_zero or conv:
                    break
            i += 1

        assert i < 8
        assert calving_flux[-1] < np.max(calving_flux)
        assert calving_flux[-1] > 2
        assert mu_star[-1] == 0

        mbmod = massbalance.MultipleFlowlineMassBalance
        mb = mbmod(gdir, use_inversion_flowlines=True,
                   mb_model_class=massbalance.ConstantMassBalance,
                   bias=0)
        flux_mb = (mb.get_specific_mb() * gdir.rgi_area_m2) * 1e-9 / rho
        np.testing.assert_allclose(flux_mb, calving_flux[-1], atol=0.001)

        return calving_flux, mu_star

    def track_endloop_calving_flux(self, values):
        return values[0][-1]

    def track_startloop_mu_star(self, values):
        return values[1][0]

    def track_endloop_mu_star(self, values):
        return values[1][-1]
