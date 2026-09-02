# Python imports
import unittest
import numpy as np
import pandas as pd
import os
import shutil
import xarray as xr
import pytest
import oggm
from numpy.testing import assert_allclose
from scipy import optimize as optimization

salem = pytest.importorskip('salem')
gpd = pytest.importorskip('geopandas')

# Locals
import oggm.cfg as cfg
from oggm import tasks, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tests.funcs import get_test_dir
from oggm.utils import get_demo_file
from oggm.core import gis, centerlines
from oggm.core.massbalance import ConstantMassBalance
from oggm.exceptions import InvalidParamsError

pytestmark = pytest.mark.test_env("benchmark")
do_plot = False


class TestSouthGlacier(unittest.TestCase):

    # Test case obtained from ITMIX
    # Data available at:
    # oggm-sample-data/tree/master/benchmarks/south_glacier
    #
    # Citation:
    #
    # Flowers, G.E., N. Roux, S. Pimentel, and C.G. Schoof (2011). Present
    # dynamics and future prognosis of a slowly surging glacier.
    # The Cryosphere, 5, 299-313. DOI: 10.5194/tc-5-299-2011, 2011.

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('dem_SouthGlacier.tif')
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['prcp_fac'] = 2.5
        cfg.PARAMS['baseline_climate'] = 'CRU'

        self.tf = get_demo_file('cru_ts4.01.1901.2016.SouthGlacier.tmp.dat.nc')
        self.pf = get_demo_file('cru_ts4.01.1901.2016.SouthGlacier.pre.dat.nc')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def get_ref_data(self, gdir):

        # Reference data
        df = salem.read_shapefile(get_demo_file('IceThick_SouthGlacier.shp'))
        coords = np.array([p.xy for p in df.geometry]).squeeze()
        df['lon'] = coords[:, 0]
        df['lat'] = coords[:, 1]
        df = df[['lon', 'lat', 'thick']]
        ii, jj = gdir.grid.transform(df['lon'], df['lat'], crs=salem.wgs84,
                                     nearest=True)
        df['i'] = ii
        df['j'] = jj
        df['ij'] = ['{:04d}_{:04d}'.format(i, j) for i, j in zip(ii, jj)]
        df = df.groupby('ij').mean()
        # Averaging converts to floats
        df['i'] = df['i'].astype(np.int64)
        df['j'] = df['j'].astype(np.int64)
        return df

    def test_mb(self):

        # This is a function to produce the MB function needed by Anna

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)

        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_mb_period='2000-01-01_2010-01-01')

        mbref = salem.GeoTiff(get_demo_file('mb_SouthGlacier.tif'))
        demref = salem.GeoTiff(get_demo_file('dem_SouthGlacier.tif'))

        mbref = mbref.get_vardata()
        mbref[mbref == -9999] = np.nan
        demref = demref.get_vardata()[np.isfinite(mbref)]
        mbref = mbref[np.isfinite(mbref)] * 1000

        # compute the bias to make it 0 SMB on the 2D DEM
        rho = cfg.PARAMS['ice_density']
        mbmod = ConstantMassBalance(gdirs[0], bias=0, y0=1995)
        sec_in_year = np.mean([mbmod.sec_in_year(yr)
                               for yr in np.arange(1980, 2011)])
        mymb = mbmod.get_annual_mb(demref) * sec_in_year * rho
        mbmod = ConstantMassBalance(gdirs[0], y0=1995, bias=np.average(mymb))
        mymb = mbmod.get_annual_mb(demref) * sec_in_year * rho
        np.testing.assert_allclose(np.average(mymb), 0., atol=1e-3)

        # Same for ref
        mbref = mbref - np.average(mbref)
        np.testing.assert_allclose(np.average(mbref), 0., atol=1e-3)

        # Fit poly
        p = np.polyfit(demref, mbref, deg=2)
        poly = np.poly1d(p)
        myfit = poly(demref)
        np.testing.assert_allclose(np.average(myfit), 0., atol=1e-3)

        if do_plot:
            import matplotlib.pyplot as plt
            plt.scatter(mbref, demref, s=5,
                        label='Obs (2007-2012), shifted to Avg(SMB) = 0')
            plt.scatter(mymb, demref, s=5, label='OGGM MB')
            plt.scatter(myfit, demref, s=5, label='Polyfit', c='C3')
            plt.xlabel('MB (mm w.e yr-1)')
            plt.ylabel('Altidude (m)')
            plt.legend()
            plt.show()

    def test_inversion_attributes(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_mb_period='2000-01-01_2010-01-01')

        # Tested tasks
        task_list = [
            tasks.gridded_attributes,
            tasks.gridded_mb_attributes,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # Check certain things
        gdir = gdirs[0]
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:

            # The max catchment area should be area of glacier
            assert (ds['catchment_area'].max() ==
                    ds['glacier_mask'].sum() * gdir.grid.dx**2)
            assert (ds['catchment_area_on_catch'].max() ==
                    ds['glacier_mask'].sum() * gdir.grid.dx**2)

            # In the lowest parts of the glaciers the data should be equivalent
            ds_low = ds.isel(y=ds.y < 6741500)
            np.testing.assert_allclose(ds_low['lin_mb_above_z'],
                                       ds_low['lin_mb_above_z_on_catch'])
            np.testing.assert_allclose(ds_low['oggm_mb_above_z'],
                                       ds_low['oggm_mb_above_z_on_catch'])

        # Build some loose tests based on correlation
        df = self.get_ref_data(gdir)
        vns = ['topo',
               'slope',
               'aspect',
               'slope_factor',
               'dis_from_border',
               'catchment_area',
               'catchment_area_on_catch',
               'lin_mb_above_z',
               'lin_mb_above_z_on_catch',
               'oggm_mb_above_z',
               'oggm_mb_above_z_on_catch',
               ]

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            for vn in vns:
                df[vn] = ds[vn].isel(x=('z', df['i']), y=('z', df['j']))

        # Loose tests based on correlations
        cf = df.corr()
        assert cf.loc['slope', 'slope_factor'] < -0.85
        assert cf.loc['slope', 'thick'] < -0.4
        assert cf.loc['dis_from_border', 'thick'] > 0.2
        assert cf.loc['oggm_mb_above_z', 'thick'] > 0.5
        assert cf.loc['lin_mb_above_z', 'thick'] > 0.5
        assert cf.loc['lin_mb_above_z', 'oggm_mb_above_z'] > 0.95

    def test_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_mb_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                            smooth_radius=None,
                            varname_suffix='_alt')
        execute_entity_task(tasks.distribute_thickness_interp, gdirs,
                            varname_suffix='_int')

        # Reference data
        gdir = gdirs[0]
        df = self.get_ref_data(gdir)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:

            v = ds.distributed_thickness_alt
            df['oggm_alt'] = v.isel(x=('z', df['i']), y=('z', df['j']))
            v = ds.distributed_thickness_int
            df['oggm_int'] = v.isel(x=('z', df['i']), y=('z', df['j']))

            ds['ref'] = xr.zeros_like(ds.distributed_thickness_int) * np.nan
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd_int = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        rmsd_alt = ((df.oggm_int - df.thick) ** 2).mean() ** .5
        assert rmsd_int < 85
        assert rmsd_alt < 85

        dfm = df.mean()
        np.testing.assert_allclose(dfm.thick, dfm.oggm_int, 50)
        np.testing.assert_allclose(dfm.thick, dfm.oggm_alt, 50)

        if do_plot:
            import matplotlib.pyplot as plt
            df.plot(kind='scatter', x='oggm_int', y='thick')
            plt.axis('equal')
            df.plot(kind='scatter', x='oggm_alt', y='thick')
            plt.axis('equal')
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            ds.ref.plot(ax=ax1)
            ds.distributed_thickness_int.plot(ax=ax2)
            ds.distributed_thickness_alt.plot(ax=ax3)
            plt.tight_layout()
            plt.show()

    @pytest.mark.slow
    def test_optimize_inversion(self):

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_mb_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Reference data
        gdir = gdirs[0]
        df = self.get_ref_data(gdir)

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)

        glen_a = cfg.PARAMS['inversion_glen_a']
        fs = cfg.PARAMS['inversion_fs']

        def to_optimize(x):
            tasks.mass_conservation_inversion(gdir,
                                              glen_a=glen_a * x[0],
                                              fs=fs * x[1])
            tasks.distribute_thickness_per_altitude(gdir, smooth_radius=None)
            with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
                thick = ds.distributed_thickness.isel(x=('z', df['i']),
                                                      y=('z', df['j']))
                out = (np.abs(thick - df.thick)).mean()
            return out

        opti = optimization.minimize(to_optimize, [1., 1.],
                                     bounds=((0.01, 10), (0.01, 10)),
                                     tol=0.1)
        # Check results and save.
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=glen_a*opti['x'][0],
                            fs=0)
        execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs,
                            smooth_radius=None)

        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            df['oggm'] = ds.distributed_thickness.isel(x=('z', df['i']),
                                                       y=('z', df['j']))
            ds['ref'] = xr.zeros_like(ds.distributed_thickness) * np.nan
            ds['ref'].data[df['j'], df['i']] = df['thick']

        rmsd = ((df.oggm - df.thick) ** 2).mean() ** .5
        assert rmsd < 30

        dfm = df.mean()
        np.testing.assert_allclose(dfm.thick, dfm.oggm, 10)
        if do_plot:
            import matplotlib.pyplot as plt
            df.plot(kind='scatter', x='oggm', y='thick')
            plt.axis('equal')
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ds.ref.plot(ax=ax1)
            ds.distributed_thickness.plot(ax=ax2)
            plt.tight_layout()
            plt.show()

    def test_workflow(self):

        # This is a check that the inversion workflow works fine

        # Download the RGI file for the run
        # Make a new dataframe of those
        rgidf = gpd.read_file(get_demo_file('SouthGlacier.shp'))

        # Go - initialize working directories
        gdirs = workflow.init_glacier_directories(rgidf)

        # Preprocessing tasks
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
            tasks.compute_centerlines,
            tasks.initialize_flowlines,
            tasks.catchment_area,
            tasks.catchment_intersections,
            tasks.catchment_width_geom,
            tasks.catchment_width_correction,
            tasks.compute_downstream_line,
            tasks.compute_downstream_bedshape,
        ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        execute_entity_task(tasks.process_cru_data, gdirs,
                            tmp_file=self.tf,
                            pre_file=self.pf)
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                            ref_mb_period='2000-01-01_2010-01-01')
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs,
                            mb_years=[2000, 2009])

        # Inversion tasks
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        # We use the default parameters for this run
        execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        execute_entity_task(tasks.filter_inversion_output, gdirs)

        df = utils.compile_glacier_statistics(gdirs)
        df['inv_thickness_m'] = df['inv_volume_km3'] / df['rgi_area_km2'] * 1e3
        assert df.inv_thickness_m.iloc[0] < 100

        df = utils.compile_fixed_geometry_mass_balance(gdirs)
        assert len(df) > 100
        df.columns = ['glacier']
        assert np.all(df.std() > 450)

        if do_plot:
            import matplotlib.pyplot as plt
            from oggm.graphics import plot_inversion
            plot_inversion(gdirs)
            plt.show()


@pytest.mark.slow
class TestCoxeGlacier(unittest.TestCase):

    # Test case for a tidewater glacier

    def setUp(self):

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        self.rgi_file = get_demo_file('rgi_RGI50-01.10299.shp')

        # Init
        cfg.initialize()
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('dem_RGI50-01.10299.tif')
        cfg.PARAMS['border'] = 40
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True
        cfg.PARAMS['prcp_fac'] = 2.5
        cfg.PARAMS['baseline_climate'] = 'CRU'
        cfg.PARAMS['evolution_model'] = 'FluxBased'

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_set_width(self):
        entity = gpd.read_file(self.rgi_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # Test that area and area-altitude elev is fine
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo_smoothed'][:]
        rhgt = topo[np.where(mask)][:]

        fls = gdir.read_store('inversion_flowlines')
        hgt, widths = gdir.get_inversion_flowline_hw()

        bs = 100
        bins = np.arange(utils.nicenumber(np.min(hgt), bs, lower=True),
                         utils.nicenumber(np.max(hgt), bs) + 1,
                         bs)
        h1, b = np.histogram(hgt, weights=widths, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        assert utils.rmsd(h1, h2) < 0.02  # less than 2% error
        new_area = np.sum(widths * fls[-1].dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area, gdir.rgi_area_m2)

        centerlines.terminus_width_correction(gdir, new_width=714)

        fls = gdir.read_store('inversion_flowlines')
        hgt, widths = gdir.get_inversion_flowline_hw()

        # Check that the width is ok
        np.testing.assert_allclose(fls[-1].widths[-1] * gdir.grid.dx, 714)

        # Check for area distrib
        bins = np.arange(utils.nicenumber(np.min(hgt), bs, lower=True),
                         utils.nicenumber(np.max(hgt), bs) + 1,
                         bs)
        h1, b = np.histogram(hgt, weights=widths, density=True, bins=bins)
        h2, b = np.histogram(rhgt, density=True, bins=bins)
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)
        assert utils.rmsd(h1, h2) < 0.02  # less than 2% error
        new_area = np.sum(widths * fls[-1].dx * gdir.grid.dx)
        np.testing.assert_allclose(new_area, gdir.rgi_area_m2)

    def test_run(self):

        entity = gpd.read_file(self.rgi_file).iloc[0]
        entity.RGIId = 'RGI60-01.10299'
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # Climate tasks
        tasks.process_dummy_cru_file(gdir, seed=0)
        tasks.mb_calibration_from_geodetic_mb(gdir)
        tasks.apparent_mb_from_any_mb(gdir)

        # Inversion tasks
        tasks.find_inversion_calving_from_any_mb(gdir)

        # Final preparation for the run
        tasks.init_present_time_glacier(gdir)

        # check that calving happens in the real context as well
        tasks.run_constant_climate(gdir, bias=0, y0=1985, nyears=200,
                                   temperature_bias=-0.5)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
            assert ds.calving_m3[-1] > 10


# --- Diagnostics of a preprocessing run (`oggm_prepro_diag`) ---
#
# These do not need a glacier directory: the tool reads the *summary* files a
# preprocessing run leaves behind, so the fixture below writes a (tiny) fake
# one, with the pathologies which have to be handled: a glacier which errored
# during the preprocessing, one whose run stopped early, one which fell back to
# a fixed geometry spinup, and one which starts earlier than the others.

FAKE_REGIONS = {'11': 6, '16': 4}
FAKE_YEARS = np.arange(1980, 2021)
FAKE_RHO = 900.


def _fake_prepro_stats(reg, n):
    """The glacier statistics file of one fake region."""
    ids = [f'RGI60-{reg}.{i + 1:05d}' for i in range(n)]
    df = pd.DataFrame(index=pd.Index(ids, name='rgi_id'))
    df['rgi_region'] = reg
    df['rgi_area_km2'] = np.arange(1, n + 1) * 2.
    df['rgi_year'] = 2000
    df['error_task'] = None
    df['error_msg'] = None
    df['melt_f'] = 5.
    df['prcp_fac'] = 2.
    df['temp_bias'] = 0.5
    df['bias'] = 0.
    df['baseline_climate_source'] = 'FAKE'
    df['reference_period'] = '2000-01-01_2020-01-01'
    df['used_spinup_option'] = 'dynamic melt_f calibration (full success)'
    # object dtype: like the real file, this one has NaNs where it failed
    df['run_dynamic_spinup_success'] = np.array([True] * n, dtype=object)
    df['dynamic_spinup_target_year'] = 2000.
    df['dynamic_spinup_period'] = 20.
    df['area_mismatch_dynamic_spinup_km2_percent'] = 0.5
    df['dmdtda_mismatch_dynamic_calibration'] = 100.
    df['dmdtda_dynamic_calibration_given_error'] = 1000.
    df['dmdtda_dynamic_calibration_error_scaling_factor'] = 0.2
    df['melt_f_before_dynamic_calibration'] = 5.
    df['melt_f_dynamic_calibration'] = 5.

    # The first glacier of each region failed during the preprocessing: it
    # never ran, and is absent from the run output
    df.loc[ids[0], 'error_task'] = 'simple_glacier_masks'
    df.loc[ids[0], 'error_msg'] = 'GeometryError: nominal glacier'
    for c in ['used_spinup_option', 'run_dynamic_spinup_success',
              'dynamic_spinup_period', 'melt_f', 'prcp_fac', 'temp_bias']:
        df.loc[ids[0], c] = np.nan
    # The third one had to fall back to a fixed geometry spinup
    df.loc[ids[2], 'used_spinup_option'] = 'fixed geometry spinup'
    df.loc[ids[2], 'run_dynamic_spinup_success'] = False
    # The fourth one errored *during* the dynamic melt_f calibration, which
    # runs with `ignore_errors`: the error stays on record and its
    # initialisation is never written down, but the fallback did run and its
    # output is complete. It must therefore stay in the population.
    df.loc[ids[3], 'error_task'] = 'run_dynamic_melt_f_calibration_spinup_historical'
    df.loc[ids[3], 'error_msg'] = 'KeyError: not all values found in index'
    for c in ['used_spinup_option', 'run_dynamic_spinup_success',
              'dmdtda_mismatch_dynamic_calibration',
              'dmdtda_dynamic_calibration_given_error',
              'dmdtda_dynamic_calibration_error_scaling_factor',
              'area_mismatch_dynamic_spinup_km2_percent']:
        df.loc[ids[3], c] = np.nan
    # ... and the last one was moved by the dynamic melt_f calibration
    df.loc[ids[-1], 'melt_f_dynamic_calibration'] = 6.
    df.loc[ids[-1], 'melt_f'] = 6.
    return df


def _fake_prepro_run_output(sdf, reg, spinup=True):
    """The compiled run output of one fake region.

    The volume decreases linearly, by 1 km3 per 100 years and per glacier for
    the spinup run and half of that for the fixed geometry one, so that the
    geodetic mass balance of the region can be computed by hand.
    """
    # Only the glacier which failed during the preprocessing is missing from
    # the run output - an error on record does not imply a missing run
    ids = [i for i in sdf.index
           if sdf.loc[i, 'error_task'] != 'simple_glacier_masks']
    nt, ng = len(FAKE_YEARS), len(ids)

    vol = np.zeros((nt, ng))
    for j, rid in enumerate(ids):
        v0 = sdf.loc[rid, 'rgi_area_km2'] * 0.1  # km3
        dvdt = -v0 / 100 * (1 if spinup else 0.5)
        vol[:, j] = (v0 + dvdt * (FAKE_YEARS - 2000)) * 1e9  # m3

    # The second glacier *of the run output* stopped early - no output at the
    # end. In region 11 that is the 6 km2 one: the run output does not contain
    # the glacier which errored during the preprocessing (the 2 km2 one), so
    # the population of region 11 is 4 + 8 + 10 + 12 = 34 km2.
    vol[-3:, 1] = np.nan
    # The last glacier starts earlier than the others: everyone else only has
    # data from 1990 on
    early = ng - 1
    for j in range(ng):
        if j != early:
            vol[FAKE_YEARS < 1990, j] = np.nan

    area = np.repeat(sdf.loc[ids, 'rgi_area_km2'].values[np.newaxis, :] * 1e6,
                     nt, axis=0)
    area[np.isnan(vol)] = np.nan

    err = np.array([''] * ng, dtype='<U20')
    partial = np.zeros(ng)
    partial[1] = 1

    ds = xr.Dataset(
        {'volume': (('time', 'rgi_id'), vol),
         'area': (('time', 'rgi_id'), area),
         # `area_min_h` is what the tool must use - make it clearly different
         'area_min_h': (('time', 'rgi_id'), area * 0.9),
         'mass_kg': (('time', 'rgi_id'), vol * FAKE_RHO),
         'is_fixed_geometry_spinup': (('time', 'rgi_id'), np.zeros((nt, ng))),
         'error_during_run': ('rgi_id', err),
         'is_partial_output': ('rgi_id', partial),
         },
        coords={'time': FAKE_YEARS, 'rgi_id': ids},
    )
    ds.attrs['oggm_version'] = 'fake'
    ds.attrs['creation_date'] = '2026-01-01'
    return ds


@pytest.fixture(scope='module')
def fake_prepro_run(tmp_path_factory):
    """A (tiny) fake `oggm_prepro` output directory."""
    # The diagnostics log at the WORKFLOW level, which cfg defines
    cfg.initialize_minimal()
    base = tmp_path_factory.mktemp('prepro_diag') / 'fake_exp'
    sdir = base / 'RGI62' / 'b_160' / 'L5' / 'summary'
    sdir.mkdir(parents=True)
    for reg, n in FAKE_REGIONS.items():
        sdf = _fake_prepro_stats(reg, n)
        sdf.to_csv(sdir / f'glacier_statistics_{reg}.csv')
        _fake_prepro_run_output(sdf, reg, spinup=True).to_netcdf(
            sdir / f'spinup_historical_run_output_{reg}.nc')
        _fake_prepro_run_output(sdf, reg, spinup=False).to_netcdf(
            sdir / f'historical_run_output_{reg}.nc')
    return base


@pytest.fixture
def fake_geodetic_obs(monkeypatch):
    """No download: a fake Hugonnet et al. dataset over the fake glaciers.

    The observation is exactly the dmdtda of the spinup run (-0.9 m w.e. yr-1,
    see `_fake_prepro_run_output`), so a correct comparison must come out with
    a zero bias for that run.
    """
    obs_dmdtda = -0.9

    def fake_get(file_path=None, rgi_version=None, regional=False):
        if regional:
            df = pd.DataFrame(index=pd.Index([11, 16], name='reg'))
            df['period'] = '2000-01-01_2020-01-01'
            df['dmdt'] = -1.
            df['err_dmdt'] = 0.5
            df['area'] = 1e9
            df['tarea'] = 0.5e9
            df['dmdtda'] = -2.
            df['dmdtda_full_area'] = df['dmdt'] * 1e12 / df['area'] / 1000
            df['err_dmdtda_full_area'] = (df['err_dmdt'] * 1e12 /
                                          df['area'] / 1000)
            return df
        rows = []
        for reg, n in FAKE_REGIONS.items():
            for i in range(n):
                rows.append({'rgiid': f'RGI60-{reg}.{i + 1:05d}',
                             'period': '2000-01-01_2020-01-01',
                             'area': (i + 1) * 2. * 1e6,
                             'dmdtda': obs_dmdtda, 'err_dmdtda': 0.2,
                             'reg': int(reg), 'is_cor': False})
        return pd.DataFrame(rows).set_index('rgiid')

    monkeypatch.setattr(utils, 'get_geodetic_mb_dataframe', fake_get)
    return obs_dmdtda


def test_prepro_diag_read(fake_prepro_run):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    assert prun.level == 5
    assert prun.rgi_version == '62'
    assert prun.border == 160
    assert prun.name == 'fake_exp'
    assert prun.regions == ['11', '16']
    assert prun.runs == ['spinup', 'fixed_geom']
    assert len(prun.stats) == sum(FAKE_REGIONS.values())

    # The summary dir, the level dir and the dir above all work
    for p in [fake_prepro_run / 'RGI62' / 'b_160' / 'L5' / 'summary',
              fake_prepro_run / 'RGI62' / 'b_160' / 'L5',
              fake_prepro_run / 'RGI62' / 'b_160']:
        assert diagnostics.read_prepro_run(p).regions == ['11', '16']

    # One region only
    assert diagnostics.read_prepro_run(fake_prepro_run,
                                       rgi_region='11').regions == ['11']


def test_prepro_diag_completion(fake_prepro_run):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    df = diagnostics.compute_completion(prun)

    # Region 11: 6 glaciers, the first errored, the second stopped early
    assert df.loc['11', 'n_glaciers'] == 6
    # Two glaciers have an error on record (the 2 and the 8 km2 ones), but
    # only the first of them is actually missing from the runs
    assert df.loc['11', 'n_ok_stats'] == 4
    assert df.loc['11', 'n_ok_spinup'] == 4
    assert df.loc['11', 'n_population'] == 4
    # The population follows the runs, so the glacier which errored in the
    # calibration but has a complete fallback run is in it
    ids = diagnostics.read_prepro_run(fake_prepro_run).population('11')[0]
    assert 'RGI60-11.00004' in ids

    # The areas are 2, 4, 6, ... km2, so the failed ones are the small ones
    assert_allclose(df.loc['11', 'rgi_area_km2'], 42)
    assert_allclose(df.loc['11', 'area_ok_stats_km2'], 32)
    assert_allclose(df.loc['11', 'area_population_km2'], 34)
    # ... which makes `ok_stats` the *lower* of the two, as in the real runs
    assert (df.loc['11', 'perc_area_ok_stats'] <
            df.loc['11', 'perc_area_ok_spinup'])
    assert_allclose(df.loc['11', 'perc_area_population'], 34 / 42 * 100)
    # Both percentages are of the same total, the RGI area of the region
    assert_allclose(df.loc['11', 'perc_area_ok_stats'], 32 / 42 * 100)

    # The global row is the sum
    assert df.loc['global', 'n_glaciers'] == 10
    assert (df.loc['global', 'n_population'] ==
            df.loc['11', 'n_population'] + df.loc['16', 'n_population'])

    # The failed glaciers are accounted for, with their task and their area
    errs = diagnostics.compute_errors(prun)
    sel = errs.loc[(errs['region'] == 'global') &
                   (errs['source'] == 'statistics')].set_index('error')
    assert sel.loc['simple_glacier_masks', 'n'] == 2  # one per region
    assert_allclose(sel.loc['simple_glacier_masks', 'area_km2'], 4)
    # The calibration error is on record even though those glaciers ran
    assert sel.loc['run_dynamic_melt_f_calibration_spinup_historical',
                   'n'] == 2
    sel = errs.loc[(errs['region'] == '11') & (errs['source'] == 'run_spinup')]
    assert sel['n'].iloc[0] == 1  # the one which stopped early


def test_prepro_diag_spinup(fake_prepro_run):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    df = diagnostics.compute_spinup(prun)

    # One glacier of each region fell back to a fixed geometry spinup: the
    # third one, i.e. 6 km2 out of 42 in region 11
    assert df.loc['11', 'n_fixed_geometry_spinup'] == 1
    assert_allclose(df.loc['11', 'perc_area_fixed_geometry_spinup'],
                    6 / 42 * 100)
    # Two glaciers have no initialisation on record: the one which never ran
    # and the one whose calibration errored - but the second one does have a
    # complete run, and saying it had "no spinup" would overstate the failure
    assert df.loc['11', 'n_not_recorded'] == 2
    assert df.loc['11', 'n_not_recorded_but_ran'] == 1
    assert_allclose(df.loc['11', 'area_not_recorded_km2'], 10)
    assert_allclose(df.loc['11', 'area_not_recorded_but_ran_km2'], 8)

    # The mismatch of the fake run is 100 kg m-2 yr-1 for a tolerance of
    # 1000 * 0.2, so everything which was calibrated is inside it
    assert df.loc['11', 'n_dmdtda_match'] == df.loc['11', 'n_dmdtda_calibrated']

    # The last glacier starts in 1980, all the others in 1990
    assert df.loc['11', 'first_output_yr_min'] == 1980
    assert df.loc['11', 'first_output_yr_max'] == 1990
    assert df.loc['11', 'n_before_common_start'] == 1
    assert df.loc['global', 'first_output_yr_min'] == 1980


def test_prepro_diag_timeseries(fake_prepro_run):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    ts = diagnostics.compute_timeseries(prun)

    sel = ts.loc[(ts['region'] == '11') & (ts['run'] == 'spinup')]
    sel = sel.set_index('year')

    # The years before 1990 are in the table, but only one glacier deep, and
    # they are not part of the common period
    assert not sel.loc[1985, 'is_common_period']
    assert sel.loc[1985, 'n_glaciers'] == 1
    assert sel.loc[1995, 'is_common_period']
    assert sel.loc[1995, 'n_glaciers'] == 4
    assert diagnostics.common_period_start(ts, '11', 'spinup') == 1990

    # `area_min_h` is what is used, and it is not `area`
    assert_allclose(sel.loc[1995, 'area_min_h_km2'],
                    sel.loc[1995, 'area_km2'] * 0.9)
    # The population of region 11 is 4 glaciers of 4, 8, 10 and 12 km2 (the
    # 2 km2 one errored, the 6 km2 one stopped early)
    assert_allclose(sel.loc[1995, 'area_km2'], 34)

    # The global row is the sum of the regions
    glob = ts.loc[(ts['region'] == 'global') &
                  (ts['run'] == 'spinup')].set_index('year')
    reg = [ts.loc[(ts['region'] == r) & (ts['run'] == 'spinup')]
           .set_index('year')['volume_km3'] for r in prun.regions]
    assert_allclose(glob['volume_km3'], sum(reg))


def test_prepro_diag_geodetic(fake_prepro_run, fake_geodetic_obs):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    df = diagnostics.compute_geodetic(prun)

    # By construction the fake glaciers lose 1% of their initial volume per
    # year in the spinup run, i.e. dmdtda = -0.1 km3 km-2 * 900 / 100 yr,
    # which is -0.9 m w.e. yr-1 - and that is what the fake observation says
    assert_allclose(df.loc['11', 'spinup_dmdtda'], -0.9, atol=1e-10)
    assert_allclose(df.loc['11', 'fixed_geom_dmdtda'], -0.45, atol=1e-10)
    assert_allclose(df.loc['11', 'hug_pergla_dmdtda'], fake_geodetic_obs)
    assert_allclose(df.loc['11', 'bias_spinup_vs_hug_pergla_dmdtda'], 0,
                    atol=1e-10)

    # dmdt is the total, in Gt yr-1: 36 km2 * -0.9 m w.e. yr-1
    assert_allclose(df.loc['11', 'spinup_dmdt_Gt'], 34 * -0.9 * 1e-3,
                    rtol=1e-6)

    # The global row is *not* the mean of the regional specific rates: it is
    # recomputed from the totals (here they are all the same, so it is -0.9)
    assert_allclose(df.loc['global', 'spinup_dmdtda'], -0.9, atol=1e-10)
    assert_allclose(df.loc['global', 'spinup_dmdt_Gt'],
                    df.loc['11', 'spinup_dmdt_Gt'] +
                    df.loc['16', 'spinup_dmdt_Gt'])

    # The published regional dmdtda is on the measured area and must not be
    # confused with the one we compare to
    assert_allclose(df.loc['11', 'hug_reg_dmdtda_published'], -2.)
    assert_allclose(df.loc['11', 'hug_reg_dmdtda'], -1.)


def test_prepro_diag_area_match(fake_prepro_run):

    from oggm import diagnostics

    prun = diagnostics.read_prepro_run(fake_prepro_run)
    df = diagnostics.compute_area_match(prun)

    # The RGI year of the fake glaciers is 2000, where the modelled
    # `area_min_h` is 0.9 * the RGI area of the population
    assert_allclose(df.loc['11', 'rgi_year_wmedian'], 2000)
    assert_allclose(df.loc['11', 'rgi_area_km2'], 42)
    assert_allclose(df.loc['11', 'rgi_area_population_km2'], 34)
    assert_allclose(df.loc['11', 'area_min_h_at_rgi_yr_spinup_km2'], 34 * 0.9)
    # Against the glaciers which ran - the number which is about the model
    assert_allclose(df.loc['11', 'area_mismatch_pop_spinup_percent'], -10)
    # Against the whole region - the missing glaciers are in there too
    assert_allclose(df.loc['11', 'area_mismatch_spinup_percent'],
                    (34 * 0.9 - 42) / 42 * 100)


@pytest.mark.slow
def test_prepro_diag_cli(fake_prepro_run, fake_geodetic_obs, tmp_path):

    pytest.importorskip('matplotlib')
    from oggm.cli.prepro_diag import run_prepro_diag, parse_args

    out_dir = tmp_path / 'diag'
    run_prepro_diag(input_dir=str(fake_prepro_run), output_dir=str(out_dir))

    report = (out_dir / 'report.txt').read_text()
    assert 'OGGM preprocessing diagnostics - fake_exp' in report
    for section in ['Completion', 'Dynamic spinup', 'Geodetic mass balance',
                    'Modelled area at the RGI date', 'Volume and area',
                    'Calibrated mass balance parameters']:
        assert section in report
    # The area convention is stated, and it is the right one
    assert 'area_min_h' in report

    for table in ['completion', 'spinup', 'geodetic', 'timeseries',
                  'mb_params', 'area_match', 'errors']:
        assert (out_dir / 'tables' / f'{table}.csv').exists()

    for plot in ['dmdtda_by_region', 'mass_loss_by_region',
                 'volume_by_region', 'area_min_h_by_region',
                 'volume_norm_all_regions', 'completion_by_region',
                 'mb_params_hist']:
        assert (out_dir / 'plots' / f'{plot}.png').exists()
    assert (out_dir / 'plots' / 'per_region' / 'RGI11.png').exists()

    # The command line arguments end up where they should
    kwargs = parse_args(['--input', 'in_dir', '--output-dir', 'out_dir',
                         '--rgi-region', '11', '--no-plots'])
    assert kwargs['input_dir'] == 'in_dir'
    assert kwargs['output_dir'] == 'out_dir'
    assert kwargs['rgi_region'] == ['11']
    assert not kwargs['make_plots']
    with pytest.raises(InvalidParamsError):
        parse_args([])
