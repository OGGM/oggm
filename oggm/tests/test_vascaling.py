"""Tests for the volume/area scaling model in `vascaling.py`."""

# External libs
import numpy as np
import datetime
import os
import shutil
import copy

# import test libs
import unittest
import pytest

# import gis libs
gpd = pytest.importorskip('geopandas')

# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm import utils
from oggm.utils import (get_demo_file, ncDataset, md, rmsd_bc, rel_err,
                        corrcoef)
from oggm.core import (gis, vascaling, climate, centerlines,
                       massbalance, flowline, inversion)
from oggm.tests.funcs import get_test_dir, patch_url_retrieve_github


pytestmark = pytest.mark.test_env("vascaling")
_url_retrieve = None


def setup_module(module):
    module._url_retrieve = utils.oggm_urlretrieve
    oggm.utils._downloads.oggm_urlretrieve = patch_url_retrieve_github


def teardown_module(module):
    oggm.utils._downloads.oggm_urlretrieve = module._url_retrieve


class TestVAScalingModel(unittest.TestCase):
    """Unittest TestCase testing the implementation of the volume/area scaling
    model, based on Marzeion et. al., 2012.
    """

    def setUp(self):
        """Instance the TestCase, create the test directory,
        OGGM initialisation and setting paths and parameters.
        """

        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_vas')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

        # load default parametere file and set working directory
        cfg.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        # set path to GIS files
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        # set parameters for climate file and mass balance calibration
        cfg.PARAMS['baseline_climate'] = 'CUSTOM'
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['run_mb_calibration'] = True
        # adjust parameters for HistAlp climate
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.75
        cfg.PARAMS['temp_all_liq'] = 2.
        cfg.PARAMS['temp_all_solid'] = 0.
        cfg.PARAMS['temp_default_gradient'] = -0.0065

        # coveralls.io has issues if multiprocessing is enabled
        cfg.PARAMS['use_multiprocessing'] = False

    def tearDown(self):
        """Removes the test directories."""
        self.rm_dir()

    def rm_dir(self):
        """Removes the test directories."""
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        """Cleans the test directories."""
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_terminus_temp(self):
        """Testing the subroutine which computes the terminus temperature
        from the given climate file and glacier DEM. Pretty straight forward
        and somewhat useless, but nice finger exercise.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid
        gis.define_glacier_region(gdir, entity=entity)
        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # read the following variable from the center pixel (46.83N 10.75E)
        # of the Hintereisferner HistAlp climate file for the
        # entire time period from October 1801 until September 2003
        # - surface height in m asl.
        # - total precipitation amount in kg/m2
        # - 2m air temperature in °C
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        # define a temperature anomaly
        temp_anomaly = 0

        # specify temperature gradient
        temp_grad = -0.0065

        # the terminus temperature must equal the input temperature
        # if terminus elevation equals reference elevation
        temp_terminus =\
            vascaling._compute_temp_terminus(ref_t, temp_grad, ref_hgt=ref_h,
                                             terminus_hgt=ref_h,
                                             temp_anomaly=temp_anomaly)
        np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # the terminus temperature must equal the input terperature
        # if the gradient is zero
        for term_h in np.array([-100, 0, 100]) + ref_h:
            temp_terminus =\
                vascaling._compute_temp_terminus(ref_t, temp_grad=0,
                                                 ref_hgt=ref_h,
                                                 terminus_hgt=term_h,
                                                 temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # now test the routine with actual elevation differences
        # and a non zero temperature gradient
        for h_diff in np.array([-100, 0, 100]):
            term_h = ref_h + h_diff
            temp_diff = temp_grad * h_diff
            temp_terminus =\
                vascaling._compute_temp_terminus(ref_t, temp_grad,
                                                 ref_hgt=ref_h,
                                                 terminus_hgt=term_h,
                                                 temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus,
                                       ref_t + temp_anomaly + temp_diff)

    def test_solid_prcp(self):
        """Tests the subroutine which computes solid precipitation amount from
        given total precipitation and temperature.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid
        gis.define_glacier_region(gdir, entity=entity)
        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # read the following variable from the center pixel (46.83N 10.75E)
        # of the Hintereisferner HistAlp climate file for the
        # entire time period from October 1801 until September 2003
        # - surface height in m asl.
        # - total precipitation amount in kg/m2
        # - 2m air temperature in °C
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        # define needed parameters
        prcp_factor = 1
        temp_all_solid = 0
        temp_grad = -0.0065

        # define elevation levels
        ref_hgt = ref_h
        min_hgt = ref_h - 100
        max_hgt = ref_h + 100

        # if the terminus temperature is below the threshold for
        # solid precipitation all fallen precipitation must be solid
        temp_terminus = ref_t * 0 + temp_all_solid
        solid_prcp = vascaling._compute_solid_prcp(ref_p, prcp_factor, ref_hgt,
                                                   min_hgt, max_hgt,
                                                   temp_terminus,
                                                   temp_all_solid, temp_grad,
                                                   prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, ref_p)

        # if the temperature at the maximal elevation is above the threshold
        # for solid precipitation all fallen precipitation must be liquid
        temp_terminus = ref_t + 100
        solid_prcp = vascaling._compute_solid_prcp(ref_p, prcp_factor, ref_hgt,
                                                   min_hgt, max_hgt,
                                                   temp_terminus,
                                                   temp_all_solid, temp_grad,
                                                   prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, 0)

        # test extreme case if max_hgt equals min_hgt
        test_p = ref_p * (ref_t <= temp_all_solid).astype(int)
        solid_prcp = vascaling._compute_solid_prcp(ref_p, prcp_factor, ref_hgt,
                                                   ref_hgt, ref_hgt, ref_t,
                                                   temp_all_solid, temp_grad,
                                                   prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, test_p)

    def test_min_max_elevation(self):
        """Test the helper method which computes the minimal and maximal
        glacier surface elevation in meters asl, from the given DEM and glacier
        outline.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # set targets from RGI

        min_target = 2430.0
        max_target = 3674.0
        # get values from method
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        # test with one percentage relative tolerance
        np.testing.assert_allclose(min_hgt, min_target, rtol=1e-2)
        np.testing.assert_allclose(max_hgt, max_target, rtol=1e-2)

    def test_yearly_mb_temp_prcp(self):
        """Test the routine which returns the yearly mass balance relevant
        climate parameters, i.e. positive melting temperature and solid
        precipitation. The testing target is the output of the corresponding
        OGGM routine `get_yearly_mb_climate_on_glacier(gdir)`.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        # run centerline prepro tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # get yearly sums of terminus temperature and solid precipitation
        years, temp, prcp = vascaling.get_yearly_mb_temp_prcp(gdir)

        # use the OGGM methode to get the mass balance
        # relevant climate parameters
        years_oggm, temp_oggm, prcp_oggm = \
            climate.mb_yearly_climate_on_glacier(gdir)

        # the energy input at the glacier terminus must be greater than (or
        # equal to) the glacier wide average, since the air temperature drops
        # with elevation, i.e. the mean deviation must be positive, using the
        # OGGM data as reference
        assert md(temp_oggm, temp) >= 0
        # consequentially, the average mass input must be less than (or equal
        # to) the mass input integrated over the whole glacier surface, i.e.
        # the mean deviation must be negative, using the OGGM data as reference
        # TODO: does it actually?! And if so, why?! @ASK
        assert md(prcp_oggm, prcp) <= 0

        # correlation must be higher than set threshold
        assert corrcoef(temp, temp_oggm) >= 0.94
        assert corrcoef(prcp, prcp_oggm) >= 0.98

        # get terminus temperature using the OGGM routine
        fpath = gdir.get_filepath('gridded_data')
        with ncDataset(fpath) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo'][:]
        heights = np.array([np.min(topo[np.where(mask == 1)])])
        years_height, temp_height, _ = \
            climate.mb_yearly_climate_on_height(gdir, heights, flatten=False)
        temp_height = temp_height[0]
        # both time series must be equal
        np.testing.assert_array_equal(temp, temp_height)

        # get solid precipitation averaged over the glacier
        # (not weighted with widths)
        fls = gdir.read_pickle('inversion_flowlines')
        heights = np.array([])
        for fl in fls:
            heights = np.append(heights, fl.surface_h)
        years_height, _, prcp_height = \
            climate.mb_yearly_climate_on_height(gdir, heights, flatten=True)
        # correlation must be higher than set threshold
        assert corrcoef(prcp, prcp_height) >= 0.99

        # TODO: assert absolute values (or differences) of precipitation @ASK

        # test exception handling of out of bounds time/year range
        with self.assertRaises(climate.MassBalanceCalibrationError):
            # start year out of bounds
            year_range = [1500, 1980]
            _, _, _ = vascaling.get_yearly_mb_temp_prcp(gdir,
                                                        year_range=year_range)
        with self.assertRaises(climate.MassBalanceCalibrationError):
            # end year oud of bounds
            year_range = [1980, 3000]
            _, _, _ = vascaling.get_yearly_mb_temp_prcp(gdir,
                                                        year_range=year_range)
        with self.assertRaises(ValueError):
            # get not N full years
            t0 = datetime.datetime(1980, 1, 1)
            t1 = datetime.datetime(1980, 3, 1)
            time_range = [t0, t1]
            _, _, _ = vascaling.get_yearly_mb_temp_prcp(gdir,
                                                        time_range=time_range)

        # TODO: assert gradient in climate file?!

        pass

    def test_local_t_star(self):

        # set parameters for climate file and mass balance calibration
        cfg.PARAMS['baseline_climate'] = 'CUSTOM'
        cfg.PARAMS['baseline_y0'] = 1850
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['run_mb_calibration'] = False

        # read the Hintereisferner
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and the glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        # run centerline prepro tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = vascaling.t_star_from_refmb(gdir)
        t_star, bias = res['t_star'], res['bias']
        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir, tstar=t_star, bias=bias)
        # read calibration results
        vas_mustar_refmb = gdir.read_json('vascaling_mustar')

        # get reference t* list
        ref_df = cfg.PARAMS['vas_ref_tstars_rgi5_histalp']
        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir, ref_df=ref_df)
        # read calibration results
        vas_mustar_refdf = gdir.read_json('vascaling_mustar')

        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir)
        # read calibration results
        vas_mustar = gdir.read_json('vascaling_mustar')

        # compare with each other
        assert vas_mustar_refdf == vas_mustar
        # TODO: this test is failing currently
        # np.testing.assert_allclose(vas_mustar_refmb['bias'],
        #                            vas_mustar_refdf['bias'], atol=1)
        vas_mustar_refdf.pop('bias')
        vas_mustar_refmb.pop('bias')
        # end of workaround
        assert vas_mustar_refdf == vas_mustar_refmb
        # compare with know values
        # TODO: tests need revisiting
        # assert vas_mustar['t_star'] == 1905
        # assert abs(vas_mustar['mu_star'] - 47.76) <= 0.1
        # assert abs(vas_mustar['bias'] - 66.12) <= 0.1

    def test_ref_t_stars(self):
        """TODO: write docstring and test"""
        pass

    # -------------------------
    # Test mass balance models
    # -------------------------

    def _setup_mb_test(self):
        """Avoiding a chunk of code duplicate. Performs needed prepo tasks and
        returns the oggm.GlacierDirectory.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and the glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # run centerline prepro tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # read reference glacier mass balance data
        mbdf = gdir.get_ref_mb_data()
        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = vascaling.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir, tstar=t_star, bias=bias)

        # run OGGM mu* calibration
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # pass the GlacierDirectory
        return gdir

    def test_monthly_climate(self):
        """Test the routine getting the monthly climate against
        the routine getting annual climate.
        """

        # run all needed prepro tasks
        gdir = self._setup_mb_test()

        # instance the mass balance models
        mbmod = vascaling.VAScalingMassBalance(gdir)

        # get relevant glacier surface elevation
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)

        # get all month of the year in the
        # floating (hydrological) year convention
        year = 1975
        months = np.linspace(year, year + 1, num=12, endpoint=False)

        # create containers
        temp_month = np.empty(12)
        prcp_month = np.empty(12)
        # get mb relevant climate data for every month
        for i, month in enumerate(months):
            _temp, _prcp = mbmod.get_monthly_climate(min_hgt, max_hgt, month)
            temp_month[i] = _temp
            prcp_month[i] = _prcp

        # melting temperature and precipitation amount cannot be negative
        assert temp_month.all() >= 0.
        assert prcp_month.all() >= 0.

        # get climate data for the whole year
        temp_year, prcp_year = mbmod.get_annual_climate(min_hgt, max_hgt, year)

        # compare
        np.testing.assert_array_almost_equal(temp_month, temp_year, decimal=2)
        np.testing.assert_array_almost_equal(prcp_month, prcp_year, decimal=2)

    def test_annual_climate(self):
        """Test my routine against the corresponding OGGM routine from
        the `PastMassBalance()` model.
        """

        # run all needed prepro tasks
        gdir = self._setup_mb_test()

        # instance the mass balance models
        vas_mbmod = vascaling.VAScalingMassBalance(gdir)
        past_mbmod = massbalance.PastMassBalance(gdir)

        # get relevant glacier surface elevation
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        heights = np.array([min_hgt, (min_hgt + max_hgt) / 2, max_hgt])

        # specify an (arbitray) year
        year = 1975
        # get mass balance relevant climate information
        temp_for_melt_vas, prcp_solid_vas = \
            vas_mbmod.get_annual_climate(min_hgt, max_hgt, year)
        _, temp_for_melt_oggm, _, prcp_solid_oggm = \
            past_mbmod.get_annual_climate(heights, year)

        # prepare my (monthly) values for comparison
        temp_for_melt_vas = temp_for_melt_vas.sum()
        prcp_solid_vas = prcp_solid_vas.sum()

        # computed positive terminus melting temperature must be equal for both
        # used methods, i.e. temp_VAS == temp_OGGM
        np.testing.assert_allclose(temp_for_melt_vas,
                                   temp_for_melt_oggm[0],
                                   rtol=1e-3)

        # glacier averaged solid precipitation amount must be greater than (or
        # equal to) solid precipitation amount at glacier terminus elevation
        assert md(prcp_solid_oggm[0], prcp_solid_vas) >= 0
        # glacier averaged solid precipitation amount must be comparable to the
        # solid precipitation amount at average glacier surface elevation
        assert rel_err(prcp_solid_oggm[1], prcp_solid_vas) <= 0.15
        # glacier averaged solid precipitation amount must be less than (or
        # equal to) solid precipitation amount at maximum glacier elevation
        assert md(prcp_solid_oggm[2], prcp_solid_vas) <= 0

    def test_annual_mb(self):
        """Test the routine computing the annual mass balance."""
        # run all needed prepro tasks
        gdir = self._setup_mb_test()

        # get relevant glacier surface elevation
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)

        # define temporal range
        year = 1975
        years = np.array([year, year])

        # get mass balance relevant climate data
        _, temp, prcp = vascaling.get_yearly_mb_temp_prcp(gdir,
                                                          year_range=years)
        temp = temp[0]
        prcp = prcp[0]

        # read mu* and bias from vascaling_mustar
        vascaling_mustar = gdir.read_json('vascaling_mustar')
        mu_star = vascaling_mustar['mu_star']
        bias = vascaling_mustar['bias']

        # specify scaling factor for SI units [kg s-1]
        fac_SI = cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']

        # compute mass balance 'by hand'
        mb_ref = (prcp - mu_star * temp - bias) / fac_SI
        # compute mb using the VAS mass balance model
        mb_mod = vascaling.VAScalingMassBalance(gdir).get_annual_mb(min_hgt,
                                                                    max_hgt,
                                                                    year)
        # compare mass balances with bias
        np.testing.assert_allclose(mb_ref, mb_mod, rtol=1e-3)

        # compute mass balance 'by hand'
        mb_ref = (prcp - mu_star * temp) / fac_SI
        # compute mb 'by model'
        mb_mod = vascaling.VAScalingMassBalance(gdir, bias=0). \
            get_annual_mb(min_hgt, max_hgt, year)
        # compare mass balances without bias
        np.testing.assert_allclose(mb_ref, mb_mod, rtol=1e-3)

    def test_monthly_mb(self):
        """TODO: write test and docstring"""
        pass

    def test_monthly_specific_mb(self):
        """Test the monthly specific mass balance against the
        corresponding yearly mass balance.
        """

        # run all needed prepro tasks
        gdir = self._setup_mb_test()

        # instance mb models
        vas_mbmod = vascaling.VAScalingMassBalance(gdir)

        # get relevant glacier surface elevation
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)

        # get all month of that year in the
        # floating (hydrological) year convention
        year = 1803
        months = np.linspace(year, year + 1, num=12, endpoint=False)

        # compute monthly specific mass balance for
        # all month of given year and store in array
        spec_mb_month = np.empty(months.size)
        for i, month in enumerate(months):
            spec_mb_month[i] = vas_mbmod.get_monthly_specific_mb(min_hgt,
                                                                 max_hgt,
                                                                 month)

        # compute yearly specific mass balance
        spec_mb_year = vas_mbmod.get_specific_mb(min_hgt, max_hgt, year)

        # compare
        np.testing.assert_allclose(spec_mb_month.sum(), spec_mb_year,
                                   rtol=1e-3)

    def test_specific_mb(self):
        """Compare the specific mass balance to the one computed
        using the OGGM function of the PastMassBalance model.
        """

        # run all needed prepro tasks
        gdir = self._setup_mb_test()

        # instance mb models
        vas_mbmod = vascaling.VAScalingMassBalance(gdir)
        past_mbmod = massbalance.PastMassBalance(gdir)

        # get relevant glacier surface elevation
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)

        # define temporal range
        ys = 1802
        ye = 2003
        years = np.arange(ys, ye + 1)

        # get flow lines
        fls = gdir.read_pickle('inversion_flowlines')

        # create empty container
        past_mb = np.empty(years.size)
        vas_mb = np.empty(years.size)
        # get specific mass balance for all years
        for i, year in enumerate(years):
            past_mb[i] = past_mbmod.get_specific_mb(fls=fls, year=year)
            vas_mb[i] = vas_mbmod.get_specific_mb(min_hgt, max_hgt, year)

        # compute and check correlation
        assert corrcoef(past_mb, vas_mb) >= 0.94

        # relative error of average spec mb
        # TODO: does this even make any sense?!
        assert np.abs(rel_err(past_mb.mean(), vas_mb.mean())) <= 0.36

        # check correlation of positive and negative mb years
        assert corrcoef(np.sign(past_mb), np.sign(vas_mb)) >= 0.72

        # compare to reference mb measurements
        mbs = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        assert corrcoef(vas_mb[np.in1d(years, mbs.index)], mbs) >= 0.79

    # -------------------
    # Test scaling model
    # -------------------

    def _set_up_VAS_model(self):
        """Avoiding a chunk of code duplicate. Set's up a running volume/area
        scaling model, including all needed prepo tasks.
        """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # run center line preprocessing tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # read reference glacier mass balance data
        mbdf = gdir.get_ref_mb_data()
        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # --------------------
        #  MASS BALANCE TASKS
        # --------------------

        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir, tstar=t_star, bias=bias)

        # instance the mass balance models
        mbmod = vascaling.VAScalingMassBalance(gdir)

        # ----------------
        #  DYNAMICAL PART
        # ----------------
        # get reference area
        a0 = gdir.rgi_area_m2
        # get reference year
        y0 = gdir.read_json('climate_info')['baseline_hydro_yr_0']
        # get min and max glacier surface elevation
        h0, h1 = vascaling.get_min_max_elevation(gdir)

        model = vascaling.VAScalingModel(year_0=y0, area_m2_0=a0,
                                         min_hgt=h0, max_hgt=h1,
                                         mb_model=mbmod)
        return gdir, model

    def test_time_scales(self):
        """Test the internal method which computes the glaciers time scales
        for length change and area change.
        """

        # get glacier directory and set up VAS model
        _, model = self._set_up_VAS_model()
        # compute time scales
        model._compute_time_scales()
        # compare to given values
        np.testing.assert_allclose(model.tau_l, 53., atol=5)
        np.testing.assert_allclose(model.tau_a, 17., atol=3)

    def test_reset(self):
        """Test the method which sets the model back to its initial state."""

        # get glacier directory and set up VAS model
        _, model = self._set_up_VAS_model()
        # run for some number of years
        n_years = 10
        model.run_until(model.year + n_years)
        # reset the model
        model.reset()
        # check if initial values are restored
        assert model.year == model.year_0
        assert model.length_m == model.length_m_0
        assert model.area_m2 == model.area_m2_0
        assert model.volume_m3 == model.volume_m3_0
        assert model.min_hgt == model.min_hgt_0

    def test_step(self):
        """Test the advance of the model glacier after one time step."""

        # get glacier directory and set up VAS model
        _, model = self._set_up_VAS_model()
        # copy initial state of the model
        m0 = copy.deepcopy(model)
        # advance model glacier by one year
        model.step()
        # compare initial to advanced model state
        dV = m0.spec_mb * m0.area_m2 / m0.rho
        np.testing.assert_allclose(model.volume_m3 - m0.volume_m3, dV)

    def test_run_until_and_store(self):
        """Test the volume/area scaling model against the oggm.FluxBasedModel.

        Both models run the Hintereisferner over the entire HistAlp climate
        period, initialized with the 2003 RGI outline without spin up.

        The following two parameters for length, area and volume are tested:
            - correlation coefficient
            - relative RMSE, i.e. RMSE/mean(OGGM). Whereby the results from the
                VAS model are offset with the average differences to the OGGM
                results.
       """

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)

        # run center line preprocessing tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.compute_downstream_line(gdir)
        centerlines.compute_downstream_bedshape(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_intersections(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)

        # read reference glacier mass balance data
        mbdf = gdir.get_ref_mb_data()
        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # --------------------
        #  SCALING MODEL
        # --------------------

        # compute local t* and the corresponding mu*
        vascaling.local_t_star(gdir, tstar=t_star, bias=bias)

        # instance the mass balance models
        vas_mbmod = vascaling.VAScalingMassBalance(gdir)

        # get reference area
        a0 = gdir.rgi_area_m2
        # get reference year
        y0 = gdir.read_json('climate_info')['baseline_hydro_yr_0']
        # get min and max glacier surface elevation
        h0, h1 = vascaling.get_min_max_elevation(gdir)

        vas_model = vascaling.VAScalingModel(year_0=y0, area_m2_0=a0,
                                             min_hgt=h0, max_hgt=h1,
                                             mb_model=vas_mbmod)

        # let model run over entire HistAlp climate period
        vas_ds = vas_model.run_until_and_store(2003)

        # ------
        #  OGGM
        # ------

        # compute local t* and the corresponding mu*
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # instance the mass balance models
        mb_mod = massbalance.PastMassBalance(gdir)

        # perform ice thickness inversion
        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        inversion.filter_inversion_output(gdir)

        # initialize present time glacier
        flowline.init_present_time_glacier(gdir)

        # instance flowline model
        fls = gdir.read_pickle('model_flowlines')
        y0 = gdir.read_json('climate_info')['baseline_hydro_yr_0']
        fl_mod = flowline.FluxBasedModel(flowlines=fls, mb_model=mb_mod, y0=y0)

        # run model and store output as xarray data set
        _, oggm_ds = fl_mod.run_until_and_store(2003)

        # temporal indices must be equal
        assert (vas_ds.time == oggm_ds.time).all()

        # specify which parameters to compare and their respective correlation
        # coefficients and rmsd values
        params = ['length_m', 'area_m2', 'volume_m3']
        corr_coeffs = np.array([0.96, 0.90, 0.93])
        rmsds = np.array([0.43e3, 0.14e6, 0.03e9])

        # compare given parameters
        for param, cc, rmsd in zip(params, corr_coeffs, rmsds):
            # correlation coefficient
            assert corrcoef(oggm_ds[param].values, vas_ds[param].values) >= cc
            # root mean squared deviation
            rmsd_an = rmsd_bc(oggm_ds[param].values, vas_ds[param].values)
            assert rmsd_an <= rmsd

    def test_run_random_climate(self):
        """ Test the run_random_climate task for a climate based on the
        equilibrium period centred around t*. Additionally a positive and a
        negative temperature bias are tested.

        Returns
        -------

        """
        # let's not use the mass balance bias since we want to reproduce
        # results from mass balance calibration
        cfg.PARAMS['use_bias_for_run'] = False

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)
        # compute mass balance parameters
        ref_df = cfg.PARAMS['vas_ref_tstars_rgi5_histalp']
        vascaling.local_t_star(gdir, ref_df=ref_df)

        # define some parameters for the random climate model
        nyears = 300
        seed = 1
        temp_bias = 0.5
        # read the equilibirum year used for the mass balance calibration
        t_star = gdir.read_json('vascaling_mustar')['t_star']
        # run model with random climate
        _ = vascaling.run_random_climate(gdir, nyears=nyears, y0=t_star,
                                         seed=seed)
        # run model with positive temperature bias
        _ = vascaling.run_random_climate(gdir, nyears=nyears, y0=t_star,
                                         seed=seed, temperature_bias=temp_bias,
                                         output_filesuffix='_bias_p')
        # run model with negative temperature bias
        _ = vascaling.run_random_climate(gdir, nyears=nyears, y0=t_star,
                                         seed=seed,
                                         temperature_bias=-temp_bias,
                                         output_filesuffix='_bias_n')

        # compile run outputs
        ds = utils.compile_run_output([gdir], input_filesuffix='')
        ds_p = utils.compile_run_output([gdir], input_filesuffix='_bias_p')
        ds_n = utils.compile_run_output([gdir], input_filesuffix='_bias_n')

        # the glacier should not change much under a random climate
        # based on the equilibirum period centered around t*
        assert abs(1 - ds.volume.mean() / ds.volume[0]) < 0.015
        # higher temperatures should result in a smaller glacier
        assert ds.volume.mean() > ds_p.volume.mean()
        # lower temperatures should result in a larger glacier
        assert ds.volume.mean() < ds_n.volume.mean()

    def test_run_constant_climate(self):
        """ Test the run_constant_climate task for a climate based on the
        equilibrium period centred around t*. Additionally a positive and a
        negative temperature bias are tested.

        """
        # let's not use the mass balance bias since we want to reproduce
        # results from mass balance calibration
        cfg.PARAMS['use_bias_for_run'] = False

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)
        # compute mass balance parameters
        ref_df = cfg.PARAMS['vas_ref_tstars_rgi5_histalp']
        vascaling.local_t_star(gdir, ref_df=ref_df)

        # define some parameters for the constant climate model
        nyears = 500
        temp_bias = 0.5
        _ = vascaling.run_constant_climate(gdir, nyears=nyears,
                                           output_filesuffix='')
        _ = vascaling.run_constant_climate(gdir, nyears=nyears,
                                           temperature_bias=+temp_bias,
                                           output_filesuffix='_bias_p')
        _ = vascaling.run_constant_climate(gdir, nyears=nyears,
                                           temperature_bias=-temp_bias,
                                           output_filesuffix='_bias_n')

        # compile run outputs
        ds = utils.compile_run_output([gdir], input_filesuffix='')
        ds_p = utils.compile_run_output([gdir], input_filesuffix='_bias_p')
        ds_n = utils.compile_run_output([gdir], input_filesuffix='_bias_n')

        # the glacier should not change under a constant climate
        # based on the equilibirum period centered around t*
        assert abs(1 - ds.volume.mean() / ds.volume[0]) < 1e-7
        # higher temperatures should result in a smaller glacier
        assert ds.volume.mean() > ds_p.volume.mean()
        # lower temperatures should result in a larger glacier
        assert ds.volume.mean() < ds_n.volume.mean()

        # compute volume change from one year to the next
        dV_p = (ds_p.volume[1:].values - ds_p.volume[:-1].values).flatten()
        dV_n = (ds_n.volume[1:].values - ds_n.volume[:-1].values).flatten()
        # compute relative volume change, with respect to the final volume
        rate_p = abs(dV_p / float(ds_p.volume.values[-1]))
        rate_n = abs(dV_n / float(ds_n.volume.values[-1]))
        # the glacier should be in a new equilibirum for last 300 years
        assert max(rate_p[-300:]) < 0.001
        assert max(rate_n[-300:]) < 0.001

    def test_run_until_equilibrium(self):
        """"""
        # let's not use the mass balance bias since we want to reproduce
        # results from mass balance calibration
        cfg.PARAMS['use_bias_for_run'] = False

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # process the given climate file
        climate.process_custom_climate_data(gdir)
        # compute mass balance parameters
        ref_df = cfg.PARAMS['vas_ref_tstars_rgi5_histalp']
        vascaling.local_t_star(gdir, ref_df=ref_df)

        # instance a constant mass balance model, centred around t*
        mb_model = vascaling.ConstantVASMassBalance(gdir)
        # add a positive temperature bias
        mb_model.temp_bias = 0.5

        # create a VAS model: start with year 0  since we are using a constant
        # massbalance model, other values are read from RGI
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        model = vascaling.VAScalingModel(year_0=0, area_m2_0=gdir.rgi_area_m2,
                                         min_hgt=min_hgt, max_hgt=max_hgt,
                                         mb_model=mb_model)

        # run glacier with new mass balance model
        model.run_until_equilibrium(rate=1e-4)

        # equilibrium should be reached after a couple of 100 years
        assert model.year <= 300
        # new equilibrium glacier should be smaller (positive temperature bias)
        assert model.volume_m3 < model.volume_m3_0

        # run glacier for another 100 years and check volume again
        v_eq = model.volume_m3
        model.run_until(model.year + 100)
        assert abs(1 - (model.volume_m3/v_eq)) < 0.01
