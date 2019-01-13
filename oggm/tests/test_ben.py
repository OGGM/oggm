"""TODO: Description """

# External libs
import numpy as np
import os
import shutil

# import unittest
import unittest

# import gis libs
import geopandas as gpd


# import OGGM modules
import oggm
import oggm.cfg as cfg

from oggm import utils
from oggm.utils import get_demo_file, ncDataset

from oggm.core import (gis, ben, climate, centerlines, massbalance)

from oggm.tests.funcs import get_test_dir


class TestBensModel(unittest.TestCase):
    """ Unittest TestCase testing the implementation of Bens model. @TODO """

    def setUp(self):
        """ Creates two different test directories, one for the HistAlp or
            costume climate file and one for the CRU climate file.
            OGGM initialisation, paths and parameters are set.
            TODO: do I need a CRU test directory?!
        """
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        # self.testdir_cru = os.path.join(get_test_dir(), 'tmp_prepro_cru')
        # if not os.path.exists(self.testdir_cru):
        #     os.makedirs(self.testdir_cru)
        self.clean_dir()

        # Init
        cfg.initialize()
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['run_mb_calibration'] = True
        cfg.PARAMS['baseline_climate'] = ''
        # coveralls.io has issues if multiprocessing is enabled
        cfg.PARAMS['use_multiprocessing'] = False

    def tearDown(self):
        """Removes the test directories."""
        self.rm_dir()

    def rm_dir(self):
        """Removes the test directories."""
        shutil.rmtree(self.testdir)
        # shutil.rmtree(self.testdir_cru)

    def clean_dir(self):
        """Cleans the test directories."""
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)
        # shutil.rmtree(self.testdir_cru)
        # os.makedirs(self.testdir_cru)

    def test_terminus_temp(self):
        """Testing the subroutine which computes the terminus temperature
        from the given climate file and glacier DEM. Pretty straight forward
        and somewhat useless, but nice finger exercise."""

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
        temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad,
                                                   ref_hgt=ref_h,
                                                   terminus_hgt=ref_h,
                                                   temp_anomaly=temp_anomaly)
        np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # the terminus temperature must equal the input terperature
        # if the gradient is zero
        for term_h in np.array([-100, 0, 100]) + ref_h:
            temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad=0,
                                                       ref_hgt=ref_h,
                                                       terminus_hgt=term_h,
                                                       temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # now test the routine with actual elevation differences
        # and a non zero temperature gradient
        for h_diff in np.array([-100, 0, 100]):
            term_h = ref_h + h_diff
            temp_diff = temp_grad * h_diff
            temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad,
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
        solid_prcp = ben._compute_solid_prcp(ref_p, prcp_factor, ref_hgt,
                                             min_hgt, max_hgt,
                                             temp_terminus, temp_all_solid,
                                             temp_grad,
                                             prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, ref_p)

        # if the temperature at the maximal elevation is above the threshold
        # for solid precipitation all fallen precipitation must be liquid
        temp_terminus = ref_t + 100
        solid_prcp = ben._compute_solid_prcp(ref_p, prcp_factor, ref_hgt,
                                             min_hgt, max_hgt,
                                             temp_terminus, temp_all_solid,
                                             temp_grad,
                                             prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, 0)

    def test_min_max_elevation(self):
        """ Test the helper method which computes the minimal and
            maximal glacier surface elevation in meters asl,
            from the given DEM and glacier outline.
        """
        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)

        # set targets
        min_target = 2446.0
        max_target = 3684.0
        # get values from method
        min_hgt, max_hgt = ben.get_min_max_elevation(gdir)
        assert min_target == min_hgt
        assert max_target == max_hgt

    def test_yearly_mb_temp_prcp(self):
        """ Test the routine which returns the yearly mass balance relevant
        climate parameters, i.e. positive melting temperature and solid
        precipitation. The testing target is the output of the corresponding
        OGGM routine `get_yearly_mb_climate_on_glacier(gdir)`."""

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
        prcp_anomaly = 0
        prcp_grad = 3e-4
        prcp_factor = cfg.PARAMS['prcp_scaling_factor']
        prcp_factor = 1
        temp_all_solid = cfg.PARAMS['temp_all_solid']
        temp_grad = cfg.PARAMS['temp_default_gradient']

        # get yearly sums of terminus temperature and solid precipitation
        years, temp, prcp = ben.get_yearly_mb_temp_prcp(gdir)

        # use the OGGM methode to get the mass balance
        # relevant climate parameters
        years_oggm, temp_oggm, prcp_oggm = \
            climate.mb_yearly_climate_on_glacier(gdir)

        # compute averages
        yearly_temp_mean = temp.mean()
        yearly_prcp_mean = prcp.mean()
        yearly_temp_oggm_mean = temp_oggm.mean()
        yearly_prcp_oggm_mean = prcp_oggm.mean()

        # the average glacier wide energy input
        # must be less than at the terminus
        assert(yearly_temp_oggm_mean <= yearly_temp_mean)
        # the average glacier wide mass input must be higher
        # TODO: does it acutally?! And if so, why?! @ASK
        assert (yearly_prcp_oggm_mean >= yearly_prcp_mean)

        # compute differences to mean
        temp_diff = temp - yearly_temp_mean
        temp_oggm_diff = temp_oggm - yearly_temp_oggm_mean
        prcp_diff = prcp - yearly_prcp_mean
        prcp_oggm_diff = prcp_oggm - yearly_prcp_oggm_mean
        # compute correlation between anomalies
        temp_diff_corr = np.corrcoef(temp_diff, temp_oggm_diff)[0, 1]
        prcp_diff_corr = np.corrcoef(prcp_diff, prcp_oggm_diff)[0, 1]

        # correlation must be higher than set threshold
        corr_threshold = 0.8
        assert(temp_diff_corr >= corr_threshold)
        corr_threshold = 0.9
        assert(prcp_diff_corr >= corr_threshold)

        # get terminus temperature using the OGGM routine
        fpath = gdir.get_filepath('gridded_data')
        with ncDataset(fpath) as nc:
            mask = nc.variables['glacier_mask'][:]
            topo = nc.variables['topo'][:]
        heights = np.array([np.min(topo[np.where(mask == 1)])])
        years_height, temp_height, _ = \
            climate.mb_yearly_climate_on_height(gdir, heights, flatten=False)

        temp_height = temp_height[0]
        # compute correlation
        temp_term_corr = np.corrcoef(temp, temp_height)[0, 1]
        # both temperature time series have to be equal
        corr_threshold = 1
        assert(temp_term_corr >= corr_threshold)
        np.testing.assert_allclose(temp, temp_height)

        # get solid precipitation averaged over the glacier (not weighted with widths)
        fls = gdir.read_pickle('inversion_flowlines')
        heights = np.array([])
        for fl in fls:
            heights = np.append(heights, fl.surface_h)
        years_height, _, prcp_height = \
            climate.mb_yearly_climate_on_height(gdir, heights, flatten=True)
        # compute correlation
        prcp_corr = np.corrcoef(prcp, prcp_height)[0, 1]
        # correlation must be higher than set threshold
        corr_threshold = 0.90
        assert (prcp_corr >= corr_threshold)

        # TODO: assert absolute values (or differences) of precipitation @ASK

        pass

    def test_local_t_star(self):
        """ The cumulative specific mass balance over the 31-year
            climate period centered around t* must be zero,
            given a successful mu* calibration. The mass balance
            bias is not applied.

            TODO: The comparison with the MB profiles is omitted for now.
        """
        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # set the precipitation scaling factor
        # TODO: omit or keep?! Only necessary for profiles
        # cfg.PARAMS['prcp_scaling_factor'] = 2.9

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
        # read reference glacier mass balance data
        mbdf = gdir.get_ref_mb_data()
        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # compute local t* and the corresponding mu*
        ben.local_t_star(gdir, tstar=t_star, bias=bias)
        # read calibration results
        ben_params = gdir.read_json('ben_params')

        # get min and max glacier elevation
        min_hgt, max_hgt = ben.get_min_max_elevation(gdir)

        # define mass balance model
        ben_mb = ben.BenMassBalance(gdir,
                                    mu_star=ben_params['mu_star'],
                                    bias=0)
        # define 31-year climate period around t*
        mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
        years = np.arange(t_star - mu_hp, t_star + mu_hp + 1)
        # iterate over all years in climate period
        # and compute specific mass balance
        mb_yearly = np.empty(years.size)
        for i, year in enumerate(years):
            mb_yearly[i] = ben_mb.get_specific_mb(min_hgt, max_hgt, year)

        # compute sum over climate period
        mb_sum = np.sum(mb_yearly)
        # check for apparent mb to be zero (to the third decimal digit)
        np.testing.assert_allclose(mb_sum, 0, atol=1e-3)

    def test_monthly_climate(self):
        """ Compute and return monthly positive terminus temperature
        and solid precipitation amount for given month.

        :param min_hgt: (float) glacier terminus elevation [m asl.]
        :param max_hgt: (float) maximal glacier surface elevation [m asl.]
        :param year: (float) floating year, following the
            hydrological year convention
        :return:
            temp_for_melt: (float) positive terminus temperature [°C]
            prcp_solid: (float) solid precipitation amount [kg/m^2]
        """
        # set min and max height at reference
        # test size/lenght of returned data sets
        # test absolute values (mean, sum, min, max?!)
        # test

    def _setup_mb_test(self):
        """ Avoiding a chunk of code duplicate. """
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
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # compute local t* and the corresponding mu*
        ben.local_t_star(gdir, tstar=t_star, bias=bias)

        # run OGGM mu* calibration
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        climate.mu_star_calibration(gdir)

        # pass the GlacierDirectory
        return gdir

    def test_monthly_climate(self):
        """ Test the routine getting the monthly climate against
        the routine getting annual climate. """
        # run all needed prepo tasks
        gdir = self._setup_mb_test()

        # get relevant glacier surface elevation
        min_hgt, max_hgt = ben.get_min_max_elevation(gdir)

        # instance the mass balance models
        mbmod = ben.BenMassBalance(gdir)

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
        """ Test my routine against the corresponding OGGM routine from
        the `PastMassBalance()` model. """
        # run all needed prepo tasks
        gdir = self._setup_mb_test()

        # get relevant glacier surface elevation
        min_hgt, max_hgt = ben.get_min_max_elevation(gdir)
        heights = np.array([min_hgt, (min_hgt + max_hgt) / 2, max_hgt])

        # instance the mass balance models
        ben_mbmod = ben.BenMassBalance(gdir)
        past_mbmod = massbalance.PastMassBalance(gdir)

        # specify a year
        year = 1975
        # get mass balance relevant climate information
        temp_for_melt_ben, prcp_solid_ben = \
            ben_mbmod.get_annual_climate(min_hgt, max_hgt, year)
        _, temp_for_melt_oggm, _, prcp_solid_oggm = \
            past_mbmod.get_annual_climate(heights, year)

        # prepare my (monthly) values for comparison
        temp_for_melt_ben = temp_for_melt_ben.sum()
        prcp_solid_ben = prcp_solid_ben.sum()

        # compare terminus temperature
        np.testing.assert_allclose(temp_for_melt_ben,
                                   temp_for_melt_oggm[0],
                                   atol=1e-3)
        # compare solid precipitation
        assert prcp_solid_ben >= prcp_solid_oggm[0]
        assert abs(1 - prcp_solid_ben/prcp_solid_oggm[1]) <= 0.15
        assert prcp_solid_ben <= prcp_solid_oggm[2]

    def test_annual_mb(self):
        pass

    def test_monthly_mb(self):
        pass

    def test_monthly_specific_mb(self):
        pass

    def test_specific_mb(self):
        """ Compare the specific mass balance to the one computed
        using the OGGM function of the PastMassBalance model. """
        # run all needed prepo tasks
        gdir = self._setup_mb_test()

        # define temporal range
        ys = 1802
        ye = 2003
        years = np.arange(ys, ye + 1)

        # get relevant glacier surface elevation
        min_hgt, max_hgt = ben.get_min_max_elevation(gdir)

        # get flowlines
        fls = gdir.read_pickle('inversion_flowlines')

        # instance mb models
        ben_mbmod = ben.BenMassBalance(gdir)
        past_mbmod = massbalance.PastMassBalance(gdir)

        # create empty container
        past_mb = np.empty(years.size)
        ben_mb = np.empty(years.size)
        # get specific mass balance for all years
        for i, year in enumerate(years):
            past_mb[i] = past_mbmod.get_specific_mb(fls=fls, year=year)
            ben_mb[i] = ben_mbmod.get_specific_mb(min_hgt, max_hgt, year)

        # compute ans check correlation
        assert np.corrcoef(past_mb, ben_mb)[0, 1] > 0.9

        # compute average
        past_avg = past_mb.mean()
        ben_avg = ben_mb.mean()
        # check relative error
        assert (1 - ben_avg / past_avg) < 0.3

        # check signs
        past_sign = np.sign(past_mb)
        ben_sign = np.sign(ben_mb)
        assert np.corrcoef(past_sign, ben_sign)[0, 1] >= 0.75
