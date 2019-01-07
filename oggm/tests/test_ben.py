"""TODO: Description """

# External libs
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import os
import shutil
import datetime

# import unittest
import unittest
import pytest

# import gis libs
import salem
import rasterio
import geopandas as gpd
import shapely.geometry as shpg


# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH, BASENAMES

from oggm import utils
from oggm.utils import get_demo_file, tuple2int
from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist)

from oggm import workflow, tasks

from oggm.core import (gis, ben, inversion, climate, centerlines, flowline,
                       massbalance)

from oggm.core.massbalance import MassBalanceModel

from oggm.tests.funcs import get_test_dir, patch_url_retrieve_github


class TestBensModel(unittest.TestCase):
    """ Unittest TestCase testing the implementation of Bens model. @TODO """

    def setUp(self):
        """ Creates two different test directories, one for the HistAlp or costume climate file
        and one for the CRU climate file. OGGM initialisation, paths and parameters are set.
        """
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.testdir_cru = os.path.join(get_test_dir(), 'tmp_prepro_cru')
        if not os.path.exists(self.testdir_cru):
            os.makedirs(self.testdir_cru)
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

    def tearDown(self):
        """Removes the test directories."""
        self.rm_dir()

    def rm_dir(self):
        """Removes the test directories."""
        shutil.rmtree(self.testdir)
        shutil.rmtree(self.testdir_cru)

    def clean_dir(self):
        """Cleans the test directories."""
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)
        shutil.rmtree(self.testdir_cru)
        os.makedirs(self.testdir_cru)

    def test_oggm_mb_climate(self):
        """ Testing the `mb_climate_on_height(gdir, heights)` function.

        Said function computes the monthly mass balance relevant climate parameters,
        such as melting temperature and solid precipitation. The tests are performed
        using the Hintereisferner and the HistAlp climate data,
        for three different settings:
        - all (climate) data points available
        - only one year (1802)
        - only two years (1803 - 1804)

        For each setting all of the following cases are tested:
        - shape of returned values (one value per month and elevation level)
        - returned melting temperature at reference elevation in comparison to reference temperature
        - same results for same elevation levels
        - only solid precipitation given high elevation (8000 m) i.e. subzero temperatures
        - zero solid precipitation given low elevation (-8000 m) i.e. highly positive temperatures
        - zero melting temperature given high elevation (8000 m) i.e. subzero temperatures

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
        # of the Hintereisferner HistAlp climate file for the entire time period
        # from October 1801 until September 2003
        # - surface height in m asl.
        # - total precipitation amount in kg/m2
        # - 2m air temperature in °C
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            # compute positive melting temperature
            ref_t = np.where(ref_t < cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        # define some (arbitrary) elevation levels
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        # get mass balance relevant climate parameters,
        # i.e. melting temperature and solid precipitation
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts)
        # remove precipitation scaling factor
        prcp /= cfg.PARAMS['prcp_scaling_factor']

        # define number of points of time axis (n_years * 12 month)
        ref_nt = 202*12
        # time variable is supposed to have 12 values per year
        self.assertTrue(len(time) == ref_nt)
        # temperature and precipitation variables are supposed to have
        # one row per elevation level and one column per month/year combination
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        # compare the mass balance relevant temperature at reference elevation
        # to reference melting temperature
        np.testing.assert_allclose(temp[0, :], ref_t)
        # test if the method returns the same temperature values
        # for the same elevation level
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        # test if the method returns the same precipitation values
        # for the same elevation level
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        # the temperature at an elevation of 8000 meters (hgts[3])
        # has to be below 0°C and therefore all fallen precipitation
        # will be solid
        np.testing.assert_allclose(prcp[3, :], ref_p)
        # the temperature at an elevation of -8000 meters (hgts[2])
        # has to be high enough so that all fallen precipitation
        # will be liquid (i.e. solid precipitation == 0)
        np.testing.assert_allclose(prcp[2, :], ref_p*0)
        # the temperature at an elevation of 8000 meters (hgts[3])
        # has to be below 0°C and therefore the melting temperature
        # will be zero too
        np.testing.assert_allclose(temp[3, :], ref_t*0)

        # perform the same tests but for data of 1802 only
        yr = [1802, 1802]
        # get mass balance relevant climate parameters
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts,
                                                        year_range=yr)
        # remove precipitation scaling parameter
        prcp /= cfg.PARAMS['prcp_scaling_factor']
        # define number of points of time axis (12 month)
        ref_nt = 1*12
        # same assertions as above
        self.assertTrue(len(time) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], ref_t[0:12])
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], ref_p[0:12])
        np.testing.assert_allclose(prcp[2, :], ref_p[0:12]*0)
        np.testing.assert_allclose(temp[3, :], ref_p[0:12]*0)

        # perform the same tests but for data between 1803 and 1804 only
        yr = [1803, 1804]
        # get mass balance relevant climate parameters
        time, temp, prcp = climate.mb_climate_on_height(gdir, hgts,
                                                        year_range=yr)
        # remove precipitation scaling factor
        prcp /= cfg.PARAMS['prcp_scaling_factor']
        # define number of points of time axis (2 years * 12 month)
        ref_nt = 2*12
        # same assertions as above
        self.assertTrue(len(time) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], ref_t[12:36])
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], ref_p[12:36])
        np.testing.assert_allclose(prcp[2, :], ref_p[12:36]*0)
        np.testing.assert_allclose(temp[3, :], ref_p[12:36]*0)

    def test_oggm_yearly_mb_climate(self):
        """ Testing the `mb_yearly_climate_on_height(gdir, heights)` function.

        Said function computes the yearly summed mass balance relevant climate parameters,
        such as melting temperature and solid precipitation. The tests are performed
        using the Hintereisferner and the HistAlp climate data,
        for two different settings
        - all (climate) data points available
        - only two years (1803 - 1804)
        once for the given elevation levels (`flatten=False`) and
        once as a glacier average (`flatten=True`).

        The following cases are tested, but not all for each test setting.
        - shape of returned values (one value per year and elevation level)
        - returned melting temperature at reference elevation in comparison to reference temperature
        - same results for same elevation levels
        - only solid precipitation given high elevation (8000 m) i.e. subzero temperatures
        - zero solid precipitation given low elevation (-8000 m) i.e. highly positive temperatures
        - zero melting temperature given high elevation (8000 m) i.e. subzero temperatures

        """

        cfg.PARAMS['prcp_scaling_factor'] = 1

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        climate.process_custom_climate_data(gdir)

        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]
            ref_t = np.where(ref_t <= cfg.PARAMS['temp_melt'], 0,
                             ref_t - cfg.PARAMS['temp_melt'])

        # NORMAL --------------------------------------------------------------
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))

        yr = [1802, 1802]
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                year_range=yr)
        ref_nt = 1
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(years == 1802)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(temp[0, :], np.sum(ref_t[0:12]))
        np.testing.assert_allclose(temp[0, :], temp[1, :])
        np.testing.assert_allclose(prcp[0, :], prcp[1, :])
        np.testing.assert_allclose(prcp[3, :], np.sum(ref_p[0:12]))
        np.testing.assert_allclose(prcp[2, :], np.sum(ref_p[0:12])*0)
        np.testing.assert_allclose(temp[3, :], np.sum(ref_p[0:12])*0)

        yr = [1803, 1804]
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                year_range=yr)
        ref_nt = 2
        self.assertTrue(len(years) == ref_nt)
        np.testing.assert_allclose(years, yr)
        self.assertTrue(temp.shape == (4, ref_nt))
        self.assertTrue(prcp.shape == (4, ref_nt))
        np.testing.assert_allclose(prcp[2, :], [0, 0])
        np.testing.assert_allclose(temp[3, :], [0, 0])

        # FLATTEN -------------------------------------------------------------
        hgts = np.array([ref_h, ref_h, -8000, 8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                flatten=True)

        ref_nt = 202
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(temp.shape == (ref_nt,))
        self.assertTrue(prcp.shape == (ref_nt,))

        yr = [1802, 1802]
        hgts = np.array([ref_h])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                year_range=yr,
                                                                flatten=True)
        ref_nt = 1
        self.assertTrue(len(years) == ref_nt)
        self.assertTrue(years == 1802)
        self.assertTrue(temp.shape == (ref_nt,))
        self.assertTrue(prcp.shape == (ref_nt,))
        np.testing.assert_allclose(temp[:], np.sum(ref_t[0:12]))

        yr = [1802, 1802]
        hgts = np.array([8000])
        years, temp, prcp = climate.mb_yearly_climate_on_height(gdir, hgts,
                                                                year_range=yr,
                                                                flatten=True)
        np.testing.assert_allclose(prcp[:], np.sum(ref_p[0:12]))

    def test_oggm_local_t_star(self):
        """


        """
        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # set the precipitation scaling factor @ASK
        cfg.PARAMS['prcp_scaling_factor'] = 2.9

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and the glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
        # run centerline prepro tasks
        centerlines.compute_centerlines(gdir)
        centerlines.initialize_flowlines(gdir)
        centerlines.catchment_area(gdir)
        centerlines.catchment_width_geom(gdir)
        centerlines.catchment_width_correction(gdir)
        # process the given climate file
        climate.process_custom_climate_data(gdir)
        # compute mu candidates for every 31 year period
        climate.glacier_mu_candidates(gdir)
        # read reference glacier mass balance data
        mbdf = gdir.get_ref_mb_data()
        # compute the reference t* for the glacier
        # given the reference of mass balance measurements
        res = climate.t_star_from_refmb(gdir, mbdf=mbdf['ANNUAL_BALANCE'])
        t_star, bias = res['t_star'], res['bias']

        # compute local t* and the corresponding mu*
        climate.local_t_star(gdir, tstar=t_star, bias=bias)
        # compute the mu* for each flow line
        climate.mu_star_calibration(gdir)
        # use the mu candidate at t* as glacier wide reference value
        ci = gdir.read_pickle('climate_info')
        mu_ref = ci['mu_candidates_glacierwide'].loc[t_star]

        # Check for apparent mb to be zeros
        # ---------------------------------

        # read the inversion flow lines
        fls = gdir.read_pickle('inversion_flowlines')
        # initialize the total mass balance
        tmb = 0.
        # for each flow line
        for fl in fls:
            # each grid point must have a value of apparent mass balance
            self.assertTrue(fl.apparent_mb.shape == fl.widths.shape)
            # the flow line mu* must be similar to the glacier wide mu*
            np.testing.assert_allclose(mu_ref, fl.mu_star, atol=1e-3)
            # add the flow line apparent mass balance to the total mass balance
            tmb += np.sum(fl.apparent_mb * fl.widths)
            # the flux correction flag can not be set
            assert not fl.flux_needs_correction

        # the total apparant mass balance must be zero
        np.testing.assert_allclose(tmb, 0., atol=0.01)
        # the flux through the last grid point of the main flow line must be zero
        np.testing.assert_allclose(fls[-1].flux[-1], 0., atol=0.01)

        # read the result from the mu* calibration
        df = gdir.read_json('local_mustar')
        # check the "mu* all same" flag
        assert df['mu_star_allsame']
        # compare values of mu* between the local_mustar.json
        # and the climate_info.pkl files
        np.testing.assert_allclose(mu_ref, df['mu_star_flowline_avg'],
                                   atol=1e-3)
        np.testing.assert_allclose(mu_ref, df['mu_star_glacierwide'],
                                   atol=1e-3)

        # Check mass balance profile/gradient
        # -----------------------------------

        # read the inversion flow lines
        fls = gdir.read_pickle('inversion_flowlines')
        # create empty containers
        mb_on_h = np.array([])
        h = np.array([])
        # for each flow line
        for fl in fls:
            # get mass balance relevant climate parameters
            # on the elevation of each grid point
            y, t, p = climate.mb_yearly_climate_on_height(gdir, fl.surface_h)
            # compute average value per elevation level
            # over all years with reference mb measurements
            selind = np.searchsorted(y, mbdf.index)
            t = np.mean(t[:, selind], axis=1)
            p = np.mean(p[:, selind], axis=1)
            # compute point mass balance per grid point
            # (i.e. elevation level) and add to container
            mb_on_h = np.append(mb_on_h, p - mu_ref * t)
            # add elevation levels to container
            h = np.append(h, fl.surface_h)

        # get reference mass balance profile data from WGMS
        dfg = gdir.get_ref_mb_profile().mean()
        # limit point mass balance to elevations below 3100 m asl. @ASK
        dfg = dfg[dfg.index < 3100]
        pok = np.where(h < 3100)
        # compute slope of mass balance profile
        from scipy.stats import linregress
        slope_obs, _, _, _, _ = linregress(dfg.index, dfg.values)
        slope_our, _, _, _, _ = linregress(h[pok], mb_on_h[pok])
        # slopes of observed and computed mass balance profile
        # must equal to at least 90%
        np.testing.assert_allclose(slope_obs, slope_our, rtol=0.1)

        # reset precipitation scaling factor to default value
        cfg.PARAMS['prcp_scaling_factor'] = 2.5

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
        # of the Hintereisferner HistAlp climate file for the entire time period
        # from October 1801 until September 2003
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
        temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad, ref_hgt=ref_h,
                                                   terminus_hgt=ref_h,
                                                   temp_anomaly=temp_anomaly)
        np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # the terminus temperature must equal the input terperature
        # if the gradient is zero
        for term_h in np.array([-100, 0, 100]) + ref_h:
            temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad=0, ref_hgt=ref_h,
                                                       terminus_hgt=term_h,
                                                       temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # now test the routine with actual elevation differences
        # and a non zero temperature gradient
        for h_diff in np.array([-100, 0, 100]):
            term_h = ref_h + h_diff
            temp_diff = temp_grad * h_diff
            temp_terminus = ben._compute_temp_terminus(ref_t, temp_grad, ref_hgt=ref_h,
                                                       terminus_hgt=term_h,
                                                       temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly + temp_diff)

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
        # of the Hintereisferner HistAlp climate file for the entire time period
        # from October 1801 until September 2003
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

        # if the terminus temperature is below the threshold for solid precipitation
        # all fallen precipitation must be solid
        temp_terminus = ref_t * 0 + temp_all_solid
        solid_prcp = ben._compute_solid_prcp(ref_p, prcp_factor, ref_hgt, min_hgt, max_hgt,
                                             temp_terminus, temp_all_solid, temp_grad,
                                             prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, ref_p)

        # if the temperature at the maximal elevation is above the threshold
        # for solid precipitation all fallen precipitation must be liquid
        temp_terminus = ref_t + 100
        solid_prcp = ben._compute_solid_prcp(ref_p, prcp_factor, ref_hgt, min_hgt, max_hgt,
                                             temp_terminus, temp_all_solid, temp_grad,
                                             prcp_grad=0, prcp_anomaly=0)
        np.testing.assert_allclose(solid_prcp, 0)

    def test_yearly_mb_temp_prcp(self):
        """ Test the routine which returns the yearly mass balance relevant
        climate parameters, i.e. positive melting temperature and solid precipitation.
        The testing target is the output of the corresponding OGGM routine
        `get_yearly_mb_climate_on_glacier(gdir)`."""

        # read the Hintereisferner DEM
        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        # initialize the GlacierDirectory
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)
        # define the local grid and glacier mask
        gis.define_glacier_region(gdir, entity=entity)
        gis.glacier_masks(gdir)
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
        # of the Hintereisferner HistAlp climate file for the entire time period
        # from October 1801 until September 2003
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

        # use the OGGM methode to get the mass balance relevant climate parameters
        years_oggm, temp_oggm, prcp_oggm = climate.mb_yearly_climate_on_glacier(gdir)

        # compute averages
        yearly_temp_mean = temp.mean()
        yearly_prcp_mean = prcp.mean()
        yearly_temp_oggm_mean = temp_oggm.mean()
        yearly_prcp_oggm_mean = prcp_oggm.mean()

        # the average glacier wide energy input must be less than at the terminus
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
        years_height, temp_height, _ = climate.mb_yearly_climate_on_height(gdir,
                                                                           heights,
                                                                           flatten=False)
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
        years_height, _, prcp_height = climate.mb_yearly_climate_on_height(gdir,
                                                                           heights,
                                                                           flatten=True)
        # compute correlation
        prcp_corr = np.corrcoef(prcp, prcp_height)[0,1]
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
        ben_mb = ben.BenMassBalance(gdir, mu_star=ben_params['mu_star'], bias=0)
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
        raise NotImplementedError

    def test_monthly_mb(self):
        raise NotImplementedError

    def test_annual_climate(self):
        raise NotImplementedError

    def test_annual_mb(self):
        raise NotImplementedError

    def test_monthly_specific_mb(self):
        raise NotImplementedError

    def test_specific_mb(self):
        raise NotImplementedError
