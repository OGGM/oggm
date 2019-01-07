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
