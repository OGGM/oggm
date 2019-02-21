"""Flowline modelling: bed shapes and model numerics.


"""
# Builtins
import logging
import warnings
import copy
from collections import OrderedDict
from time import gmtime, strftime
import os

# External libs
import numpy as np
import shapely.geometry as shpg
import xarray as xr
import salem
import shutil

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance,
                                   RandomMassBalance)
from oggm.core.centerlines import Centerline, line_order

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, SEC_IN_HOUR
from oggm.cfg import G, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)


class Flowline(Centerline):
    """The is the input flowline for the model."""

    def __init__(self, line=None, dx=1, map_dx=None,
                 surface_h=None, bed_h=None, rgi_id=None):
        """ Instanciate.

        #TODO: documentation
        """

        # This is do add flexibility for testing. I have no time for fancier
        # stuff right now, but I will do this better one day:
        if dx is None:
            dx = 1.
        if line is None:
            coords = np.arange(0, len(surface_h)-0.5, dx)
            line = shpg.LineString(np.vstack([coords, coords*0.]).T)

        super(Flowline, self).__init__(line, dx, surface_h)

        self._thick = (surface_h - bed_h).clip(0.)
        self.map_dx = map_dx
        self.dx_meter = map_dx * self.dx
        self.bed_h = bed_h
        self.rgi_id = rgi_id

    @Centerline.widths.getter
    def widths(self):
        """Compute the widths out of H and shape"""
        return self.widths_m / self.map_dx

    @property
    def thick(self):
        """Needed for overriding later"""
        return self._thick

    @thick.setter
    def thick(self, value):
        self._thick = value.clip(0)

    @Centerline.surface_h.getter
    def surface_h(self):
        return self._thick + self.bed_h

    @surface_h.setter
    def surface_h(self, value):
        self.thick = value - self.bed_h

    @property
    def length_m(self):
        # We define the length a bit differently: but more robust
        pok = np.where(self.thick > 0.)[0]
        return len(pok) * self.dx_meter

    @property
    def volume_m3(self):
        return np.sum(self.section * self.dx_meter)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_m2(self):
        return np.sum(self.widths_m * self.dx_meter)

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        raise NotImplementedError()

    def to_dataset(self):
        """Makes an xarray Dataset out of the flowline."""

        h = self.surface_h
        nx = len(h)
        ds = xr.Dataset()
        ds.coords['x'] = np.arange(nx)
        ds.coords['c'] = [0, 1]
        ds['linecoords'] = (['x', 'c'], np.asarray(self.line.coords))
        ds['surface_h'] = (['x'],  h)
        ds['bed_h'] = (['x'],  self.bed_h)
        ds.attrs['class'] = type(self).__name__
        ds.attrs['map_dx'] = self.map_dx
        ds.attrs['dx'] = self.dx
        self._add_attrs_to_dataset(ds)
        return ds


class ParabolicBedFlowline(Flowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, bed_shape=None, rgi_id=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(ParabolicBedFlowline, self).__init__(line, dx, map_dx,
                                                   surface_h, bed_h,
                                                   rgi_id=rgi_id)

        assert np.all(np.isfinite(bed_shape))
        self.bed_shape = bed_shape

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return np.sqrt(4*self.thick/self.bed_shape)

    @property
    def section(self):
        return 2./3. * self.widths_m * self.thick

    @section.setter
    def section(self, val):
        self.thick = (0.75 * val * np.sqrt(self.bed_shape))**(2./3.)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['bed_shape'] = (['x'],  self.bed_shape)


class RectangularBedFlowline(Flowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, widths=None, rgi_id=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(RectangularBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id)

        self._widths = widths

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return self._widths * self.map_dx

    @property
    def section(self):
        return self.widths_m * self.thick

    @section.setter
    def section(self, val):
        self.thick = val / self.widths_m

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'],  self._widths)


class TrapezoidalBedFlowline(Flowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, widths=None, lambdas=None, rgi_id=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(TrapezoidalBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id)

        self._w0_m = widths * self.map_dx - lambdas * self.thick

        if np.any(self._w0_m <= 0):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        self._prec = np.where(lambdas == 0)

        self._lambdas = lambdas

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return self._w0_m + self._lambdas * self.thick

    @property
    def section(self):
        return (self.widths_m + self._w0_m) / 2 * self.thick

    @section.setter
    def section(self, val):
        b = 2 * self._w0_m
        a = 2 * self._lambdas
        with np.errstate(divide='ignore', invalid='ignore'):
            thick = (np.sqrt(b**2 + 4 * a * val) - b) / a
        thick[self._prec] = val[self._prec] / self._w0_m[self._prec]
        self.thick = thick

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'],  self.widths)
        ds['lambdas'] = (['x'],  self._lambdas)


class MixedBedFlowline(Flowline):
    """A more advanced Flowline."""

    def __init__(self, *, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, section=None, bed_shape=None,
                 is_trapezoid=None, lambdas=None, widths_m=None, rgi_id=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        width_m is optional - for thick=0
        """

        super(MixedBedFlowline, self).__init__(line=line, dx=dx, map_dx=map_dx,
                                               surface_h=surface_h.copy(),
                                               bed_h=bed_h.copy(),
                                               rgi_id=rgi_id)

        # To speedup calculations if no trapezoid bed is present
        self._do_trapeze = np.any(is_trapezoid)

        # Parabolic
        assert len(bed_shape) == self.nx
        self.bed_shape = bed_shape.copy()
        self._sqrt_bed = np.sqrt(bed_shape)

        # Trapeze
        assert len(lambdas) == self.nx
        assert len(is_trapezoid) == self.nx
        self._lambdas = lambdas.copy()
        self._ptrap = np.where(is_trapezoid)[0]
        self.is_trapezoid = is_trapezoid
        self.is_rectangular = self.is_trapezoid & (self._lambdas == 0)

        # Sanity
        self.bed_shape[is_trapezoid] = np.NaN
        self._lambdas[~is_trapezoid] = np.NaN

        # Here we have to compute the widths out of section and lambda
        thick = surface_h - bed_h
        with np.errstate(divide='ignore', invalid='ignore'):
            self._w0_m = section / thick - lambdas * thick / 2

        assert np.all(section >= 0)
        need_w = (section == 0) & is_trapezoid
        if np.any(need_w):
            if widths_m is None:
                raise ValueError('We need a non-zero section for trapezoid '
                                 'shapes unless you provide widths_m.')
            self._w0_m[need_w] = widths_m[need_w]

        self._w0_m[~is_trapezoid] = np.NaN

        if (np.any(self._w0_m[self._ptrap] <= 0) or
                np.any(~np.isfinite(self._w0_m[self._ptrap]))):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        assert np.all(self.bed_shape[~is_trapezoid] > 0)

        self._prec = np.where(is_trapezoid & (lambdas == 0))

        assert np.allclose(section, self.section)

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        out = np.sqrt(4*self.thick/self.bed_shape)
        if self._do_trapeze:
            out[self._ptrap] = (self._w0_m[self._ptrap] +
                                self._lambdas[self._ptrap] *
                                self.thick[self._ptrap])
        return out

    @property
    def section(self):
        out = 2./3. * self.widths_m * self.thick
        if self._do_trapeze:
            out[self._ptrap] = ((self.widths_m[self._ptrap] +
                                 self._w0_m[self._ptrap]) / 2 *
                                self.thick[self._ptrap])
        return out

    @section.setter
    def section(self, val):
        out = (0.75 * val * self._sqrt_bed)**(2./3.)
        if self._do_trapeze:
            b = 2 * self._w0_m[self._ptrap]
            a = 2 * self._lambdas[self._ptrap]
            with np.errstate(divide='ignore', invalid='ignore'):
                out[self._ptrap] = ((np.sqrt(b ** 2 + 4 * a * val[self._ptrap])
                                     - b) / a)
            out[self._prec] = val[self._prec] / self._w0_m[self._prec]
        self.thick = out

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""

        ds['section'] = (['x'],  self.section)
        ds['bed_shape'] = (['x'],  self.bed_shape)
        ds['is_trapezoid'] = (['x'], self.is_trapezoid)
        ds['widths_m'] = (['x'], self._w0_m)
        ds['lambdas'] = (['x'],  self._lambdas)


class FlowlineModel(object):
    """Interface to the actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=None, inplace=False, is_tidewater=False,
                 mb_elev_feedback='annual', check_for_boundaries=True):
        """Create a new flowline model from the flowlines and a MB model.

        Parameters
        ----------
        flowlines : list
            a list of Flowlines instances, sorted by order
        mb_model : MassBalanceModel
            the MB model to use
        y0 : int
            the starting year of the simultation
        glen_a : float
            glen's parameter A
        fs: float
            sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        is_tidewater : bool
            this changes how the last grid points of the domain are handled
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries : bool
            whether the model should raise an error when the glacier exceeds
            the domain boundaries.
        """

        self.is_tidewater = is_tidewater

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback
        self.mb_model = mb_model

        # Defaults
        if glen_a is None:
            glen_a = cfg.PARAMS['glen_a']
        if fs is None:
            fs = cfg.PARAMS['fs']
        self.glen_a = glen_a
        self.fs = fs
        self.glen_n = cfg.PARAMS['glen_n']
        self.rho = cfg.PARAMS['ice_density']

        self.check_for_boundaries = check_for_boundaries and not is_tidewater

        # we keep glen_a as input, but for optimisation we stick to "fd"
        self._fd = 2. / (cfg.PARAMS['glen_n']+2) * self.glen_a

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.fls = None
        self._trib = None
        self.reset_flowlines(flowlines, inplace=inplace)

    @property
    def mb_model(self):
        return self._mb_model

    @mb_model.setter
    def mb_model(self, value):
        # We need a setter because the MB func is stored as an attr too
        _mb_call = None
        if value:
            if self.mb_elev_feedback in ['always', 'monthly']:
                _mb_call = value.get_monthly_mb
            elif self.mb_elev_feedback in ['annual', 'never']:
                _mb_call = value.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_model = value
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = dict()
        self._mb_current_heights = dict()

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_flowlines(self, flowlines, inplace=False):
        """Reset the initial model flowlines"""

        if not inplace:
            flowlines = copy.deepcopy(flowlines)

        try:
            len(flowlines)
        except TypeError:
            flowlines = [flowlines]

        self.fls = flowlines

        # list of tributary coordinates and stuff
        trib_ind = []
        for fl in self.fls:
            if fl.flows_to is None:
                trib_ind.append((None, None, None, None))
                continue
            idl = self.fls.index(fl.flows_to)
            ide = fl.flows_to_indice
            if fl.flows_to.nx >= 9:
                gk = GAUSSIAN_KERNEL[9]
                id0 = ide-4
                id1 = ide+5
            elif fl.flows_to.nx >= 7:
                gk = GAUSSIAN_KERNEL[7]
                id0 = ide-3
                id1 = ide+4
            elif fl.flows_to.nx >= 5:
                gk = GAUSSIAN_KERNEL[5]
                id0 = ide-2
                id1 = ide+3
            trib_ind.append((idl, id0, id1, gk))
        self._trib = trib_ind

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return np.sum([f.area_m2 for f in self.fls])

    @property
    def volume_m3(self):
        return np.sum([f.volume_m3 for f in self.fls])

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def length_m(self):
        return self.fls[-1].length_m

    def get_mb(self, heights, year=None, fl_id=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        # Do we even have to optimise?
        if self.mb_elev_feedback == 'always':
            return self._mb_call(heights, year, fl_id=fl_id)

        # Ok, user asked for it
        if fl_id is None:
            raise ValueError('Need fls_id')

        if self.mb_elev_feedback == 'never':
            # The very first call we take the heights
            if fl_id not in self._mb_current_heights:
                # We need to reset just this tributary
                self._mb_current_heights[fl_id] = heights
            # All calls we replace
            heights = self._mb_current_heights[fl_id]

        date = utils.floatyear_to_date(year)
        if self.mb_elev_feedback in ['annual', 'never']:
            # ignore month changes
            date = (date[0], date[0])

        if self._mb_current_date == date:
            if fl_id not in self._mb_current_out:
                # We need to reset just this tributary
                self._mb_current_out[fl_id] = self._mb_call(heights, year,
                                                            fl_id=fl_id)
        else:
            # We need to reset all
            self._mb_current_date = date
            self._mb_current_out = dict()
            self._mb_current_out[fl_id] = self._mb_call(heights, year,
                                                        fl_id=fl_id)

        return self._mb_current_out[fl_id]

    def to_netcdf(self, path):
        """Creates a netcdf group file storing the state of the model."""

        flows_to_id = []
        for trib in self._trib:
            flows_to_id.append(trib[0] if trib[0] is not None else -1)

        ds = xr.Dataset()
        try:
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ds['flowlines'] = ('flowlines', np.arange(len(flows_to_id)))
            ds['flows_to_id'] = ('flowlines', flows_to_id)
            ds.to_netcdf(path)
            for i, fl in enumerate(self.fls):
                ds = fl.to_dataset()
                ds.to_netcdf(path, 'a', group='fl_{}'.format(i))
        finally:
            ds.close()

    def check_domain_end(self):
        """Returns False if the glacier reaches the domains bound."""
        return np.isclose(self.fls[-1].thick[-1], 0)

    def step(self, dt):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1):
        """Runs the model from the current year up to a given year y1.

        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run
        """

        t = (y1-self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(t-self.t)

        # Check for domain bounds
        if self.check_for_boundaries:
            if self.fls[-1].thick[-1] > 10:
                raise RuntimeError('Glacier exceeds domain boundaries.')

        # Check for NaNs
        for fl in self.fls:
            if np.any(~np.isfinite(fl.thick)):
                raise FloatingPointError('NaN in numerical solution.')

    def run_until_and_store(self, y1, run_path=None, diag_path=None,
                            store_monthly_step=False):
        """Runs the model and returns intermediate steps in xarray datasets.

        This function repeatedly calls FlowlineModel.run_until for either
        monthly or yearly time steps up till the upper time boundary y1.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run
        run_path : str
            Path and filename where to store the model run dataset
        diag_path : str
            Path and filename where to store the model diagnostics dataset
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly)

        Returns
        -------
        run_ds : xarray.Dataset
            stores the entire glacier geometry. It is useful to visualize the
            glacier geometry or to restart a new run from a modelled geometry.
            The glacier state is stored at the begining of each hydrological
            year (not in between in order to spare disk space).
        diag_ds : xarray.Dataset
            stores a few diagnostic variables such as the volume, area, length
            and ELA of the glacier.
        """

        # time
        yearly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)

        if store_monthly_step:
            monthly_time = utils.monthly_timeseries(self.yr, y1)
        else:
            monthly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)
        yrs, months = utils.floatyear_to_date(monthly_time)
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months)

        # init output
        if run_path is not None:
            self.to_netcdf(run_path)
        ny = len(yearly_time)
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]
        nm = len(monthly_time)
        sects = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        widths = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'OGGM model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)

        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # Variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        if self.is_tidewater:
            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
                                                          'calving flux')
            diag_ds['calving_m3'].attrs['unit'] = 'm 3'

        # Run
        j = 0
        for i, (yr, mo) in enumerate(zip(monthly_time, months)):
            self.run_until(yr)
            # Model run
            if mo == 1:
                for s, w, fl in zip(sects, widths, self.fls):
                    s[j, :] = fl.section
                    w[j, :] = fl.widths_m
                j += 1
            # Diagnostics
            diag_ds['volume_m3'].data[i] = self.volume_m3
            diag_ds['area_m2'].data[i] = self.area_m2
            diag_ds['length_m'].data[i] = self.length_m
            diag_ds['ela_m'].data[i] = self.mb_model.get_ela(year=yr)
            if self.is_tidewater:
                diag_ds['calving_m3'].data[i] = self.calving_m3_since_y0

        # to datasets
        run_ds = []
        for (s, w) in zip(sects, widths):
            ds = xr.Dataset()
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                 gmtime())
            ds.coords['time'] = yearly_time
            ds['time'].attrs['description'] = 'Floating hydrological year'
            varcoords = OrderedDict(time=('time', yearly_time),
                                    year=('time', yearly_time))
            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                            coords=varcoords)
            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                            coords=varcoords)
            run_ds.append(ds)

        # write output?
        if run_path is not None:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(run_ds):
                ds.to_netcdf(run_path, 'a', group='fl_{}'.format(i),
                             encoding=encode)
        if diag_path is not None:
            diag_ds.to_netcdf(diag_path)

        return run_ds, diag_ds

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """ Runs the model until an equilibrium state is reached.

        Be careful: This only works for CONSTANT (not time-dependant)
        mass-balance models.
        Otherwise the returned state will not be in equilibrium! Don't try to
        calculate an equilibrium state with a RandomMassBalance model!
        """

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if np.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = np.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')


class FluxBasedModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=0.05,
                 min_dt=1*SEC_IN_HOUR, max_dt=10*SEC_IN_DAY,
                 time_stepping='user',
                 **kwargs):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """
        super(FluxBasedModel, self).__init__(flowlines, mb_model=mb_model,
                                             y0=y0, glen_a=glen_a, fs=fs,
                                             inplace=inplace,
                                             **kwargs)

        if time_stepping == 'ambitious':
            cfl_number = 0.1
            min_dt = 1*SEC_IN_DAY
            max_dt = 15*SEC_IN_DAY
        elif time_stepping == 'default':
            cfl_number = 0.05
            min_dt = 1*SEC_IN_HOUR
            max_dt = 10*SEC_IN_DAY
        elif time_stepping == 'conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR
            max_dt = 5*SEC_IN_DAY
        elif time_stepping == 'ultra-conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR / 10
            max_dt = 5*SEC_IN_DAY
        else:
            if time_stepping != 'user':
                raise ValueError('time_stepping not understood.')

        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.cfl_number = cfl_number
        self.calving_m3_since_y0 = 0.  # total calving since time y0

        # Do we want to use shape factors?
        self.sf_func = None
        # Use .get to obtain default None for non-existing key
        # necessary to pass some tests
        # TODO: change to direct dictionary query after tests are adapted?
        use_sf = cfg.PARAMS.get('use_shape_factor_for_fluxbasedmodel')
        if use_sf == 'Adhikari' or use_sf == 'Nye':
            self.sf_func = utils.shape_factor_adhikari
        elif use_sf == 'Huss':
            self.sf_func = utils.shape_factor_huss

        # Optim
        self._stags = []
        for fl, trib in zip(self.fls, self._trib):
            nx = fl.nx
            if trib[0] is not None:
                nx = fl.nx + 1
            elif self.is_tidewater:
                nx = fl.nx + 1
            a = np.zeros(nx+1)
            b = np.zeros(nx+1)
            c = np.zeros(nx+1)
            d = np.ones(nx+1)  # shape factor default is 1
            e = np.zeros(nx-1)
            f = np.zeros(nx)
            self._stags.append((a, b, c, d, e, f))

    def step(self, dt):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

        # Loop over tributaries to determine the flux rate
        flxs = []
        aflxs = []

        for fl, trib, (slope_stag, thick_stag, section_stag, sf_stag,
                       znxm1, znx) in zip(self.fls, self._trib, self._stags):

            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # Reset
            znxm1[:] = 0
            znx[:] = 0

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid points
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
            elif self.is_tidewater:
                # For tidewater glacier, we trick and set the outgoing thick
                # to zero (for numerical stability and this should quite OK
                # represent what happens at the calving tongue)
                surface_h = np.append(surface_h, surface_h[-1] - thick[-1])
                thick = np.append(thick, 0)
                section = np.append(section, 0)

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            if self.sf_func is not None:
                # TODO: maybe compute new shape factors only every year?
                sf = self.sf_func(fl.widths_m, fl.thick, fl.is_rectangular)
                if is_trib or self.is_tidewater:
                    # for water termination or inflowing tributary, the sf
                    # makes no sense
                    sf = np.append(sf, 1.)
                sf_stag[1:-1] = (sf[0:-1] + sf[1:]) / 2.
                sf_stag[[0, -1]] = sf[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho*G*slope_stag)**N
            u_stag = (thick_stag**(N+1)) * self._fd * rhogh * sf_stag**N + \
                     (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flx_stag = u_stag * section_stag / dx

            # Store the results
            if is_trib:
                flxs.append(flx_stag[:-1])
                aflxs.append(znxm1)
                u_stag = u_stag[:-1]
            elif self.is_tidewater:
                flxs.append(flx_stag[:-1])
                aflxs.append(znxm1)
                u_stag = u_stag[:-1]
            else:
                flxs.append(flx_stag)
                aflxs.append(znx)

            # CFL condition
            maxu = np.max(np.abs(u_stag))
            if maxu > 0.:
                _dt = self.cfl_number * dx / maxu
            else:
                _dt = self.max_dt
            if _dt < dt:
                dt = _dt

        # Time step
        self.dt_warning = dt < min_dt
        dt = np.clip(dt, min_dt, self.max_dt)

        # A second loop for the mass exchange
        for i, (fl, flx_stag, aflx, trib) in enumerate(zip(self.fls, flxs,
                                                           aflxs, self._trib)):

            dx = fl.dx_meter

            # Mass balance
            widths = fl.widths_m
            mb = self.get_mb(fl.surface_h, self.yr, fl_id=i)
            # Allow parabolic beds to grow
            widths = np.where((mb > 0.) & (widths == 0), 10., widths)
            mb = dt * mb * widths

            # Update section with flowing and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt +
                           aflx*dt + mb)

            # Keep positive values only and store
            fl.section = new_section.clip(0)

            # Add the last flux to the tributary
            # this is ok because the lines are sorted in order
            if trib[0] is not None:
                aflxs[trib[0]][trib[1]:trib[2]] += (flx_stag[-1].clip(0) *
                                                    trib[3])
            elif self.is_tidewater:
                # -2 because the last flux is zero per construction
                # TODO: not sure if this is the way to go yet,
                # but mass conservation is OK
                self.calving_m3_since_y0 += flx_stag[-2].clip(0)*dt*dx

        # Next step
        self.t += dt
        return dt


class MassConservationChecker(FluxBasedModel):
    """This checks if the FluzBasedmodel is conserving mass."""

    def __init__(self, flowlines, **kwargs):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """
        super(MassConservationChecker, self).__init__(flowlines, **kwargs)
        self.total_mass = 0.

    def step(self, dt):

        mbs = []
        sections = []
        for fl in self.fls:
            # Mass balance
            widths = fl.widths_m
            mb = self.get_mb(fl.surface_h, self.yr, fl_id=id(fl))
            mbs.append(mb * widths)
            sections.append(np.copy(fl.section))
            dx = fl.dx_meter

        dt = super(MassConservationChecker, self).step(dt)

        for mb, sec in zip(mbs, sections):
            mb = dt * mb
            # there can't be more negative mb than there is section
            # this isn't an exact solution unfortunately
            # TODO: exact solution for mass conservation
            mb = mb.clip(-sec)
            self.total_mass += np.sum(mb * dx)


class KarthausModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 fixed_dt=None, min_dt=SEC_IN_DAY,
                 max_dt=31*SEC_IN_DAY, inplace=False):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        #TODO: Changed from assumed N=3 to N
        """

        if len(flowlines) > 1:
            raise ValueError('Karthaus model does not work with tributaries.')

        super(KarthausModel, self).__init__(flowlines, mb_model=mb_model,
                                            y0=y0, glen_a=glen_a, fs=fs,
                                            inplace=inplace)
        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt
        dt = np.clip(dt, min_dt, self.max_dt)

        fl = self.fls[0]
        dx = fl.dx_meter
        width = fl.widths_m
        thick = fl.thick

        MassBalance = self.get_mb(fl.surface_h, self.yr, fl_id=id(fl))

        SurfaceHeight = fl.surface_h

        # Surface gradient
        SurfaceGradient = np.zeros(fl.nx)
        SurfaceGradient[1:fl.nx-1] = (SurfaceHeight[2:] -
                                      SurfaceHeight[:fl.nx-2])/(2*dx)
        SurfaceGradient[-1] = 0
        SurfaceGradient[0] = 0

        # Diffusivity
        N = self.glen_n
        Diffusivity = width * (self.rho*G)**3 * thick**3 * SurfaceGradient**2
        Diffusivity *= 2/(N+2) * self.glen_a * thick**2 + self.fs

        # on stagger
        DiffusivityStaggered = np.zeros(fl.nx)
        SurfaceGradientStaggered = np.zeros(fl.nx)

        DiffusivityStaggered[1:] = (Diffusivity[:fl.nx-1] + Diffusivity[1:])/2.
        DiffusivityStaggered[0] = Diffusivity[0]

        SurfaceGradientStaggered[1:] = (SurfaceHeight[1:] -
                                        SurfaceHeight[:fl.nx-1])/dx
        SurfaceGradientStaggered[0] = 0

        GradxDiff = SurfaceGradientStaggered * DiffusivityStaggered

        # Yo
        NewIceThickness = np.zeros(fl.nx)
        NewIceThickness[:fl.nx-1] = (thick[:fl.nx-1] + (dt/width[0:fl.nx-1]) *
                                     (GradxDiff[1:]-GradxDiff[:fl.nx-1])/dx +
                                     dt * MassBalance[:fl.nx-1])

        NewIceThickness[-1] = thick[fl.nx-2]

        fl.thick = NewIceThickness.clip(0)

        # Next step
        self.t += dt


class MUSCLSuperBeeModel(FlowlineModel):
    """This model is based on Jarosch et al. 2013 (doi:10.5194/tc-7-229-2013)

       The equation references in the comments refer to the paper for clarity
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=None,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31*SEC_IN_DAY,
                 inplace=False):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        if len(flowlines) > 1:
            raise ValueError('MUSCL SuperBee model does not work with '
                             'tributaries.')

        super(MUSCLSuperBeeModel, self).__init__(flowlines, mb_model=mb_model,
                                                 y0=y0, glen_a=glen_a, fs=fs,
                                                 inplace=inplace)
        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def phi(self, r):
        """ a definition of the limiting scheme to use"""
        # minmod limiter Eq. 28
        # val_phi = numpy.maximum(0,numpy.minimum(1,r))

        # superbee limiter Eq. 29
        val_phi = np.maximum(0, np.minimum(2*r, 1), np.minimum(r, 2))

        return val_phi

    def step(self, dt):
        """Advance one step."""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Guarantee a precise arrival on a specific date if asked
            min_dt = dt if dt < self.min_dt else self.min_dt
            dt = np.clip(dt, min_dt, self.max_dt)

            fl = self.fls[0]
            dx = fl.dx_meter

            # Switch to the notation from the MUSCL_1D example
            # This is useful to ensure that the MUSCL-SuperBee code
            # is working as it has been benchmarked many times

            # mass balance
            m_dot = self.get_mb(fl.surface_h, self.yr, fl_id=id(fl))
            # get in the surface elevation
            S = fl.surface_h
            # get the bed
            B = fl.bed_h
            # define Glen's law here
            N = self.glen_n
            # this is the correct Gamma !!
            Gamma = 2.*self.glen_a*(self.rho*G)**N / (N+2.)
            # time stepping
            c_stab = 0.165

            # define the finite difference indices
            k = np.arange(0, fl.nx)
            kp = np.hstack([np.arange(1, fl.nx), fl.nx-1])
            kpp = np.hstack([np.arange(2, fl.nx), fl.nx-1, fl.nx-1])
            km = np.hstack([0, np.arange(0, fl.nx-1)])
            kmm = np.hstack([0, 0, np.arange(0, fl.nx-2)])

            # I'm gonna introduce another level of adaptive time stepping here,
            # which is probably not necessary. However I keep it to be
            # consistent with my benchmarked and tested code.
            # If the OGGM time stepping is correctly working, this loop
            # should never run more than once
            # Fabi: actually no, it is your job to choose the right time step
            # but you can let OGGM decide whether a new time step is needed
            # or not -> to be meliorated one day
            stab_t = 0.
            while stab_t < dt:
                H = S - B

                # MUSCL scheme up. "up" denotes here the k+1/2 flux boundary
                r_up_m = (H[k]-H[km])/(H[kp]-H[k])  # Eq. 27
                H_up_m = H[k] + 0.5 * self.phi(r_up_m)*(H[kp]-H[k])  # Eq. 23
                # Eq. 27, k+1 is used instead of k
                r_up_p = (H[kp]-H[k])/(H[kpp]-H[kp])
                # Eq. 24
                H_up_p = H[kp] - 0.5 * self.phi(r_up_p)*(H[kpp]-H[kp])

                # surface slope gradient
                s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((N-1.)/2.)
                # like Eq. 30, now using Eq. 23 instead of Eq. 24
                D_up_m = Gamma * H_up_m**(N+2.) * s_grad_up
                D_up_p = Gamma * H_up_p**(N+2.) * s_grad_up  # Eq. 30

                # Eq. 31
                D_up_min = np.minimum(D_up_m, D_up_p)
                # Eq. 32
                D_up_max = np.maximum(D_up_m, D_up_p)
                D_up = np.zeros(fl.nx)

                # Eq. 33
                cond = (S[kp] <= S[k]) & (H_up_m <= H_up_p)
                D_up[cond] = D_up_min[cond]
                cond = (S[kp] <= S[k]) & (H_up_m > H_up_p)
                D_up[cond] = D_up_max[cond]
                cond = (S[kp] > S[k]) & (H_up_m <= H_up_p)
                D_up[cond] = D_up_max[cond]
                cond = (S[kp] > S[k]) & (H_up_m > H_up_p)
                D_up[cond] = D_up_min[cond]

                # MUSCL scheme down. "down" denotes the k-1/2 flux boundary
                r_dn_m = (H[km]-H[kmm])/(H[k]-H[km])
                H_dn_m = H[km] + 0.5 * self.phi(r_dn_m)*(H[k]-H[km])
                r_dn_p = (H[k]-H[km])/(H[kp]-H[k])
                H_dn_p = H[k] - 0.5 * self.phi(r_dn_p)*(H[kp]-H[k])

                # calculate the slope gradient
                s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((N-1.)/2.)
                D_dn_m = Gamma * H_dn_m**(N+2.) * s_grad_dn
                D_dn_p = Gamma * H_dn_p**(N+2.) * s_grad_dn

                D_dn_min = np.minimum(D_dn_m, D_dn_p)
                D_dn_max = np.maximum(D_dn_m, D_dn_p)
                D_dn = np.zeros(fl.nx)

                cond = (S[k] <= S[km]) & (H_dn_m <= H_dn_p)
                D_dn[cond] = D_dn_min[cond]
                cond = (S[k] <= S[km]) & (H_dn_m > H_dn_p)
                D_dn[cond] = D_dn_max[cond]
                cond = (S[k] > S[km]) & (H_dn_m <= H_dn_p)
                D_dn[cond] = D_dn_max[cond]
                cond = (S[k] > S[km]) & (H_dn_m > H_dn_p)
                D_dn[cond] = D_dn_min[cond]

                # Eq. 37
                dt_stab = c_stab * dx**2. / max(max(abs(D_up)), max(abs(D_dn)))
                dt_use = min(dt_stab, dt-stab_t)
                stab_t = stab_t + dt_use

                # explicit time stepping scheme, Eq. 36
                div_q = (D_up * (S[kp] - S[k])/dx - D_dn *
                         (S[k] - S[km])/dx)/dx
                # Eq. 35
                S = S[k] + (m_dot + div_q)*dt_use

                # Eq. 7
                S = np.maximum(S, B)

        # Done with the loop, prepare output
        fl.thick = S-B

        # Next step
        self.t += dt


class FileModel(object):
    """Duck FlowlineModel which actually reads the stuff out of a nc file."""

    def __init__(self, path):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        self.fls = glacier_from_netcdf(path)
        dss = []
        for flid, fl in enumerate(self.fls):
            ds = xr.open_dataset(path, group='fl_{}'.format(flid))
            ds.load()
            dss.append(ds)
        self.last_yr = ds.year.values[-1]
        self.dss = dss
        self.reset_y0()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for ds in self.dss:
            ds.close()

    def reset_y0(self, y0=None):
        """Reset the initial model time"""

        if y0 is None:
            y0 = self.dss[0].time[0]
        self.y0 = y0
        self.yr = y0

    @property
    def area_m2(self):
        return np.sum([f.area_m2 for f in self.fls])

    @property
    def volume_m3(self):
        return np.sum([f.volume_m3 for f in self.fls])

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def length_m(self):
        return self.fls[-1].length_m

    def run_until(self, year=None, month=None):
        """Mimics the model's behavior.

        Is quite slow, I must say.
        """

        if month is not None:
            for fl, ds in zip(self.fls, self.dss):
                sel = ds.ts_section.isel(time=(ds.year == year) &
                                              (ds.month == month))
                fl.section = sel.values
        else:
            for fl, ds in zip(self.fls, self.dss):
                sel = ds.ts_section.sel(time=year)
                fl.section = sel.values
        self.yr = sel.time.values

    def area_m2_ts(self, rollmin=0):
        """rollmin is the number of years you want to smooth onto"""
        sel = 0
        for fl, ds in zip(self.fls, self.dss):
            widths = ds.ts_width_m.copy()
            widths[:] = np.where(ds.ts_section > 0., ds.ts_width_m, 0.)
            sel += widths.sum(dim='x') * fl.dx_meter
        sel = sel.to_series()
        if rollmin != 0:
            sel = sel.rolling(rollmin).min()
            sel.iloc[0:rollmin] = sel.iloc[rollmin]
        return sel

    def area_km2_ts(self, **kwargs):
        return self.area_m2_ts(**kwargs) * 1e-6

    def volume_m3_ts(self):
        sel = 0
        for fl, ds in zip(self.fls, self.dss):
            sel += ds.ts_section.sum(dim='x') * fl.dx_meter
        return sel.to_series()

    def volume_km3_ts(self):
        return self.volume_m3_ts() * 1e-9

    def length_m_ts(self, rollmin=0):
        """rollmin is the number of years you want to smooth onto"""
        fl = self.fls[-1]
        ds = self.dss[-1]
        sel = ds.ts_section.copy()
        sel[:] = ds.ts_section != 0.
        sel = sel.sum(dim='x') * fl.dx_meter
        sel = sel.to_series()
        if rollmin != 0:
            sel = sel.rolling(rollmin).min()
            sel.iloc[0:rollmin] = sel.iloc[rollmin]
        return sel


def flowline_from_dataset(ds):
    """Instanciates a flowline from an xarray Dataset."""

    cl = globals()[ds.attrs['class']]
    line = shpg.LineString(ds['linecoords'].values)
    args = dict(line=line, dx=ds.dx, map_dx=ds.map_dx,
                surface_h=ds['surface_h'].values,
                bed_h=ds['bed_h'].values)

    have = {'c', 'x', 'surface_h', 'linecoords', 'bed_h', 'z', 'p', 'n',
            'time', 'month', 'year', 'ts_width_m', 'ts_section'}
    missing_vars = set(ds.variables.keys()).difference(have)
    for k in missing_vars:
        data = ds[k].values
        if ds[k].dims[0] == 'z':
            data = data[0]
        args[k] = data
    return cl(**args)


def glacier_from_netcdf(path):
    """Instanciates a list of flowlines from an xarray Dataset."""

    with xr.open_dataset(path) as ds:
        fls = []
        for flid in ds['flowlines'].values:
            with xr.open_dataset(path, group='fl_{}'.format(flid)) as _ds:
                fls.append(flowline_from_dataset(_ds))

        for i, fid in enumerate(ds['flows_to_id'].values):
            if fid != -1:
                fls[i].set_flows_to(fls[fid])

    # Adds the line level
    for fl in fls:
        fl.order = line_order(fl)

    return fls


@entity_task(log, writes=['model_flowlines'])
def init_present_time_glacier(gdir):
    """Merges data from preprocessing tasks. First task after inversion!

    This updates the `mode_flowlines` file and creates a stand-alone numerical
    glacier ready to run.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Some vars
    map_dx = gdir.grid.dx
    def_lambda = cfg.PARAMS['trapezoid_lambdas']
    min_shape = cfg.PARAMS['mixed_min_shape']

    cls = gdir.read_pickle('inversion_flowlines')
    invs = gdir.read_pickle('inversion_output')

    # Fill the tributaries
    new_fls = []
    flows_to_ids = []
    for cl, inv in zip(cls, invs):

        # Get the data to make the model flowlines
        line = cl.line
        section = inv['volume'] / (cl.dx * map_dx)
        surface_h = cl.surface_h
        bed_h = surface_h - inv['thick']
        widths_m = cl.widths * map_dx

        assert np.all(cl.widths > 0)
        bed_shape = 4 * inv['thick'] / (cl.widths * map_dx) ** 2

        lambdas = inv['thick'] * np.NaN
        lambdas[bed_shape < min_shape] = def_lambda
        lambdas[inv['is_rectangular']] = 0.

        # Last pix of not tidewater are always parab (see below)
        if not gdir.is_tidewater and inv['is_last']:
            lambdas[-5:] = np.nan

        # Update bed_h where we now have a trapeze
        w0_m = cl.widths * map_dx - lambdas * inv['thick']
        b = 2 * w0_m
        a = 2 * lambdas
        with np.errstate(divide='ignore', invalid='ignore'):
            thick = (np.sqrt(b ** 2 + 4 * a * section) - b) / a
        ptrap = (lambdas != 0) & np.isfinite(lambdas)
        bed_h[ptrap] = cl.surface_h[ptrap] - thick[ptrap]

        # For the very last pixs of a glacier, the section might be zero after
        # the inversion, and the bedshapes are chaotic. We interpolate from
        # the downstream. This is not volume conservative
        if not gdir.is_tidewater and inv['is_last']:
            dic_ds = gdir.read_pickle('downstream_line')
            bed_shape[-5:] = np.nan

            # Interpolate
            bed_shape = utils.interp_nans(np.append(bed_shape,
                                                    dic_ds['bedshapes'][0]))
            bed_shape = bed_shape[:-1].clip(min_shape)

            # Correct the section volume
            h = inv['thick']
            section[-5:] = (2 / 3 * h * np.sqrt(4 * h / bed_shape))[-5:]

            # Add the downstream
            bed_shape = np.append(bed_shape, dic_ds['bedshapes'])
            lambdas = np.append(lambdas, dic_ds['bedshapes'] * np.NaN)
            section = np.append(section, dic_ds['bedshapes'] * 0.)
            surface_h = np.append(surface_h, dic_ds['surface_h'])
            bed_h = np.append(bed_h, dic_ds['surface_h'])
            widths_m = np.append(widths_m, dic_ds['bedshapes'] * 0.)
            line = dic_ds['full_line']

        nfl = MixedBedFlowline(line=line, dx=cl.dx, map_dx=map_dx,
                               surface_h=surface_h, bed_h=bed_h,
                               section=section, bed_shape=bed_shape,
                               is_trapezoid=np.isfinite(lambdas),
                               lambdas=lambdas,
                               widths_m=widths_m,
                               rgi_id=cl.rgi_id)

        # Update attrs
        nfl.mu_star = cl.mu_star

        if cl.flows_to:
            flows_to_ids.append(cls.index(cl.flows_to))
        else:
            flows_to_ids.append(None)

        new_fls.append(nfl)

    # Finalize the linkages
    for fl, fid in zip(new_fls, flows_to_ids):
        if fid:
            fl.set_flows_to(new_fls[fid])

    # Adds the line level
    for fl in new_fls:
        fl.order = line_order(fl)

    # Write the data
    gdir.write_pickle(new_fls, 'model_flowlines')


def robust_model_run(gdir, output_filesuffix=None, mb_model=None,
                     ys=None, ye=None, zero_initial_glacier=False,
                     init_model_fls=None, store_monthly_step=False,
                     **kwargs):
    """Trial-error-and-retry algorithm to run the flowline model.

     Runs a model simulation with the default time stepping scheme and,
     if failing, tries a more conservative one.
     This is a rather clumsy way to deal with numerical instabilities:
     for most glaciers the default numerical parameters work fine, but for some
     glaciers numerical instabilities might arise and lead to overfloating
     errors or NaNs. This catches those, and tries again.

     Pros of the method:
     - it is cheap, because the "quicker" time stepping is tested first
     - it is easy

     Cons:
     - the model might be unstable without necessarily leading to NaN in the
       solution. These cases will not be caught
     - it is inelegant

     Possibly a method based on mass-conservation checks would be more robust.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    mb_model : :py:class:`core.MassBalanceModel`
        a MassBalanceModel instance
    ys : int
        start year of the model run (default: from the config file)
    ye : int
        end year of the model run (default: from the config file)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
     """

    kwargs.setdefault('fs', cfg.PARAMS['fs'])
    kwargs.setdefault('glen_a', cfg.PARAMS['glen_a'])

    run_path = gdir.get_filepath('model_run', filesuffix=output_filesuffix,
                                 delete=True)
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)

    steps = ['default', 'conservative', 'ultra-conservative']
    for step in steps:
        log.info('(%s) trying %s time stepping scheme.', gdir.rgi_id, step)
        if init_model_fls is None:
            fls = gdir.read_pickle('model_flowlines')
        else:
            fls = copy.deepcopy(init_model_fls)
        if zero_initial_glacier:
            for fl in fls:
                fl.thick = fl.thick * 0.
        model = FluxBasedModel(fls, mb_model=mb_model, y0=ys,
                               inplace=True,
                               time_stepping=step,
                               is_tidewater=gdir.is_tidewater,
                               **kwargs)

        with np.warnings.catch_warnings():
            # For operational runs we ignore the warnings
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                model.run_until_and_store(ye, run_path=run_path,
                                          diag_path=diag_path,
                                          store_monthly_step=store_monthly_step
                                          )
            except (RuntimeError, FloatingPointError):
                if step == 'ultra-conservative':
                    raise
                continue

        # If we get here we good
        log.info('(%s) %s time stepping was successful!', gdir.rgi_id, step)
        break
    return model


@entity_task(log)
def run_random_climate(gdir, nyears=1000, y0=None, halfsize=15,
                       bias=None, seed=None, temperature_bias=None,
                       store_monthly_step=False,
                       climate_filename='climate_monthly',
                       climate_input_filesuffix='',
                       output_filesuffix='', init_model_fls=None,
                       zero_initial_glacier=False,
                       unique_samples=False,
                       **kwargs):
    """Runs the random mass-balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.robust_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed accross glaciers can
        be usefull if you want to have the same climate years for all of them
    temperature_bias : float
        add a bias to the temperature timeseries
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_monthly' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    unique_samples: bool
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     y0=y0, halfsize=halfsize,
                                     bias=bias, seed=seed,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix,
                                     unique_samples=unique_samples)

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=mb, ys=0, ye=nyears,
                            store_monthly_step=store_monthly_step,
                            init_model_fls=init_model_fls,
                            zero_initial_glacier=zero_initial_glacier,
                            **kwargs)


@entity_task(log)
def run_constant_climate(gdir, nyears=1000, y0=None, halfsize=15,
                         bias=None, temperature_bias=None,
                         store_monthly_step=False,
                         output_filesuffix='',
                         climate_filename='climate_monthly',
                         climate_input_filesuffix='',
                         init_model_fls=None,
                         zero_initial_glacier=False,
                         **kwargs):
    """Runs the constant mass-balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.robust_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation (default: as long as needed for reaching
        equilibrium)
    y0 : int
        central year of the requested climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    temperature_bias : float
        add a bias to the temperature timeseries
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_monthly' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=ConstantMassBalance,
                                     y0=y0, halfsize=halfsize,
                                     bias=bias, filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=mb, ys=0, ye=nyears,
                            store_monthly_step=store_monthly_step,
                            init_model_fls=init_model_fls,
                            zero_initial_glacier=zero_initial_glacier,
                            **kwargs)


@entity_task(log)
def run_from_climate_data(gdir, ys=None, ye=None, min_ys=None,
                          store_monthly_step=False,
                          climate_filename='climate_monthly',
                          climate_input_filesuffix='', output_filesuffix='',
                          init_model_filesuffix=None, init_model_yr=None,
                          init_model_fls=None, zero_initial_glacier=False,
                          **kwargs):
    """ Runs a glacier with climate input from e.g. CRU or a GCM.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.robust_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date)
    ye : int
        end year of the model run (no default: needs to be set)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is later.
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_monthly' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to starte from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
    if ye is None:
        raise InvalidParamsError('Need to set the `ye` kwarg!')
    if min_ys is not None:
        ys = ys if ys < min_ys else min_ys

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_run', filesuffix=init_model_filesuffix)
        with FileModel(fp) as fmod:
            if init_model_yr is None:
                init_model_yr = fmod.last_yr
            fmod.run_until(init_model_yr)
            init_model_fls = fmod.fls

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=PastMassBalance,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix)

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=mb, ys=ys, ye=ye,
                            store_monthly_step=store_monthly_step,
                            init_model_fls=init_model_fls,
                            zero_initial_glacier=zero_initial_glacier,
                            **kwargs)


@entity_task(log)
def merge_tributary_flowlines(main, tribs=[], filename='climate_monthly',
                              input_filesuffix=''):
    """Merge multiple tributary glaciers to a main glacier

    This function will merge multiple tributary glaciers to a main glacier
    and write modified `model_flowlines` to the main GlacierDirectory.
    Afterwards only the main GlacierDirectory must be processed and the results
    will cover the main and the tributary glaciers.
    The provided tributaries must have an intersecting downstream line.
    To be sure about this, use `intersect_downstream_lines` first.

    Parameters
    ----------
    main : oggm.GlacierDirectory
        The new GDir of the glacier of interest
    tribs : list or dictionary containing oggm.GlacierDirectories
        true tributary glaciers to the main glacier
    filename: str
        Baseline climate file
    input_filesuffix: str
        Filesuffix to the climate file
    """

    # If its a dict, select the relevant ones
    if isinstance(tribs, dict):
        tribs = tribs[main.rgi_id.split('_merged')[0]]
    # make sure tributaries are iteratable
    tribs = utils.tolist(tribs)

    # Buffer in pixels where to cut the incoming centerlines
    buffer = cfg.PARAMS['kbuffer']
    # Number of pixels to arbitrarily remove at junctions
    lid = int(cfg.PARAMS['flowline_junction_pix'])

    # read flowlines of the Main glacier
    mfls = main.read_pickle('model_flowlines')
    mfl = mfls.pop(-1)  # remove main line from list and treat seperately

    for trib in tribs:

        tfls = trib.read_pickle('model_flowlines')

        # order flowlines in ascending way
        tfls.sort(key=lambda x: x.order, reverse=True)

        # check if flowlines are in correct order
        order = [o.order for i, o in enumerate(tfls)]
        if not (np.all(np.diff(order) <= 0)) & \
               (tfls[0].order == np.max(order)):
            raise RuntimeError('Flowline order is not correct')

        for nr, tfl in enumerate(tfls):

            # 1. Step: Change projection to the main glaciers grid
            _line = salem.transform_geometry(tfl.line,
                                             crs=trib.grid, to_crs=main.grid)

            if nr == 0:
                # cut tributary main line to size
                # find area where lines overlap within a given buffer
                _overlap = _line.intersection(mfl.line.buffer(buffer))
                _line = _line.difference(_overlap)  # cut to new line

                # if the tributary flowline is longer than the main line,
                # _line will contain multiple LineStrings: only keep the first
                try:
                    _line = _line[0]
                except TypeError:
                    pass

                # remove cfg.PARAMS['flowline_junction_pix'] from the _line
                # gives a bigger gap at the junction and makes sure the last
                # point is not corrupted in terms of spacing
                _line = shpg.LineString(_line.coords[:-lid])

            # 2. set new line
            tfl.set_line(_line)

            # 3. set flow to attributes
            if nr == 0:
                # this one flows to the main glacier
                tfl.set_flows_to(mfl)  # set flows_to also changes mfl!
            else:
                # reset to the existing link, neccessary to set attributes
                tfl.set_flows_to(tfl.flows_to)
            # remove inflow points, will be set by other flowlines if need be
            tfl.inflow_points = []

            # 5. set grid size attributes
            dx = [shpg.Point(tfl.line.coords[i]).distance(
                shpg.Point(tfl.line.coords[i+1]))
                for i, pt in enumerate(tfl.line.coords[:-1])]  # get distance
            # and check if equally spaced
            if not np.allclose(dx, np.mean(dx), atol=1e-2):
                raise RuntimeError('Flowline is not evenly spaced.')
            tfl.dx = np.mean(dx).round(2)
            tfl.map_dx = mfl.map_dx
            tfl.dx_meter = tfl.map_dx * tfl.dx

            if nr == 0:
                # change the array size of tributary main flowline attributs
                for atr, value in tfl.__dict__.items():
                    try:
                        if len(value) > tfl.nx:
                            tfl.__setattr__(atr, value[:tfl.nx])
                    except TypeError:
                        pass

            # replace tributary flowline within the list
            tfls[nr] = tfl

        # copy climate file and local_mustar to new gdir
        climfilename = filename + '_' + trib.rgi_id + input_filesuffix + '.nc'
        climfile = os.path.join(main.dir, climfilename)
        shutil.copyfile(trib.get_filepath(filename,
                                          filesuffix=input_filesuffix),
                        climfile)
        _mu = os.path.basename(trib.get_filepath('local_mustar')).split('.')
        mufile = _mu[0] + '_' + trib.rgi_id + '.' + _mu[1]
        shutil.copyfile(trib.get_filepath('local_mustar'),
                        os.path.join(main.dir, mufile))

        mfls = tfls + mfls  # add all tributary flowlines to the main glacier
    mfls = mfls + [mfl]  # add the main glacier flowline back to the list

    # Set the new flowline levels
    for fl in mfls:
        fl.order = line_order(fl)

    # order flowlines in descending way, important for downstream tasks
    mfls.sort(key=lambda x: x.order, reverse=False)

    # Finally write the flowlines
    main.write_pickle(mfls, 'model_flowlines')
