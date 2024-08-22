"""Flowline modelling: bed shapes and model numerics.


"""
# Builtins
import logging
import copy
from collections import OrderedDict
from functools import partial
from time import gmtime, strftime
import os
import shutil
import warnings

# External libs
import numpy as np
import shapely.geometry as shpg
import xarray as xr
from scipy.linalg import solve_banded

# Optional libs
try:
    import salem
except ImportError:
    pass
import pandas as pd

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   MonthlyTIModel,
                                   RandomMassBalance)
from oggm.core.centerlines import Centerline, line_order
from oggm.core.inversion import find_sia_flux_from_thickness

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)


class Flowline(Centerline):
    """A Centerline with additional properties: input to the FlowlineModel
    """

    def __init__(self, line=None, dx=1, map_dx=None,
                 surface_h=None, bed_h=None, rgi_id=None,
                 water_level=None, gdir=None):
        """ Initialize a Flowline

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        dx : float
            Grid spacing in pixel coordinates
        map_dx : float
            DEM grid spacing in meters
        surface_h: :py:class:`numpy.ndarray`
            elevation [m] of the flowline grid points
        bed_h: :py:class:`numpy.ndarray`
            elevation[m] of the bedrock at the flowline grid points
        rgi_id : str
            The glacier's RGI identifier
        water_level : float
            The water level (to compute volume below sea-level)
        """

        # This is do add flexibility for testing
        if dx is None:
            dx = 1.
        if line is None:
            coords = np.arange(len(surface_h)) * dx
            line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        super(Flowline, self).__init__(line, dx, surface_h)

        self._thick = utils.clip_min(surface_h - bed_h, 0.)
        self.map_dx = map_dx
        self.dx_meter = map_dx * self.dx
        self.bed_h = bed_h
        self.rgi_id = rgi_id
        self.water_level = water_level
        self._point_lons = None
        self._point_lats = None
        self.map_trafo = None
        if gdir is not None:
            self.map_trafo = partial(gdir.grid.ij_to_crs, crs=salem.wgs84)
        # volume not yet removed from the flowline
        self.calving_bucket_m3 = 0

    def has_ice(self):
        return np.any(self.thick > 0)

    @utils.lazy_property
    def water_depth(self):
        return utils.clip_min(self.water_level - self.bed_h, 0)

    @utils.lazy_property
    def bed_below_wl(self):
        return self.bed_h < self.water_level

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
        self._thick = utils.clip_min(value, 0)

    @Centerline.surface_h.getter
    def surface_h(self):
        return self._thick + self.bed_h

    @surface_h.setter
    def surface_h(self, value):
        self.thick = value - self.bed_h

    @property
    def bin_area_m2(self):
        # area of the grid point
        # this takes the ice thickness into account
        return np.where(self.thick > 0, self.widths_m, 0) * self.dx_meter

    @property
    def length_m(self):
        # TODO: take calving bucket into account for fine tuned length?
        lt = cfg.PARAMS.get('min_ice_thick_for_length', 0)
        if cfg.PARAMS.get('glacier_length_method') == 'consecutive':
            if (self.thick > lt).all():
                nx = len(self.thick)
            else:
                nx = np.where(self.thick <= lt)[0][0]
        else:
            nx = len(np.where(self.thick > lt)[0])
        return nx * self.dx_meter

    @property
    def terminus_index(self):
        # the index of the last point with ice thickness above
        # min_ice_thick_for_length and consistent with length
        lt = cfg.PARAMS.get('min_ice_thick_for_length', 0)
        if cfg.PARAMS.get('glacier_length_method') == 'consecutive':
            if (self.thick > lt).all():
                ix = len(self.thick) - 1
            else:
                ix = np.where(self.thick <= lt)[0][0] - 1
        else:
            try:
                ix = np.where(self.thick > lt)[0][-1]
            except IndexError:
                ix = -1
        return ix

    def _compute_point_lls(self):
        if getattr(self, '_point_lons', None) is None:
            if getattr(self, 'map_trafo', None) is None:
                raise AttributeError('Cannot compute lons and lats on this '
                                     'flowline. It needs to be initialized '
                                     'with a gdir kwarg.')
            lons, lats = self.map_trafo(*self.line.xy)
            self._point_lons = lons
            self._point_lats = lats

    @property
    def point_lons(self):
        self._compute_point_lls()
        return self._point_lons

    @property
    def point_lats(self):
        self._compute_point_lls()
        return self._point_lats

    @property
    def volume_m3(self):
        return utils.clip_min(np.sum(self.section * self.dx_meter) -
                              getattr(self, 'calving_bucket_m3', 0), 0)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    def _vol_below_level(self, water_level=0):

        thick = np.copy(self.thick)
        n_thick = np.copy(thick)
        bwl = (self.bed_h < water_level) & (thick > 0)
        n_thick[~bwl] = 0
        self.thick = n_thick
        vol_tot = np.sum(self.section * self.dx_meter)
        n_thick[bwl] = utils.clip_max(self.surface_h[bwl],
                                      water_level) - self.bed_h[bwl]
        self.thick = n_thick
        vol_bwl = np.sum(self.section * self.dx_meter)
        self.thick = thick
        fac = vol_bwl / vol_tot if vol_tot > 0 else 0
        return utils.clip_min(vol_bwl -
                              getattr(self, 'calving_bucket_m3', 0) * fac, 0)

    @property
    def volume_bsl_m3(self):
        return self._vol_below_level(water_level=0)

    @property
    def volume_bsl_km3(self):
        return self.volume_bsl_m3 * 1e-9

    @property
    def volume_bwl_m3(self):
        return self._vol_below_level(water_level=self.water_level)

    @property
    def volume_bwl_km3(self):
        return self.volume_bwl_m3 * 1e-9

    @property
    def area_m2(self):
        # TODO: take calving bucket into account
        return np.sum(self.bin_area_m2)

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        # This must be done by child classes
        raise NotImplementedError()

    def to_geometry_dataset(self):
        """Makes an xarray Dataset out of the flowline.

        Useful only for geometry files (FileModel / restart files),
        therefore a bit cryptic regarding dimensions.
        """

        h = self.surface_h
        nx = len(h)
        ds = xr.Dataset()
        ds.coords['x'] = np.arange(nx)
        ds.coords['c'] = [0, 1]
        try:
            ds['linecoords'] = (['x', 'c'], np.asarray(self.line.coords))
        except AttributeError:
            # squeezed lines
            pass
        ds['surface_h'] = (['x'], h)
        ds['bed_h'] = (['x'], self.bed_h)
        ds.attrs['class'] = type(self).__name__
        ds.attrs['map_dx'] = self.map_dx
        ds.attrs['dx'] = self.dx
        self._add_attrs_to_dataset(ds)
        return ds

    def to_diagnostics_dataset(self):
        """Makes an xarray Dataset out of the flowline.

        Useful for run_until_and_store's flowline diagnostics data.
        """

        h = self.bed_h
        nx = len(h)
        ds = xr.Dataset()
        ds.coords['dis_along_flowline'] = np.arange(nx) * self.map_dx * self.dx
        try:
            # This is a bit bad design, but basically if some task
            # computed to the lons of the flowlines before, use it
            # we don't have access to gdir here so we cant convert the coords
            ds['point_lons'] = (['dis_along_flowline'], self.point_lons)
            ds['point_lons'].attrs['description'] = 'Longitude along the flowline'
            ds['point_lons'].attrs['unit'] = 'deg'
            ds['point_lats'] = (['dis_along_flowline'], self.point_lats)
            ds['point_lats'].attrs['description'] = 'Latitude along the flowline'
            ds['point_lats'].attrs['unit'] = 'deg'
        except AttributeError:
            # squeezed lines or we haven't computed lons and lats yet
            pass
        ds['bed_h'] = (['dis_along_flowline'], h)
        ds['bed_h'].attrs['description'] = 'Bed elevation along the flowline'
        ds['bed_h'].attrs['unit'] = 'm'
        ds.attrs['class'] = type(self).__name__
        ds.attrs['map_dx'] = self.map_dx
        ds.attrs['dx'] = self.dx
        return ds


class ParabolicBedFlowline(Flowline):
    """A parabolic shaped Flowline with one degree of freedom
    """

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, bed_shape=None, rgi_id=None,
                 water_level=None, gdir=None):
        """ Instantiate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        """
        super(ParabolicBedFlowline, self).__init__(line, dx, map_dx,
                                                   surface_h, bed_h,
                                                   rgi_id=rgi_id,
                                                   water_level=water_level,
                                                   gdir=gdir)

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

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('parabolic', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['bed_shape'] = (['x'], self.bed_shape)


class RectangularBedFlowline(Flowline):
    """Simple shaped Flowline, glacier width does not change with ice thickness

    """

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, widths=None, rgi_id=None,
                 water_level=None, gdir=None):
        """ Instantiate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`

        """
        super(RectangularBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id,
                                                     water_level=water_level,
                                                     gdir=gdir)

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

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('rectangular', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'], self._widths)


class TrapezoidalBedFlowline(Flowline):
    """A Flowline with trapezoidal shape and two degrees of freedom
    """

    def __init__(self, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, widths=None, lambdas=None, rgi_id=None,
                 water_level=None, gdir=None):
        """ Instantiate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        """
        super(TrapezoidalBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id,
                                                     water_level=water_level,
                                                     gdir=gdir)

        self._w0_m = widths * self.map_dx - lambdas * self.thick

        if np.any(self._w0_m <= 0):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        self._prec = lambdas == 0
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

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('trapezoid', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'], self.widths)
        ds['lambdas'] = (['x'], self._lambdas)


class MixedBedFlowline(Flowline):
    """A Flowline which can take a combination of different shapes (default)

    The default shape is parabolic. At ice divides a rectangular shape is used.
    And if the parabola gets too flat a trapezoidal shape is used.
    """

    def __init__(self, *, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, section=None, bed_shape=None,
                 is_trapezoid=None, lambdas=None, widths_m=None, rgi_id=None,
                 water_level=None, gdir=None):
        """ Instantiate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        """

        super(MixedBedFlowline, self).__init__(line=line, dx=dx, map_dx=map_dx,
                                               surface_h=surface_h.copy(),
                                               bed_h=bed_h.copy(),
                                               rgi_id=rgi_id,
                                               water_level=water_level,
                                               gdir=gdir)

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
        self.bed_shape[is_trapezoid] = np.nan
        self._lambdas[~is_trapezoid] = np.nan

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

        self._w0_m[~is_trapezoid] = np.nan

        if (np.any(self._w0_m[self._ptrap] <= 0) or
                np.any(~np.isfinite(self._w0_m[self._ptrap]))):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        assert np.all(self.bed_shape[~is_trapezoid] > 0)

        self._prec = is_trapezoid & (lambdas == 0)

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

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        out = np.repeat('rectangular', self.nx)
        out[~ self.is_trapezoid] = 'parabolic'
        out[self.is_trapezoid & ~ self.is_rectangular] = 'trapezoid'
        return out

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""

        ds['section'] = (['x'], self.section)
        ds['bed_shape'] = (['x'], self.bed_shape)
        ds['is_trapezoid'] = (['x'], self.is_trapezoid)
        ds['widths_m'] = (['x'], self._w0_m)
        ds['lambdas'] = (['x'], self._lambdas)


class FlowlineModel(object):
    """Interface to OGGM's flowline models"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=None, inplace=False, smooth_trib_influx=True,
                 is_tidewater=False, is_lake_terminating=False,
                 mb_elev_feedback='annual', check_for_boundaries=None,
                 water_level=None, required_model_steps='monthly'):
        """Create a new flowline model from the flowlines and a MB model.

        Parameters
        ----------
        flowlines : list
            a list of :py:class:`oggm.Flowline` instances, sorted by order
        mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
            the MB model to use
        y0 : int
            the starting year of the simulation
        glen_a : float
            glen's parameter A
        fs: float
            sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        smooth_trib_influx : bool
            whether to smooth the mass influx from the incoming tributary.
            The default is to use a gaussian kernel on a 9 grid points
            window.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries : bool
            whether the model should raise an error when the glacier exceeds
            the domain boundaries. The default is to follow
            PARAMS['error_when_glacier_reaches_boundaries']
        required_model_steps : str
            some Flowline models have an adaptive time stepping scheme, which
            is randomly taking steps towards the goal of a "run_until". The
            default ('monthly') makes sure that the model results are
            consistent whether the users want data at monthly or annual
            timesteps by forcing the model to land on monthly steps even if
            only annual updates are required. You may want to change this
            for optimisation reasons for models that don't require adaptive
            steps (for example the deltaH method).
        """

        self.is_tidewater = is_tidewater
        self.is_lake_terminating = is_lake_terminating
        self.is_marine_terminating = is_tidewater and not is_lake_terminating
        self.water_level = 0 if water_level is None else water_level

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback.lower()
        if self.mb_elev_feedback in ['never', 'annual']:
            self.mb_step = 'annual'
        elif self.mb_elev_feedback in ['always', 'monthly']:
            self.mb_step = 'monthly'
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
        if check_for_boundaries is None:
            check_for_boundaries = cfg.PARAMS[('error_when_glacier_reaches_'
                                               'boundaries')]
        self.check_for_boundaries = check_for_boundaries

        # we keep glen_a as input, but for optimisation we stick to "fd"
        self._fd = 2. / (cfg.PARAMS['glen_n']+2) * self.glen_a

        # Calving shenanigans
        self.calving_m3_since_y0 = 0.  # total calving since time y0
        self.calving_rate_myr = 0.

        # Time
        if required_model_steps not in ['annual', 'monthly']:
            raise InvalidParamsError('required_model_steps needs to be of '
                                     '`annual` or `monthly`.')
        self.required_model_steps = required_model_steps
        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.fls = None
        self._tributary_indices = None
        self.reset_flowlines(flowlines, inplace=inplace,
                             smooth_trib_influx=smooth_trib_influx)

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

    def reset_flowlines(self, flowlines, inplace=False,
                        smooth_trib_influx=True):
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
            # Important also
            fl.water_level = self.water_level
            if fl.flows_to is None:
                trib_ind.append((None, None, None, None))
                continue
            idl = self.fls.index(fl.flows_to)
            ide = fl.flows_to_indice
            if not smooth_trib_influx:
                gk = 1
                id0 = ide
                id1 = ide+1
            elif fl.flows_to.nx >= 9:
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

        self._tributary_indices = trib_ind

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
    def volume_bsl_m3(self):
        return np.sum([f.volume_bsl_m3 for f in self.fls])

    @property
    def volume_bsl_km3(self):
        return self.volume_bsl_m3 * 1e-9

    @property
    def volume_bwl_m3(self):
        return np.sum([f.volume_bwl_m3 for f in self.fls])

    @property
    def volume_bwl_km3(self):
        return self.volume_bwl_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def length_m(self):
        return self.fls[-1].length_m

    def get_mb(self, heights, year=None, fl_id=None, fls=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        # Do we even have to optimise?
        if self.mb_elev_feedback == 'always':
            return self._mb_call(heights, year=year, fl_id=fl_id, fls=fls)

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
                self._mb_current_out[fl_id] = self._mb_call(heights,
                                                            year=year,
                                                            fl_id=fl_id,
                                                            fls=fls)
        else:
            # We need to reset all
            self._mb_current_date = date
            self._mb_current_out = dict()
            self._mb_current_out[fl_id] = self._mb_call(heights,
                                                        year=year,
                                                        fl_id=fl_id,
                                                        fls=fls)

        return self._mb_current_out[fl_id]

    def to_geometry_netcdf(self, path):
        """Creates a netcdf group file storing the state of the model."""

        flows_to_id = []
        for trib in self._tributary_indices:
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
                ds = fl.to_geometry_dataset()
                ds.to_netcdf(path, 'a', group='fl_{}'.format(i))
        finally:
            ds.close()

    def check_domain_end(self):
        """Returns False if the glacier reaches the domains bound."""
        return np.isclose(self.fls[-1].thick[-1], 0)

    def step(self, dt):
        """Advance the numerical simulation of one single step.

        Important: the step dt is a maximum boundary that is *not* guaranteed
        to be met if dt is too large for the underlying numerical
        implementation. However, ``step(dt)`` should never cross the desired
        time step, i.e. if dt is small enough to ensure stability, step
        should match it.

        The caller will know how much has been actually advanced by looking
        at the output of ``step()`` or by monitoring ``self.t`` or `self.yr``

        Parameters
        ----------
        dt : float
             the step length in seconds

        Returns
        -------
        the actual dt chosen by the numerical implementation. Guaranteed to
        be dt or lower.
        """
        raise NotImplementedError

    def run_until(self, y1):
        """Runs the model from the current year up to a given year date y1.

        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.

        Parameters
        ----------
        y1 : float
            Upper time span for how long the model should run
        """

        if self.required_model_steps == 'monthly':
            # We force timesteps to monthly frequencies for consistent results
            # among use cases (monthly or yearly output) and also to prevent
            # "too large" steps in the adaptive scheme.
            ts = utils.monthly_timeseries(self.yr, y1)
            # Add the last date to be sure we end on it - implementations
            # of `step()` and of the loop below should not run twice anyways
            ts = np.append(ts, y1)
        else:
            ts = np.arange(int(self.yr), int(y1+1))

        # Loop over the steps we want to meet
        for y in ts:
            t = (y - self.y0) * SEC_IN_YEAR
            # because of CFL, step() doesn't ensure that the end date is met
            # lets run the steps until we reach our desired date
            while self.t < t:
                self.step(t - self.t)

            # Check for domain bounds
            if self.check_for_boundaries:
                if self.fls[-1].thick[-1] > 10:
                    raise RuntimeError('Glacier exceeds domain boundaries, '
                                       'at year: {}'.format(self.yr))

            # Check for nans
            for fl in self.fls:
                if np.any(~np.isfinite(fl.thick)):
                    raise FloatingPointError('nan in numerical solution, '
                                             'at year: {}'.format(self.yr))

    def run_until_and_store(self, y1,
                            diag_path=None,
                            fl_diag_path=False,
                            geom_path=False,
                            store_monthly_step=None,
                            stop_criterion=None,
                            fixed_geometry_spinup_yr=None,
                            dynamic_spinup_min_ice_thick=None,
                            ):
        """Runs the model and returns intermediate steps in xarray datasets.

        This function repeatedly calls FlowlineModel.run_until for either
        monthly or yearly time steps up till the upper time boundary y1.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run (needs to be
            a full year)
        diag_path : str
            Path and filename where to store the glacier-wide diagnostics
            dataset (length, area, volume, etc.) as controlled by
            cfg.PARAMS['store_diagnostic_variables'].
            The default (None) is to not store the dataset to disk but return
            the dataset to the user after execution.
        fl_diag_path : str, None or bool
            Path and filename where to store the model diagnostics along the
            flowline(s).
        geom_path : str, None or bool
            Path and filename where to store the model geometry dataset. This
            dataset contains all necessary info to retrieve the full glacier
            geometry after the run,  with a FileModel. This is stored
            on an annual basis and can be used to restart a run from a
            past simulation's geometry ("restart file").
            The default (False) prevents creating this dataset altogether
            (for optimisation purposes).
            Set this to None to not store the dataset to disk but return
            the dataset to the user after execution.
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly).
            If unspecified, we follow the update of the MB model, which
            defaults to yearly (see __init__).
        stop_criterion : func
            a function evaluating the model state (and possibly evolution over
            time), and deciding when to stop the simulation. Its signature
            should look like:

            stop, new_state = stop_criterion(model, previous_state)

            where stop is a bool, and new_state a container (likely: dict)
            initialized by the function itself on the first call (previous_state
            can and should be None on the first call). See
            `zero_glacier_stop_criterion` for an example.
        fixed_geometry_spinup_yr : int
            if set to an integer, the model will artificially prolongate
            all outputs of run_until_and_store to encompass all time stamps
            starting from the chosen year. The only output affected are the
            glacier wide diagnostic files - all other outputs are set
            to constants during "spinup"
        dynamic_spinup_min_ice_thick : float or None
            if set to a float, additional variables are saved which are useful
            in combination with the dynamic spinup. In particular only grid
            points with a minimum ice thickness are considered for the total
            area or the total volume. This is useful to smooth out yearly
            fluctuations when matching to observations. The names of this new
            variables include the suffix _min_h (e.g. 'area_m2_min_h')

        Returns
        -------
        geom_ds : xarray.Dataset or None
            stores the entire glacier geometry. It is useful to visualize the
            glacier geometry or to restart a new run from a modelled geometry.
            The glacier state is stored at the beginning of each hydrological
            year (not in between in order to spare disk space).
        diag_ds : xarray.Dataset
            stores a few diagnostic variables such as the volume, area, length
            and ELA of the glacier.
        """

        if int(y1) != y1:
            raise InvalidParamsError('run_until_and_store only accepts '
                                     'integer year dates.')

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass balance model with an unambiguous '
                                     'hemisphere.')

        # Do we have a spinup?
        do_fixed_spinup = fixed_geometry_spinup_yr is not None
        y0 = fixed_geometry_spinup_yr if do_fixed_spinup else self.yr

        # Do we need to create a geometry or flowline diagnostics dataset?
        do_geom = geom_path is None or geom_path
        do_fl_diag = fl_diag_path is None or fl_diag_path

        # time
        yearly_time = np.arange(np.floor(y0), np.floor(y1)+1)

        if store_monthly_step is None:
            store_monthly_step = self.mb_step == 'monthly'

        if store_monthly_step:
            monthly_time = utils.monthly_timeseries(y0, y1)
        else:
            monthly_time = np.arange(np.floor(y0), np.floor(y1)+1)

        yrs, months = utils.floatyear_to_date(monthly_time)
        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]
        hyrs, hmonths = utils.calendardate_to_hydrodate(yrs, months,
                                                        start_month=sm)

        # init output
        if geom_path:
            self.to_geometry_netcdf(geom_path)

        ny = len(yearly_time)
        if ny == 1:
            yrs = [yrs]
            hyrs = [hyrs]
            months = [months]
            hmonths = [hmonths]
        nm = len(monthly_time)

        if do_geom or do_fl_diag:
            sects = [(np.zeros((ny, fl.nx)) * np.nan) for fl in self.fls]
            widths = [(np.zeros((ny, fl.nx)) * np.nan) for fl in self.fls]
            buckets = [np.zeros(ny) for _ in self.fls]

        # Diagnostics dataset
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'OGGM model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['water_level'] = self.water_level
        diag_ds.attrs['glen_a'] = self.glen_a
        diag_ds.attrs['fs'] = self.fs

        # Add MB model attributes
        diag_ds.attrs['mb_model_class'] = self.mb_model.__class__.__name__
        for k, v in self.mb_model.__dict__.items():
            if np.isscalar(v) and not k.startswith('_'):
                diag_ds.attrs['mb_model_{}'.format(k)] = v

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['calendar_year'] = ('time', yrs)
        diag_ds.coords['calendar_month'] = ('time', months)
        diag_ds.coords['hydro_year'] = ('time', hyrs)
        diag_ds.coords['hydro_month'] = ('time', hmonths)

        diag_ds['time'].attrs['description'] = 'Floating year'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'

        # Variables and attributes
        ovars = cfg.PARAMS['store_diagnostic_variables']

        if 'volume' in ovars:
            diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
            diag_ds['volume_m3'].attrs['unit'] = 'm 3'

        if 'volume_bsl' in ovars:
            diag_ds['volume_bsl_m3'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['volume_bsl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'sea-level')
            diag_ds['volume_bsl_m3'].attrs['unit'] = 'm 3'

        if 'volume_bwl' in ovars:
            diag_ds['volume_bwl_m3'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['volume_bwl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'water-level')
            diag_ds['volume_bwl_m3'].attrs['unit'] = 'm 3'

        if 'area' in ovars:
            diag_ds['area_m2'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
            diag_ds['area_m2'].attrs['unit'] = 'm 2'

        if dynamic_spinup_min_ice_thick is None:
            dynamic_spinup_min_ice_thick = cfg.PARAMS['dynamic_spinup_min_ice_thick']

        if 'area_min_h' in ovars:
            # filled with a value if dynamic_spinup_min_ice_thick is not None
            diag_ds['area_m2_min_h'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['area_m2_min_h'].attrs['description'] = \
                f'Total glacier area of gridpoints with a minimum ice' \
                f'thickness of {dynamic_spinup_min_ice_thick} m'
            diag_ds['area_m2_min_h'].attrs['unit'] = 'm 2'

        if 'length' in ovars:
            diag_ds['length_m'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['length_m'].attrs['description'] = 'Glacier length'
            diag_ds['length_m'].attrs['unit'] = 'm'

        if 'calving' in ovars:
            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
                                                          'calving flux')
            diag_ds['calving_m3'].attrs['unit'] = 'm 3'

        if 'calving_rate' in ovars:
            diag_ds['calving_rate_myr'] = ('time', np.zeros(nm) * np.nan)
            diag_ds['calving_rate_myr'].attrs['description'] = 'Calving rate'
            diag_ds['calving_rate_myr'].attrs['unit'] = 'm yr-1'

        for gi in range(10):
            vn = f'terminus_thick_{gi}'
            if vn in ovars:
                diag_ds[vn] = ('time', np.zeros(nm) * np.nan)
                diag_ds[vn].attrs['description'] = ('Thickness of grid point '
                                                    f'{gi} from terminus.')
                diag_ds[vn].attrs['unit'] = 'm'

        if do_fixed_spinup:
            is_spinup_time = monthly_time < self.yr
            diag_ds['is_fixed_geometry_spinup'] = ('time', is_spinup_time)
            desc = 'Part of the series which are spinup'
            diag_ds['is_fixed_geometry_spinup'].attrs['description'] = desc
            diag_ds['is_fixed_geometry_spinup'].attrs['unit'] = '-'

        fl_diag_dss = None
        if do_fl_diag:
            # Time invariant datasets
            fl_diag_dss = [fl.to_diagnostics_dataset() for fl in self.fls]

            # Global attributes
            for ds in fl_diag_dss:
                ds.attrs['description'] = 'OGGM model output'
                ds.attrs['oggm_version'] = __version__
                ds.attrs['calendar'] = '365-day no leap'
                ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                ds.attrs['water_level'] = self.water_level
                ds.attrs['glen_a'] = self.glen_a
                ds.attrs['fs'] = self.fs

                # Add MB model attributes
                ds.attrs['mb_model_class'] = self.mb_model.__class__.__name__
                for k, v in self.mb_model.__dict__.items():
                    if np.isscalar(v) and not k.startswith('_'):
                        if type(v) is bool:
                            v = str(v)
                        ds.attrs['mb_model_{}'.format(k)] = v

                # Coordinates
                ds.coords['time'] = yearly_time
                ds['time'].attrs['description'] = 'Floating hydrological year'

            # Variables and attributes
            ovars_fl = cfg.PARAMS['store_fl_diagnostic_variables']
            if 'volume' not in ovars_fl or 'area' not in ovars_fl:
                raise InvalidParamsError('Flowline diagnostics need at least '
                                         'volume and area as output.')

            for ds, sect, width, bucket in zip(fl_diag_dss, sects, widths, buckets):
                if 'volume' in ovars_fl:
                    ds['volume_m3'] = (('time', 'dis_along_flowline'), sect)
                    ds['volume_m3'].attrs['description'] = 'Section volume'
                    ds['volume_m3'].attrs['unit'] = 'm 3'
                if 'volume_bsl' in ovars_fl:
                    ds['volume_bsl_m3'] = (('time', 'dis_along_flowline'), sect * 0)
                    ds['volume_bsl_m3'].attrs['description'] = 'Section volume below sea level'
                    ds['volume_bsl_m3'].attrs['unit'] = 'm 3'
                if 'volume_bwl' in ovars_fl:
                    ds['volume_bwl_m3'] = (('time', 'dis_along_flowline'), sect * 0)
                    ds['volume_bwl_m3'].attrs['description'] = 'Section volume below water level'
                    ds['volume_bwl_m3'].attrs['unit'] = 'm 3'
                if 'area' in ovars_fl:
                    ds['area_m2'] = (('time', 'dis_along_flowline'), width)
                    ds['area_m2'].attrs['description'] = 'Section area'
                    ds['area_m2'].attrs['unit'] = 'm 2'
                if 'thickness' in ovars_fl:
                    ds['thickness_m'] = (('time', 'dis_along_flowline'), width * np.nan)
                    ds['thickness_m'].attrs['description'] = 'Section thickness'
                    ds['thickness_m'].attrs['unit'] = 'm'
                if 'ice_velocity' in ovars_fl:
                    if not (hasattr(self, '_surf_vel_fac') or hasattr(self, 'u_stag')):
                        raise InvalidParamsError('This flowline model does not seem '
                                                 'to be able to provide surface '
                                                 'velocities.')
                    ds['ice_velocity_myr'] = (('time', 'dis_along_flowline'), width * np.nan)
                    ds['ice_velocity_myr'].attrs['description'] = 'Ice velocity at the surface'
                    ds['ice_velocity_myr'].attrs['unit'] = 'm yr-1'
                if 'calving_bucket' in ovars_fl:
                    ds['calving_bucket_m3'] = (('time',), bucket)
                    desc = 'Flowline calving bucket (volume not yet calved)'
                    ds['calving_bucket_m3'].attrs['description'] = desc
                    ds['calving_bucket_m3'].attrs['unit'] = 'm 3'
                if do_fixed_spinup:
                    ds['is_fixed_geometry_spinup'] = ('time', is_spinup_time)
                    desc = 'Part of the series which are spinup'
                    ds['is_fixed_geometry_spinup'].attrs['description'] = desc
                    ds['is_fixed_geometry_spinup'].attrs['unit'] = '-'
                if 'flux_divergence' in ovars_fl:
                    # We need dhdt and climatic_mb to calculate divergence
                    if 'dhdt' not in ovars_fl:
                        ovars_fl.append('dhdt')
                    if 'climatic_mb' not in ovars_fl:
                        ovars_fl.append('climatic_mb')

                    ds['flux_divergence_myr'] = (('time', 'dis_along_flowline'),
                                                 width * np.nan)
                    desc = 'Ice-flux divergence'
                    ds['flux_divergence_myr'].attrs['description'] = desc
                    ds['flux_divergence_myr'].attrs['unit'] = 'm yr-1'
                if 'climatic_mb' in ovars_fl:
                    # We need dhdt to define where to calculate climatic_mb
                    if 'dhdt' not in ovars_fl:
                        ovars_fl.append('dhdt')

                    ds['climatic_mb_myr'] = (('time', 'dis_along_flowline'),
                                             width * np.nan)
                    desc = 'Climatic mass balance forcing'
                    ds['climatic_mb_myr'].attrs['description'] = desc
                    ds['climatic_mb_myr'].attrs['unit'] = 'm yr-1'

                    # needed for the calculation of mb at start of period
                    surface_h_previous = {}
                    for fl_id, fl in enumerate(self.fls):
                        surface_h_previous[fl_id] = fl.surface_h
                if 'dhdt' in ovars_fl:
                    ds['dhdt_myr'] = (('time', 'dis_along_flowline'),
                                      width * np.nan)
                    desc = 'Thickness change'
                    ds['dhdt_myr'].attrs['description'] = desc
                    ds['dhdt_myr'].attrs['unit'] = 'm yr-1'

                    # needed for the calculation of dhdt
                    thickness_previous_dhdt = {}
                    for fl_id, fl in enumerate(self.fls):
                        thickness_previous_dhdt[fl_id] = fl.thick

        # First deal with spinup (we compute volume change only)
        if do_fixed_spinup:
            spinup_vol = monthly_time * 0
            for fl_id, fl in enumerate(self.fls):

                h = fl.surface_h
                a = fl.widths_m * fl.dx_meter
                a[fl.section <= 0] = 0

                for j, yr in enumerate(monthly_time[is_spinup_time]):
                    smb = self.get_mb(h, year=yr, fl_id=fl_id, fls=self.fls)
                    spinup_vol[j] -= np.sum(smb * a)  # per second and minus because backwards

            # per unit time
            dt = (monthly_time[1:] - monthly_time[:-1]) * cfg.SEC_IN_YEAR
            spinup_vol[:-1] = spinup_vol[:-1] * dt
            spinup_vol = np.cumsum(spinup_vol[::-1])[::-1]

        # Run
        j = 0
        prev_state = None  # for the stopping criterion
        for i, (yr, mo) in enumerate(zip(monthly_time, months)):

            if yr > self.yr:
                # Here we model run - otherwise (for spinup) we
                # constantly store the same data
                self.run_until(yr)

            # Glacier geometry
            if (do_geom or do_fl_diag) and mo == 1:
                for s, w, b, fl in zip(sects, widths, buckets, self.fls):
                    s[j, :] = fl.section
                    w[j, :] = fl.widths_m
                    if self.is_tidewater:
                        try:
                            b[j] = fl.calving_bucket_m3
                        except AttributeError:
                            pass

                # Flowline diagnostics
                if do_fl_diag:
                    for fl_id, (ds, fl) in enumerate(zip(fl_diag_dss, self.fls)):
                        # area and volume are already being taken care of above
                        if 'thickness' in ovars_fl:
                            ds['thickness_m'].data[j, :] = fl.thick
                        if 'volume_bsl' in ovars_fl:
                            ds['volume_bsl_m3'].data[j, :] = fl.volume_bsl_m3
                        if 'volume_bwl' in ovars_fl:
                            ds['volume_bwl_m3'].data[j, :] = fl.volume_bwl_m3
                        if 'ice_velocity' in ovars_fl and (yr > self.y0):
                            # Velocity can only be computed with dynamics
                            var = self.u_stag[fl_id]
                            val = (var[1:fl.nx + 1] + var[:fl.nx]) / 2 * self._surf_vel_fac
                            ds['ice_velocity_myr'].data[j, :] = val * cfg.SEC_IN_YEAR
                        if 'dhdt' in ovars_fl and (yr > self.y0):
                            # dhdt can only be computed after one year
                            val = fl.thick - thickness_previous_dhdt[fl_id]
                            ds['dhdt_myr'].data[j, :] = val
                            thickness_previous_dhdt[fl_id] = fl.thick
                        if 'climatic_mb' in ovars_fl and (yr > self.y0):
                            # yr - 1 to use the mb which lead to the current
                            # state, also using previous surface height
                            val = self.get_mb(surface_h_previous[fl_id],
                                              self.yr - 1,
                                              fl_id=fl_id)
                            # only save climatic mb where dhdt is non zero,
                            # isclose for avoiding numeric represention artefacts
                            dhdt_zero = np.isclose(ds['dhdt_myr'].data[j, :],
                                                   0.)
                            ds['climatic_mb_myr'].data[j, :] = np.where(
                                dhdt_zero,
                                0.,
                                val * cfg.SEC_IN_YEAR)
                            surface_h_previous[fl_id] = fl.surface_h
                        if 'flux_divergence' in ovars_fl and (yr > self.y0):
                            # calculated after the formula dhdt = mb + flux_div
                            val = ds['dhdt_myr'].data[j, :] - ds['climatic_mb_myr'].data[j, :]
                            # special treatment for retreating: If the glacier
                            # terminus is getting ice free it means the
                            # climatic mass balance is more negative than the
                            # available ice to melt. In this case we can not
                            # calculate the flux divergence with this method
                            # exactly, we just can say it was smaller to
                            # compensate the very negative climatic mb.
                            # To avoid large spikes at the terminus in the
                            # calculated flux divergence we smooth this values
                            # with an (arbitrary) factor of 0.1.
                            has_become_ice_free = np.logical_and(
                                np.isclose(fl.thick, 0.),
                                ds['dhdt_myr'].data[j, :] < 0.
                            )
                            fac = np.where(has_become_ice_free, 0.1, 1.)
                            ds['flux_divergence_myr'].data[j, :] = val * fac

                # j is the yearly index in case we have monthly output
                # we have to count it ourselves
                j += 1

            # Diagnostics
            if 'volume' in ovars:
                diag_ds['volume_m3'].data[i] = self.volume_m3
            if 'area' in ovars:
                diag_ds['area_m2'].data[i] = self.area_m2
            if 'length' in ovars:
                diag_ds['length_m'].data[i] = self.length_m
            if 'calving' in ovars:
                diag_ds['calving_m3'].data[i] = self.calving_m3_since_y0
            if 'calving_rate' in ovars:
                diag_ds['calving_rate_myr'].data[i] = self.calving_rate_myr
            if 'volume_bsl' in ovars:
                diag_ds['volume_bsl_m3'].data[i] = self.volume_bsl_m3
            if 'volume_bwl' in ovars:
                diag_ds['volume_bwl_m3'].data[i] = self.volume_bwl_m3
            if 'area_min_h' in ovars:
                diag_ds['area_m2_min_h'].data[i] = np.sum([np.sum(
                    fl.bin_area_m2[fl.thick > dynamic_spinup_min_ice_thick])
                    for fl in self.fls])
            # Terminus thick is a bit more logic
            ti = None
            for gi in range(10):
                vn = f'terminus_thick_{gi}'
                if vn in ovars:
                    if ti is None:
                        ti = self.fls[-1].terminus_index
                    diag_ds[vn].data[i] = self.fls[-1].thick[ti - gi]

            # Decide if we continue
            if stop_criterion is not None:
                stop, prev_state = stop_criterion(self, prev_state)
                if stop:
                    break

        # to datasets
        geom_ds = None
        if do_geom:
            geom_ds = []
            for (s, w, b) in zip(sects, widths, buckets):
                ds = xr.Dataset()
                ds.attrs['description'] = 'OGGM model output'
                ds.attrs['oggm_version'] = __version__
                ds.attrs['calendar'] = '365-day no leap'
                ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                     gmtime())
                ds.attrs['water_level'] = self.water_level
                ds.attrs['glen_a'] = self.glen_a
                ds.attrs['fs'] = self.fs
                # Add MB model attributes
                ds.attrs['mb_model_class'] = self.mb_model.__class__.__name__
                for k, v in self.mb_model.__dict__.items():
                    if np.isscalar(v) and not k.startswith('_'):
                        ds.attrs['mb_model_{}'.format(k)] = v

                ds.coords['time'] = yearly_time
                ds['time'].attrs['description'] = 'Floating hydrological year'
                varcoords = OrderedDict(time=('time', yearly_time),
                                        year=('time', yearly_time))
                ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                                coords=varcoords)
                ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                                coords=varcoords)
                ds['ts_calving_bucket_m3'] = xr.DataArray(b, dims=('time', ),
                                                          coords=varcoords)

                if stop_criterion is not None:
                    # Remove probable nans
                    ds = ds.dropna('time', subset=['ts_section'])

                geom_ds.append(ds)

        # Add the spinup volume to the diag
        if do_fixed_spinup:
            # If there is calving we need to trick as well
            if 'calving_m3' in diag_ds and np.any(diag_ds['calving_m3'] > 0):
                raise NotImplementedError('Calving and fixed_geometry_spinup_yr '
                                          'not implemented yet.')
            diag_ds['volume_m3'].data[:] += spinup_vol

        if stop_criterion is not None:
            # Remove probable nans
            diag_ds = diag_ds.dropna('time', subset=['volume_m3'])

        # write output?
        if do_fl_diag:
            # Unit conversions for these
            for i, ds in enumerate(fl_diag_dss):
                dx = ds.attrs['map_dx'] * ds.attrs['dx']
                # No inplace because the other dataset uses them
                # These variables are always there (see above)
                ds['volume_m3'] = ds['volume_m3'] * dx
                ds['area_m2'] = ds['area_m2'].where(ds['volume_m3'] > 0, 0) * dx
                if stop_criterion is not None:
                    # Remove probable nans
                    fl_diag_dss[i] = ds.dropna('time', subset=['volume_m3'])

            # Write out?
            if fl_diag_path not in [True, None]:
                encode = {}
                for v in fl_diag_dss[0]:
                    encode[v] = {'zlib': True, 'complevel': 5}

                # Welcome ds
                ds = xr.Dataset()
                ds.attrs['description'] = ('OGGM model output on flowlines. '
                                           'Check groups for data.')
                ds.attrs['oggm_version'] = __version__
                # This is useful to interpret the dataset afterwards
                flows_to_id = []
                for trib in self._tributary_indices:
                    flows_to_id.append(trib[0] if trib[0] is not None else -1)
                ds['flowlines'] = ('flowlines', np.arange(len(flows_to_id)))
                ds['flows_to_id'] = ('flowlines', flows_to_id)
                ds.to_netcdf(fl_diag_path, 'w')
                for i, ds in enumerate(fl_diag_dss):
                    ds.to_netcdf(fl_diag_path, 'a', group='fl_{}'.format(i),
                                 encoding=encode)

        if do_geom and geom_path not in [True, None]:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(geom_ds):
                ds.to_netcdf(geom_path, 'a', group='fl_{}'.format(i),
                             encoding=encode)

            # Add calving to geom file because the FileModel can't infer it
            if 'calving_m3' in diag_ds:
                diag_ds[['calving_m3']].to_netcdf(geom_path, 'a')

        if diag_path not in [True, None]:
            diag_ds.to_netcdf(diag_path)

        # Decide on what to give back
        out = [diag_ds]
        if fl_diag_dss is not None:
            out.append(fl_diag_dss)
        if geom_ds is not None:
            out.append(geom_ds)
        if len(out) == 1:
            out = out[0]
        else:
            out = tuple(out)
        return out

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """ Runs the model until an equilibrium state is reached.

        Be careful: This only works for CONSTANT (not time-dependent)
        mass balance models.
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


def flux_gate_with_build_up(year, flux_value=None, flux_gate_yr=None):
    """Default scalar flux gate with build up period"""
    fac = 1 - (flux_gate_yr - year) / flux_gate_yr
    return flux_value * utils.clip_scalar(fac, 0, 1)


def k_calving_law(model, flowline, last_above_wl):
    """Compute calving from the model state using the k-calving law.

    Currently this still assumes that the model has an attribute
    called "calving_k", which might be changed in the future.

    Parameters
    ----------
    model : oggm.core.flowline.FlowlineModel
        the model instance calling the function
    flowline : oggm.core.flowline.Flowline
        the instance of the flowline object on which the calving law is called
    last_above_wl : int
        the index of the last pixel above water (in case you need to know
        where it is).
    """
    h = flowline.thick[last_above_wl]
    d = h - (flowline.surface_h[last_above_wl] - model.water_level)
    k = model.calving_k
    q_calving = k * d * h * flowline.widths_m[last_above_wl]
    return q_calving


class FluxBasedModel(FlowlineModel):
    """The flowline model used by OGGM in production.

    It solves for the SIA along the flowline(s) using a staggered grid. It
    computes the *ice flux* between grid points and transports the mass
    accordingly (also between flowlines).

    This model is numerically less stable than fancier schemes, but it
    is fast and works with multiple flowlines of any bed shape (rectangular,
    parabolic, trapeze, and any combination of them).

    We test that it conserves mass in most cases, but not on very stiff cliffs.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=100,
                 do_kcalving=None, calving_k=None, calving_law=k_calving_law,
                 calving_use_limiter=None, calving_limiter_frac=None,
                 water_level=None,
                 **kwargs):
        """Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overridden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_law : func
             option to use another calving law. This is a temporary workaround
             to test other calving laws, and the system might be improved in
             future OGGM versions.
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `flowline_model_run` does for you)
        """
        super(FluxBasedModel, self).__init__(flowlines, mb_model=mb_model,
                                             y0=y0, glen_a=glen_a, fs=fs,
                                             inplace=inplace,
                                             water_level=water_level,
                                             **kwargs)

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.min_dt = min_dt
        self.cfl_number = cfl_number

        # Calving params
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.calving_law = calving_law
        self.do_calving = do_kcalving and self.is_tidewater
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']
        self.calving_k = calving_k / cfg.SEC_IN_YEAR
        if calving_use_limiter is None:
            calving_use_limiter = cfg.PARAMS['calving_use_limiter']
        self.calving_use_limiter = calving_use_limiter
        if calving_limiter_frac is None:
            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
        if calving_limiter_frac > 0:
            raise NotImplementedError('calving limiter other than 0 not '
                                      'implemented yet')
        self.calving_limiter_frac = calving_limiter_frac

        # Flux gate
        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
        self.flux_gate_m3_since_y0 = 0.
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            flux_gate_thickness = utils.tolist(flux_gate_thickness,
                                               length=len(self.fls))
            self.flux_gate = []
            for fl, fgt in zip(self.fls, flux_gate_thickness):
                # We set the thickness to the desired value so that
                # the widths work ok
                fl = copy.deepcopy(fl)
                fl.thick = fl.thick * 0 + fgt
                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
                if slope == 0:
                    raise ValueError('I need a slope to compute the flux')
                flux = find_sia_flux_from_thickness(slope,
                                                    fl.widths_m[0],
                                                    fgt,
                                                    shape=fl.shape_str[0],
                                                    glen_a=self.glen_a,
                                                    fs=self.fs)
                self.flux_gate.append(flux)

        # convert the floats to function calls
        for i, fg in enumerate(self.flux_gate):
            if fg is None:
                continue
            try:
                # Do we have a function? If yes all good
                fg(self.yr)
            except TypeError:
                # If not, make one
                self.flux_gate[i] = partial(flux_gate_with_build_up,
                                            flux_value=fg,
                                            flux_gate_yr=(flux_gate_build_up +
                                                          self.y0))
        # Special output
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.u_stag = []
        self.shapefac_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an additional fake grid point at the end of tributaries
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.shapefac_stag.append(np.ones(nx+1))  # beware the ones!
            self.flux_stag.append(np.zeros(nx+1))

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            flux_gate = self.flux_gate[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
            elif self.do_calving and self.calving_use_limiter:
                # We lower the max possible ice deformation
                # by clipping the surface slope here. It is completely
                # arbitrary but reduces ice deformation at the calving front.
                # I think that in essence, it is also partly
                # a "calving process", because this ice deformation must
                # be less at the calving front. The result is that calving
                # front "free boards" are quite high.
                # Note that 0 is arbitrary, it could be any value below SL
                surface_h = utils.clip_min(surface_h, self.water_level)

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho*G*slope_stag)**N
            u_stag[:] = (thick_stag**(N+1)) * self._fd * rhogh * sf_stag**N + \
                        (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > cfg.FLOAT_EPS:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s. Happening at '
                            'simulation year {:.1f}, fl_id {}, '
                            'bin_id {} and max_u {:.3f} m yr-1.'
                            ''.format(cfl_dt, self.min_dt, self.yr, fl_id,
                                      np.argmax(np.abs(u_stag)),
                                      maxu * cfg.SEC_IN_YEAR))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None

            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # --- The rest is for calving only ---
            self.calving_rate_myr = 0.

            # If tributary, do calving only if we are not transferring mass
            if is_trib and flx_stag[-1] > 0:
                continue

            # No need to do calving in these cases either
            if not self.do_calving or not fl.has_ice():
                continue

            # We do calving only if the last glacier bed pixel is below water
            # (this is to avoid calving elsewhere than at the front)
            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
                continue

            # We do calving only if there is some ice above wl
            last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                       (fl.thick > 0))[0][-1]
            if fl.bed_h[last_above_wl] > self.water_level:
                continue

            # OK, we're really calving
            section = fl.section

            # Calving law
            q_calving = self.calving_law(self, fl, last_above_wl)

            # Add to the bucket and the diagnostics
            fl.calving_bucket_m3 += q_calving * dt
            self.calving_m3_since_y0 += q_calving * dt
            self.calving_rate_myr = (q_calving / section[last_above_wl] *
                                     cfg.SEC_IN_YEAR)

            # See if we have ice below sea-water to clean out first
            below_sl = (fl.surface_h < self.water_level) & (fl.thick > 0)
            to_remove = np.sum(section[below_sl]) * fl.dx_meter
            if 0 < to_remove < fl.calving_bucket_m3:
                # This is easy, we remove everything
                section[below_sl] = 0
                fl.calving_bucket_m3 -= to_remove
            elif to_remove > 0:
                # We can only remove part of if
                section[below_sl] = 0
                section[last_above_wl+1] = ((to_remove - fl.calving_bucket_m3)
                                            / fl.dx_meter)
                fl.calving_bucket_m3 = 0

            # The rest of the bucket might calve an entire grid point (or more?)
            vol_last = section[last_above_wl] * fl.dx_meter
            while fl.calving_bucket_m3 > vol_last:
                fl.calving_bucket_m3 -= vol_last
                section[last_above_wl] = 0

                # OK check if we need to continue (unlikely)
                last_above_wl -= 1
                vol_last = section[last_above_wl] * fl.dx_meter

            # We update the glacier with our changes
            fl.section = section

        # Next step
        self.t += dt
        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Velocities in OGGM's FluxBasedModel are sometimes subject to
        numerical instabilities. To deal with the issue, you can either
        set a smaller ``PARAMS['cfl_number']`` (e.g. 0.01) or smooth the
        output a bit, e.g. with ``df.rolling(5, center=True, min_periods=1).mean()``

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
            - surface_ice_velocity: m per second (corrected for surface - simplified)
        """
        import pandas as pd

        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
        df['surface_ice_velocity'] = df['ice_velocity'] * self._surf_vel_fac
        var = self.shapefac_stag[fl_id]
        df['shape_fac'] = (var[1:nx+1] + var[:nx])/2

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id]

        return df


class MassConservationChecker(FluxBasedModel):
    """This checks if the FluxBasedModel is conserving mass."""

    def __init__(self, flowlines, **kwargs):
        """ Instantiate.

        Parameters
        ----------

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
            mb = utils.clip_min(mb, -sec)
            self.total_mass += np.sum(mb * dx)


class KarthausModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31*SEC_IN_DAY,
                 inplace=False, **kwargs):
        """ Instantiate.

        Parameters
        ----------
        """

        if len(flowlines) > 1:
            raise ValueError('Karthaus model does not work with tributaries.')

        super(KarthausModel, self).__init__(flowlines, mb_model=mb_model,
                                            y0=y0, glen_a=glen_a, fs=fs,
                                            inplace=inplace, **kwargs)
        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt
        dt = utils.clip_scalar(dt, min_dt, self.max_dt)

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

        fl.thick = utils.clip_min(NewIceThickness, 0)

        # Next step
        self.t += dt
        return dt


class SemiImplicitModel(FlowlineModel):
    """Semi implicit flowline model.

    It solves the same equation as the FluxBasedModel, but the ice flux q is
    implemented as q^t = D^t * (ds/dx)^(t+1).

    It supports only a single flowline (no tributaries) with bed shapes
    rectangular, trapezoidal or a mixture of both.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 inplace=False, fixed_dt=None, cfl_number=0.5, min_dt=None,
                 **kwargs):
        """Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            wether to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx^2 / max(D/w)).
            Can be set to higher values compared to the FluxBasedModel.
            Default is 0.5, but need further investigation.
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        kwargs : dict
            Further keyword arguments for FlowlineModel

        """

        super(SemiImplicitModel, self).__init__(flowlines, mb_model=mb_model,
                                                y0=y0, glen_a=glen_a, fs=fs,
                                                inplace=inplace, **kwargs)

        if len(self.fls) > 1:
            raise ValueError('Implicit model does not work with '
                             'tributaries.')

        # convert pure RectangularBedFlowline to TrapezoidalBedFlowline with
        # lambda = 0
        if isinstance(self.fls[0], RectangularBedFlowline):
            self.fls[0] = TrapezoidalBedFlowline(
                line=self.fls[-1].line, dx=self.fls[-1].dx,
                map_dx=self.fls[-1].map_dx, surface_h=self.fls[-1].surface_h,
                bed_h=self.fls[-1].bed_h, widths=self.fls[-1].widths,
                lambdas=0, rgi_id=self.fls[-1].rgi_id,
                water_level=self.fls[-1].water_level, gdir=None)

        if isinstance(self.fls[0], MixedBedFlowline):
            if ~np.all(self.fls[0].is_trapezoid):
                raise ValueError('Implicit model only works with a pure '
                                 'trapezoidal flowline! But different lambdas '
                                 'along the flowline possible (lambda=0 is'
                                 'rectangular).')
        elif not isinstance(self.fls[0], TrapezoidalBedFlowline):
            raise ValueError('Implicit model only works with a pure '
                             'trapezoidal flowline! But different lambdas '
                             'along the flowline possible (lambda=0 is'
                             'rectangular).')

        if cfg.PARAMS['use_kcalving_for_run']:
            raise NotImplementedError("Calving is not implemented in the"
                                      "SemiImplicitModel! Set "
                                      "cfg.PARAMS['use_kcalving_for_run'] = "
                                      "False or use a FluxBasedModel.")

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        self.min_dt = min_dt

        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        if cfl_number < 0.1:
            raise InvalidParamsError("For the SemiImplicitModel you can use "
                                     "cfl numbers in the order of 0.1 - 0.5 "
                                     f"(you set {cfl_number}).")
        self.cfl_number = cfl_number

        # Special output
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        # optim
        nx = self.fls[-1].nx
        bed_h_exp = np.concatenate(([self.fls[-1].bed_h[0]],
                                    self.fls[-1].bed_h,
                                    [self.fls[-1].bed_h[-1]]))
        self.dbed_h_exp_dx = ((bed_h_exp[1:] - bed_h_exp[:-1]) /
                              self.fls[0].dx_meter)
        self.d_stag = [np.zeros(nx + 1)]
        self.d_matrix_banded = np.zeros((3, nx))
        w0 = self.fls[0]._w0_m
        self.w0_stag = (w0[0:-1] + w0[1:]) / 2
        self.rhog = (self.rho * G) ** self.glen_n

        # variables needed for the calculation of some diagnostics, this
        # calculations are done with @property, because they are not computed
        # on the fly during the dynamic run as in FluxBasedModel
        self._u_stag = [np.zeros(nx + 1)]
        self._flux_stag = [np.zeros(nx + 1)]
        self._slope_stag = [np.zeros(nx + 1)]
        self._thick_stag = [np.zeros(nx + 1)]
        self._section_stag = [np.zeros(nx + 1)]

    @property
    def slope_stag(self):
        slope_stag = self._slope_stag[0]

        surface_h = self.fls[0].surface_h
        dx = self.fls[0].dx_meter

        slope_stag[0] = 0
        slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
        slope_stag[-1] = slope_stag[-2]

        return [slope_stag]

    @property
    def thick_stag(self):
        thick_stag = self._thick_stag[0]

        thick = self.fls[0].thick

        thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
        thick_stag[[0, -1]] = thick[[0, -1]]

        return [thick_stag]

    @property
    def section_stag(self):
        section_stag = self._section_stag[0]

        section = self.fls[0].section

        section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
        section_stag[[0, -1]] = section[[0, -1]]

        return [section_stag]

    @property
    def u_stag(self):
        u_stag = self._u_stag[0]

        slope_stag = self.slope_stag[0]
        thick_stag = self.thick_stag[0]
        N = self.glen_n
        rhog = self.rhog

        rhogh = rhog * slope_stag ** N

        u_stag[:] = ((thick_stag**(N+1)) * self._fd * rhogh +
                     (thick_stag**(N-1)) * self.fs * rhogh)

        return [u_stag]

    @property
    def flux_stag(self):
        flux_stag = self._flux_stag[0]

        section_stag = self.section_stag[0]
        u_stag = self.u_stag[0]

        flux_stag[:] = u_stag * section_stag

        return [flux_stag]

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # read out variables from current flowline
        fl = self.fls[0]
        dx = fl.dx_meter
        width = fl.widths_m
        thick = fl.thick
        surface_h = fl.surface_h

        # some variables needed later
        N = self.glen_n
        rhog = self.rhog

        # calculate staggered variables
        width_stag = (width[0:-1] + width[1:]) / 2
        w0_stag = self.w0_stag
        thick_stag = (thick[0:-1] + thick[1:]) / 2.
        dsdx_stag = (surface_h[1:] - surface_h[0:-1]) / dx

        # calculate diffusivity
        # boundary condition d_stag_0 = d_stag_end = 0
        d_stag = self.d_stag[0]
        d_stag[1:-1] = ((self._fd * thick_stag ** (N + 2) +
                         self.fs * thick_stag ** N) * rhog *
                        (w0_stag + width_stag) / 2 *
                        np.abs(dsdx_stag) ** (N - 1))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt
        else:
            # use stability criterion dt <= dx^2 / max(D/w) * cfl_number
            divisor = np.max(np.abs(d_stag[1:-1] / width_stag))
            if divisor > cfg.FLOAT_EPS:
                cfl_dt = self.cfl_number * dx ** 2 / divisor
            else:
                cfl_dt = dt

            if cfl_dt < dt:
                dt = cfl_dt
                if cfl_dt < self.min_dt:
                    raise RuntimeError(
                        'CFL error: required time step smaller '
                        'than the minimum allowed: '
                        '{:.1f}s vs {:.1f}s. Happening at '
                        'simulation year {:.1f}, fl_id {}, '
                        'bin_id {} and max_D {:.3f} m2 yr-1.'
                        ''.format(cfl_dt, self.min_dt, self.yr, 0,
                                  np.argmax(np.abs(d_stag)),
                                  divisor * cfg.SEC_IN_YEAR))

        # calculate diagonals of Amat
        d0 = dt / dx ** 2 * (d_stag[:-1] + d_stag[1:]) / width
        dm = - dt / dx ** 2 * d_stag[:-1] / width
        dp = - dt / dx ** 2 * d_stag[1:] / width

        # construct banded form of the matrix, which is used during solving
        # (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)
        # original matrix:
        # d_matrix = (np.diag(dp[:-1], 1) +
        #             np.diag(np.ones(len(d0)) + d0) +
        #             np.diag(dm[1:], -1))
        self.d_matrix_banded[0, 1:] = dp[:-1]
        self.d_matrix_banded[1, :] = np.ones(len(d0)) + d0
        self.d_matrix_banded[2, :-1] = dm[1:]

        # correction term for glacier bed (original equation is an equation for
        # the surface height s, which is transformed in an equation for h, as
        # s = h + b the term below comes from the '- b'
        b_corr = - d_stag * self.dbed_h_exp_dx

        # prepare rhs
        smb = self.get_mb(surface_h, self.yr, fl_id=0, fls=self.fls)
        rhs = thick + smb * dt + dt / width * (b_corr[:-1] - b_corr[1:]) / dx

        # solve matrix and update flowline thickness
        thick_new = utils.clip_min(
            solve_banded((1, 1), self.d_matrix_banded, rhs),
            0)
        fl.thick = thick_new

        # Next step
        self.t += dt

        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
            - surface_ice_velocity: m per second (corrected for surface - simplified)
        """
        import pandas as pd

        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
        df['surface_ice_velocity'] = df['ice_velocity'] * self._surf_vel_fac

        return df


class FileModel(object):
    """Duck FlowlineModel which actually reads data out of a nc file."""

    def __init__(self, path):
        """ Instantiate."""

        self.fls = glacier_from_netcdf(path)

        fl_tss = []
        for flid, fl in enumerate(self.fls):
            with xr.open_dataset(path, group='fl_{}'.format(flid)) as ds:
                if flid == 0:
                    # Populate time
                    self.time = ds.time.values
                    try:
                        self.years = ds.year.values
                    except AttributeError:
                        raise InvalidWorkflowError('The provided model output '
                                                   'file is incomplete (likely '
                                                   'when the previous '
                                                   'run failed) or corrupt.')
                    try:
                        self.months = ds.month.values
                    except AttributeError:
                        self.months = self.years * 0 + 1

                # Read out the data
                fl_data = {
                    'ts_section': ds.ts_section.values,
                    'ts_width_m': ds.ts_width_m.values,
                }
                try:
                    fl_data['ts_calving_bucket_m3'] = ds.ts_calving_bucket_m3.values
                except AttributeError:
                    fl_data['ts_calving_bucket_m3'] = self.years * 0
                fl_tss.append(fl_data)

        self.fl_tss = fl_tss
        self.last_yr = float(ds.time[-1])

        # Calving diags
        try:
            with xr.open_dataset(path) as ds:
                self._calving_m3_since_y0 = ds.calving_m3.values
                self.water_level = ds.water_level
                self.do_calving = True
        except AttributeError:
            self._calving_m3_since_y0 = 0
            self.do_calving = False
            self.water_level = 0

        # time
        self.reset_y0()

    def __enter__(self):
        warnings.warn('FileModel no longer needs to be run as a '
                      'context manager. You can safely remove the '
                      '`with` statement.', FutureWarning)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def reset_y0(self, y0=None):
        """Reset the initial model time"""

        if y0 is None:
            y0 = float(self.time[0])
        self.y0 = y0
        self.yr = y0
        self._current_index = 0

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

    @property
    def calving_m3_since_y0(self):
        if self.do_calving:
            return self._calving_m3_since_y0[self._current_index]
        else:
            return 0

    def run_until(self, year=None, month=None):
        """Mimics the model's behavior.

        Is quite slow tbh.
        """
        try:
            if month is not None:
                pok = np.nonzero((self.years == year) & (self.months == month))[0][0]
            else:
                pok = np.nonzero(self.time == year)[0][0]
        except IndexError as err:
            raise IndexError('Index year={}, month={} not available in '
                             'FileModel.'.format(year, month)) from err

        self.yr = self.time[pok]
        self._current_index = pok

        for fl, fl_ts in zip(self.fls, self.fl_tss):
            fl.section = fl_ts['ts_section'][pok, :]
            fl.calving_bucket_m3 = fl_ts['ts_calving_bucket_m3'][pok]

    def area_m2_ts(self, rollmin=0):
        """rollmin is the number of years you want to smooth onto"""
        sel = 0
        for fl, fl_ts in zip(self.fls, self.fl_tss):
            widths = np.where(fl_ts['ts_section'] > 0., fl_ts['ts_width_m'], 0.)
            sel += widths.sum(axis=1) * fl.dx_meter
        sel = pd.Series(data=sel, index=self.time, name='area_m2')
        if rollmin != 0:
            sel = sel.rolling(rollmin).min()
            sel.iloc[0:rollmin] = sel.iloc[rollmin]
        return sel

    def area_km2_ts(self, **kwargs):
        return self.area_m2_ts(**kwargs) * 1e-6

    def volume_m3_ts(self):
        sel = 0
        for fl, fl_ts in zip(self.fls, self.fl_tss):
            sel += fl_ts['ts_section'].sum(axis=1) * fl.dx_meter
            sel -= fl_ts['ts_calving_bucket_m3']
        return pd.Series(data=sel, index=self.time, name='volume_m3')

    def volume_km3_ts(self):
        return self.volume_m3_ts() * 1e-9

    def length_m_ts(self, rollmin=0):
        raise NotImplementedError('length_m_ts is no longer available in the '
                                  'full output files. To obtain the length '
                                  'time series, refer to the diagnostic '
                                  'output file.')


class MassRedistributionCurveModel(FlowlineModel):
    """Glacier geometry updated using mass redistribution curves.

    Also known as the "delta-h method": This uses mass redistribution curves
    from Huss et al. (2010) to update the glacier geometry.

    Code by David Rounce (PyGEM) and adapted by F. Maussion.
    """

    def __init__(self, flowlines, mb_model=None, y0=0.,
                 is_tidewater=False, water_level=None,
                 do_kcalving=None, calving_k=None,
                 advance_method=1,
                 **kwargs):
        """ Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
        y0 : int
            initial year of the simulation
        advance_method : int
            different ways to handle positive MBs:
            - 0: do nothing, i.e. simply let the glacier thicken instead of
                 thinning
            - 1: add some of the mass at the end of the glacier
            - 2: add some of the mass at the end of the glacier, but
                 differently
        """
        super(MassRedistributionCurveModel, self).__init__(flowlines,
                                                           mb_model=mb_model,
                                                           y0=y0,
                                                           water_level=water_level,
                                                           mb_elev_feedback='annual',
                                                           required_model_steps='annual',
                                                           **kwargs)

        if len(self.fls) > 1:
            raise InvalidWorkflowError('MassRedistributionCurveModel is not '
                                       'set up for multiple flowlines')
        fl = self.fls[0]
        self.glac_idx_initial = fl.thick.nonzero()[0]
        self.y0 = y0

        # Some ways to deal with positive MB
        self.advance_method = advance_method

        # Frontal ablation shenanigans
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.do_calving = do_kcalving and self.is_tidewater
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']

        self.is_tidewater = is_tidewater
        self.calving_k = calving_k
        self.calving_m3_since_y0 = 0.  # total calving since time y0

    def step(self, dt):
        """Advance one step. Here it should be one year"""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')
        if dt > cfg.SEC_IN_YEAR:
            # This should not happen from how run_until is built, but
            # to match the adaptive time stepping scheme of other models
            # we don't complain here and just do one year
            dt = cfg.SEC_IN_YEAR

        elif dt < cfg.SEC_IN_YEAR:
            # Here however we complain - we really want one year exactly
            raise InvalidWorkflowError('I was asked to run for less than one '
                                       'year. Delta-H models can\'t do that.')

        # Flowline state
        fl = self.fls[0]
        fl_id = 0

        height = fl.surface_h.copy()
        section = fl.section.copy()
        thick = fl.thick.copy()
        width = fl.widths_m.copy()

        # FRONTAL ABLATION
        if self.do_calving:
            raise NotImplementedError('Frontal ablation not there yet.')

        # Redistribute mass if glacier is still there
        if not np.any(section > 0):
            # Do nothing
            self.t += dt
            return dt

        # Mass redistribution according to Huss empirical curves
        # Annual glacier mass balance [m ice s-1]
        mb = self.get_mb(height, year=self.yr, fls=self.fls, fl_id=fl_id)
        # [m ice yr-1]
        mb *= cfg.SEC_IN_YEAR

        # Ok now to the bulk of it
        # Mass redistribution according to empirical equations from
        # Huss and Hock (2015) accounting for retreat/advance.
        # glac_idx_initial is required to ensure that the glacier does not
        # advance to area where glacier did not exist before
        # (e.g., retreat and advance over a vertical cliff)

        # Note: since OGGM uses the DEM, heights along the flowline do not
        # necessarily decrease, i.e., there can be overdeepenings along the
        # flowlines that occur as the glacier retreats. This is problematic
        # for 'adding' a bin downstream in cases of glacier advance because
        # you'd be moving new ice to a higher elevation. To avoid this
        # unrealistic case, in the event that this would occur, the
        # overdeepening will simply fill up with ice first until it reaches
        # an elevation where it would put new ice into a downstream bin.

        # Bin area [m2]
        bin_area = width * fl.dx_meter
        bin_area[thick == 0] = 0

        # Annual glacier-wide volume change [m3]
        # units: m3 ice per year
        glacier_delta_v = (mb * bin_area).sum()

        # For hindcast simulations, volume change is the opposite
        # We don't implement this in OGGM right now

        # If volume loss is more than the glacier volume, melt everything and
        # stop here. Otherwise, redistribute mass loss/gains across the glacier
        glacier_volume_total = (section * fl.dx_meter).sum()
        if (glacier_volume_total + glacier_delta_v) < 0:
            # Set all to zero and return
            fl.section *= 0
            return

        # Determine where glacier exists
        glac_idx = thick.nonzero()[0]

        # Compute bin volume change [m3 ice yr-1] after redistribution
        bin_delta_v = mass_redistribution_curve_huss(height, bin_area, mb,
                                                     glac_idx, glacier_delta_v)

        # Here per construction bin_delta_v should be approx equal to glacier_delta_v
        np.testing.assert_allclose(bin_delta_v.sum(), glacier_delta_v)

        # Update cross sectional area
        # relevant issue: https://github.com/OGGM/oggm/issues/941
        #  volume change divided by length (dx); units m2
        delta_s = bin_delta_v / fl.dx_meter

        fl.section = utils.clip_min(section + delta_s, 0)
        if not np.any(delta_s > 0):
            # We shrink - all good
            fl.section = utils.clip_min(section + delta_s, 0)
            # Done
            self.t += dt
            return dt

        # We grow - that's bad because growing is hard

        # First see what happens with thickness
        # (per construction the redistribution curves are all positive btw)
        fl.section = utils.clip_min(section + delta_s, 0)
        # We decide if we really want to advance or if we don't care
        # Matthias and Dave use 5m, I find that too much, because not
        # redistributing may lead to runaway effects
        dh_thres = 1  # in meters
        if np.all((fl.thick - thick) <= dh_thres):
            # That was not much increase return
            self.t += dt
            return dt

        # Ok, we really grow then - back to previous state and decide on what to do
        fl.section = section

        if (fl.thick[-2] > 0) or (self.advance_method == 0):
            # Do not advance (same as in the melting case but thickening)
            fl.section = utils.clip_min(section + delta_s, 0)

        if self.advance_method == 1:
            # Just shift the redistribution by one pix
            new_delta_s = np.append([0], delta_s)[:-1]

            # Redis param - how much of mass we want to shift (0 - 1)
            # 0 would be like method 0 (do nothing)
            # 1 would be to shift all mass by 1 pixel
            a = 0.2
            new_delta_s = a * new_delta_s + (1 - a) * delta_s

            # Make sure we are still preserving mass
            new_delta_s *= delta_s.sum() / new_delta_s.sum()

            # Update section
            fl.section = utils.clip_min(section + new_delta_s, 0)

        elif self.advance_method == 2:

            # How much of what's positive do we want to add in front
            redis_perc = 0.01  # in %

            # Decide on volume that needs redistributed
            section_redis = delta_s * redis_perc

            # The rest is added where it was
            fl.section = utils.clip_min(section + delta_s - section_redis, 0)

            # Then lets put this "volume" where we can
            section_redis = section_redis.sum()
            while section_redis > 0:
                # Get the terminus grid
                orig_section = fl.section.copy()
                p_term = np.nonzero(fl.thick > 0)[0][-1]

                # Put ice on the next bin, until ice is as high as terminus
                # Anything else would require slope assumptions, but it would
                # be possible
                new_thick = fl.surface_h[p_term] - fl.bed_h[p_term + 1]
                if new_thick > 0:
                    # No deepening
                    fl.thick[p_term + 1] = new_thick
                    new_section = fl.section[p_term + 1]
                    if new_section > section_redis:
                        # This would be too much, just add what we have
                        orig_section[p_term + 1] = section_redis
                        fl.section = orig_section
                        section_redis = 0
                    else:
                        # OK this bin is done, continue
                        section_redis -= new_section
                else:
                    # We have a deepening, or we have to climb
                    target_h = fl.bed_h[p_term + 1] + 1
                    to_fill = (fl.surface_h < target_h) & (fl.thick > 0)

                    # Theoretical section area needed
                    fl.thick[to_fill] = target_h - fl.bed_h[to_fill]
                    new_section = fl.section.sum() - orig_section.sum()
                    if new_section > section_redis:
                        # This would be too much, just add what we have
                        orig_section[to_fill] += new_section / np.sum(to_fill)
                        fl.section = orig_section
                        section_redis = 0
                    else:
                        # OK this bin is done, continue
                        section_redis -= new_section

        elif self.advance_method == 3:
            # A bit closer to Huss and Rounce maybe?
            raise RuntimeError('not yet')

        # Done
        self.t += dt
        return dt


def mass_redistribution_curve_huss(height, bin_area, mb, glac_idx, glacier_delta_v):
    """Apply the mass redistribution curves from Huss and Hock (2015).

    This has to be followed by added logic which takes into consideration
    retreat and advance.

    Parameters
    ----------
    height : np.ndarray
        Glacier elevation [m] from previous year for each elevation bin
    bin_area : np.ndarray
        Glacier area [m2] from previous year for each elevation bin
    mb : np.ndarray
        Annual climatic mass balance [m ice yr-1] for each elevation bin for
        a single year
    glac_idx : np.ndarray
        glacier indices for present timestep
    glacier_delta_v : float
        glacier-wide volume change [m3 ice yr-1] based on the annual mb

    Returns
    -------
    bin_volume_change : np.ndarray
        Ice volume change [m3 yr-1] for each elevation bin
    """

    # Apply mass redistribution curve
    if glac_idx.shape[0] <= 3:
        # No need for a curve when glacier is so small
        return mb * bin_area

    # Select the parameters based on glacier area
    if bin_area.sum() * 1e-6 > 20:
        gamma, a, b, c = (6, -0.02, 0.12, 0)
    elif bin_area.sum() * 1e-6 > 5:
        gamma, a, b, c = (4, -0.05, 0.19, 0.01)
    else:
        gamma, a, b, c = (2, -0.30, 0.60, 0.09)

    # reset variables
    delta_h_norm = bin_area * 0

    # Normalized elevation range [-]
    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
    gla_height = height[glac_idx]
    max_elev, min_elev = gla_height.max(), gla_height.min()
    h_norm = (max_elev - gla_height) / (max_elev - min_elev)

    #  using indices as opposed to elevations automatically skips bins on
    #  the glacier that have no area such that the normalization is done
    #  only on bins where the glacier lies
    # Normalized ice thickness change [-]
    delta_h_norm[glac_idx] = (h_norm + a) ** gamma + b * (h_norm + a) + c

    # delta_h = (h_n + a)**gamma + b*(h_n + a) + c
    # limit normalized ice thickness change to between 0 - 1
    delta_h_norm = utils.clip_array(delta_h_norm, 0, 1)

    # Huss' ice thickness scaling factor, fs_huss [m ice]
    #  units: m3 / (m2 * [-]) * (1000 m / 1 km) = m ice
    fs_huss = glacier_delta_v / (bin_area * delta_h_norm).sum()

    # Volume change [m3 ice yr-1]
    return delta_h_norm * fs_huss * bin_area


def flowline_from_dataset(ds):
    """Instantiates a flowline from an xarray Dataset."""

    cl = globals()[ds.attrs['class']]
    line = shpg.LineString(ds['linecoords'].values)
    args = dict(line=line, dx=ds.dx, map_dx=ds.map_dx,
                surface_h=ds['surface_h'].values,
                bed_h=ds['bed_h'].values)

    have = {'c', 'x', 'surface_h', 'linecoords', 'bed_h', 'z', 'p', 'n',
            'time', 'month', 'year', 'ts_width_m', 'ts_section',
            'ts_calving_bucket_m3'}
    missing_vars = set(ds.variables.keys()).difference(have)
    for k in missing_vars:
        data = ds[k].values
        if ds[k].dims[0] == 'z':
            data = data[0]
        args[k] = data
    return cl(**args)


def glacier_from_netcdf(path):
    """Instantiates a list of flowlines from an xarray Dataset."""

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


def calving_glacier_downstream_line(line, n_points):
    """Extends a calving glacier flowline past the terminus."""
    if line is None:
        return None
    x, y = line.coords.xy
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    x = np.append(x, x[-1] + dx * np.arange(1, n_points+1))
    y = np.append(y, y[-1] + dy * np.arange(1, n_points+1))
    return shpg.LineString(np.array([x, y]).T)


@entity_task(log, writes=['model_flowlines'])
def init_present_time_glacier(gdir, filesuffix='',
                              use_binned_thickness_data=False):
    """Merges data from preprocessing tasks. First task after inversion!

    This updates the `mode_flowlines` file and creates a stand-alone numerical
    glacier ready to run.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    filesuffix : str
        append a suffix to the model_flowlines filename (e.g. useful for
        dynamic melt_f calibration including an inversion, so the original
        model_flowlines are not changed).
    use_binned_thickness_data : bool or str
        if you want to use thickness data, which was binned to the elevation
        band flowlines with tasks.elevation_band_flowine and
        tasks.fixed_dx_elevation_band_flowline, you can provide the name of the
        data here to create a flowline for a dynamic model run
    """

    # Some vars
    invs = gdir.read_pickle('inversion_output')

    map_dx = gdir.grid.dx
    def_lambda = cfg.PARAMS['trapezoid_lambdas']
    cls = gdir.read_pickle('inversion_flowlines')

    # Fill the tributaries
    new_fls = []
    flows_to_ids = []
    for cl, inv in zip(cls, invs):

        # Get the data to make the model flowlines
        line = cl.line
        surface_h = cl.surface_h
        widths_m = cl.widths * map_dx
        assert np.all(widths_m > 0)

        if not use_binned_thickness_data:
            # classical initialisation after the inversion
            section = inv['volume'] / (cl.dx * map_dx)
            bed_h = surface_h - inv['thick']

            bed_shape = 4 * inv['thick'] / widths_m ** 2

            lambdas = inv['thick'] * np.nan
            lambdas[inv['is_trapezoid']] = def_lambda
            lambdas[inv['is_rectangular']] = 0.

            # Where the flux and the thickness is zero we just assume trapezoid:
            lambdas[bed_shape == 0] = def_lambda

        else:
            # here we use binned thickness data for the initialisation
            elev_fl = pd.read_csv(
                gdir.get_filepath('elevation_band_flowline',
                                  filesuffix='_fixed_dx'), index_col=0)
            assert np.allclose(widths_m, elev_fl['widths_m'])
            elev_fl_thick = elev_fl[use_binned_thickness_data].values
            section = elev_fl_thick * widths_m
            lambdas = np.ones(len(section)) * def_lambda

            # for trapezoidal the calculation of thickness results in quadratic
            # equation, we only keep solution which results in a positive w0
            # value (-> w/lambda >= h),
            # it still could happen that we would need a negative w0 if the
            # section is too large, in those cases we get a negative value
            # inside sqrt (we ignore the RuntimeWarning) -> if this happens we
            # use rectangular shape with original thickness
            with np.errstate(invalid='ignore'):
                thick = ((2 * widths_m -
                          np.sqrt(4 * widths_m ** 2 -
                                  4 * lambdas * 2 * section)) /
                         (2 * lambdas))
            nan_thick = np.isnan(thick)
            thick[nan_thick] = elev_fl_thick[nan_thick]
            lambdas[nan_thick] = 0

            # finally the glacier bed and other stuff
            bed_h = surface_h - thick
            bed_shape = 4 * thick / widths_m ** 2

        if not gdir.is_tidewater and inv['is_last']:
            # for valley glaciers, simply add the downstream line, depending on
            # selected shape parabola or trapezoidal
            dic_ds = gdir.read_pickle('downstream_line')
            if cfg.PARAMS['downstream_line_shape'] == 'parabola':
                bed_shape = np.append(bed_shape, dic_ds['bedshapes'])
                lambdas = np.append(lambdas, dic_ds['bedshapes'] * np.nan)
                widths_m = np.append(widths_m, dic_ds['bedshapes'] * 0.)
            elif cfg.PARAMS['downstream_line_shape'] == 'trapezoidal':
                bed_shape = np.append(bed_shape, dic_ds['bedshapes'] * np.nan)
                lambdas = np.append(lambdas, np.ones(len(dic_ds['w0s'])) *
                                    def_lambda)
                widths_m = np.append(widths_m, dic_ds['w0s'])
            else:
                raise InvalidParamsError(
                    f"Unknown cfg.PARAMS['downstream_line_shape'] = "
                    f"{cfg.PARAMS['downstream_line_shape']} (options are "
                    f"'parabola' and 'trapezoidal').")
            section = np.append(section, dic_ds['bedshapes'] * 0.)
            surface_h = np.append(surface_h, dic_ds['surface_h'])
            bed_h = np.append(bed_h, dic_ds['surface_h'])
            line = dic_ds['full_line']

        if gdir.is_tidewater and inv['is_last']:
            # Continue the bed a little
            n_points = cfg.PARAMS['calving_line_extension']
            cf_slope = cfg.PARAMS['calving_front_slope']
            deepening = n_points * cl.dx * map_dx * cf_slope

            line = calving_glacier_downstream_line(line, n_points=n_points)
            bed_shape = np.append(bed_shape, np.zeros(n_points))
            lambdas = np.append(lambdas, np.zeros(n_points))
            section = np.append(section, np.zeros(n_points))
            # The bed slowly deepens
            bed_down = np.linspace(bed_h[-1], bed_h[-1]-deepening, n_points)
            bed_h = np.append(bed_h, bed_down)
            surface_h = np.append(surface_h, bed_down)
            widths_m = np.append(widths_m,
                                 np.zeros(n_points) + np.mean(widths_m[-5:]))

        nfl = MixedBedFlowline(line=line, dx=cl.dx, map_dx=map_dx,
                               surface_h=surface_h, bed_h=bed_h,
                               section=section, bed_shape=bed_shape,
                               is_trapezoid=np.isfinite(lambdas),
                               lambdas=lambdas,
                               widths_m=widths_m,
                               rgi_id=cl.rgi_id,
                               gdir=gdir)

        # Update attrs
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
    gdir.write_pickle(new_fls, 'model_flowlines', filesuffix=filesuffix)


def decide_evolution_model(evolution_model=None):
    """Simple utility to check and apply user choices in cfg.PARAMS"""

    if evolution_model is not None:
        return evolution_model

    from_cfg = cfg.PARAMS['evolution_model'].lower()
    if from_cfg == 'SemiImplicit'.lower():
        evolution_model = SemiImplicitModel
    elif from_cfg == 'FluxBased'.lower():
        evolution_model = FluxBasedModel
    elif from_cfg == 'MassRedistributionCurve'.lower():
        evolution_model = MassRedistributionCurveModel
    else:
        raise InvalidParamsError("PARAMS['evolution_model'] not recognized"
                                 f": {from_cfg}.")
    return evolution_model


@entity_task(log)
def flowline_model_run(gdir, output_filesuffix=None, mb_model=None,
                       ys=None, ye=None, zero_initial_glacier=False,
                       init_model_fls=None, store_monthly_step=False,
                       glen_a_fac=None, fs_fac=None,
                       fixed_geometry_spinup_yr=None,
                       store_model_geometry=None,
                       store_fl_diagnostics=None,
                       water_level=None,
                       evolution_model=None, stop_criterion=None,
                       init_model_filesuffix=None, init_model_yr=None,
                       **kwargs):
    """Runs a model simulation with the default time stepping scheme.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    output_filesuffix : str
        this adds a suffix to the output file (useful to avoid overwriting
        previous experiments)
    mb_model : :py:class:`core.MassBalanceModel`
        a MassBalanceModel instance
    ys : int
        start year of the model run (needs to be set)
    ye : int
        end year of the model run (need to be set)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    glen_a_fac : float
        perturbate the default glen_a (either from the inversion if
        PARAMS['use_inversion_params_for_run'] is True or PARAMS['glen_a'] if
        False) with a chosen multiplicative parameter. Useful for
        parameter perturbation experiments.
    fs_fac : float
        same as glen_a_fac but for sliding
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        The default is to take the water level obtained from the ice
        thickness inversion.
    stop_criterion : func
        a function which decides on when to stop the simulation. See
        `run_until_and_store` documentation for more information.
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
     """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls

        if ys != init_model_yr and ys != 0:
            log.warning(f"You are starting a run with ys={ys} "
                          f"and init_model_yr={init_model_yr}.")

    mb_elev_feedback = kwargs.get('mb_elev_feedback', 'annual')
    if store_monthly_step and (mb_elev_feedback == 'annual'):
        warnings.warn("The mass balance used to drive the ice dynamics model "
                      "is updated yearly. If you want the output to be stored "
                      "monthly and also reflect monthly processes, "
                      "set store_monthly_step=True and "
                      "mb_elev_feedback='monthly'. This is not recommended "
                      "though: for monthly MB applications, we recommend to "
                      "use the `run_with_hydro` task.")

    if cfg.PARAMS['use_inversion_params_for_run']:
        diag = gdir.get_diagnostics()
        fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
        glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
    else:
        fs = cfg.PARAMS['fs']
        glen_a = cfg.PARAMS['glen_a']

    if glen_a_fac is not None:
        glen_a *= glen_a_fac
    if fs_fac is not None:
        fs *= fs_fac

    kwargs.setdefault('fs', fs)
    kwargs.setdefault('glen_a', glen_a)

    if store_model_geometry is None:
        store_model_geometry = cfg.PARAMS['store_model_geometry']

    if store_fl_diagnostics is None:
        store_fl_diagnostics = cfg.PARAMS['store_fl_diagnostics']

    if store_model_geometry:
        geom_path = gdir.get_filepath('model_geometry',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    else:
        geom_path = False

    if store_fl_diagnostics:
        fl_diag_path = gdir.get_filepath('fl_diagnostics',
                                         filesuffix=output_filesuffix,
                                         delete=True)
    else:
        fl_diag_path = False

    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)

    if init_model_fls is None:
        fls = gdir.read_pickle('model_flowlines')
    else:
        fls = copy.deepcopy(init_model_fls)
    if zero_initial_glacier:
        for fl in fls:
            fl.thick = fl.thick * 0.

    evolution_model = decide_evolution_model(evolution_model)

    if (cfg.PARAMS['use_kcalving_for_run'] and gdir.is_tidewater and
            water_level is None):
        # check for water level
        water_level = gdir.get_diagnostics().get('calving_water_level', None)
        if water_level is None:
            raise InvalidWorkflowError('This tidewater glacier seems to not '
                                       'have been inverted with the '
                                       '`find_inversion_calving` task. Set '
                                       "PARAMS['use_kcalving_for_run'] to "
                                       '`False` or set `water_level` '
                                       'to prevent this error.')

    model = evolution_model(fls, mb_model=mb_model, y0=ys,
                            inplace=True,
                            is_tidewater=gdir.is_tidewater,
                            is_lake_terminating=gdir.is_lake_terminating,
                            water_level=water_level,
                            **kwargs)

    with warnings.catch_warnings():
        # For operational runs we ignore the warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        model.run_until_and_store(ye,
                                  geom_path=geom_path,
                                  diag_path=diag_path,
                                  fl_diag_path=fl_diag_path,
                                  store_monthly_step=store_monthly_step,
                                  fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
                                  stop_criterion=stop_criterion)

    return model


@entity_task(log)
def run_random_climate(gdir, nyears=1000, y0=None, halfsize=15,
                       ys=None, ye=None,
                       bias=0, seed=None, temperature_bias=None,
                       precipitation_factor=None,
                       store_monthly_step=False,
                       store_model_geometry=None,
                       store_fl_diagnostics=None,
                       mb_model_class=MonthlyTIModel,
                       climate_filename='climate_historical',
                       climate_input_filesuffix='',
                       output_filesuffix='', init_model_fls=None,
                       init_model_filesuffix=None,
                       init_model_yr=None,
                       zero_initial_glacier=False,
                       unique_samples=False, **kwargs):
    """Runs the random mass balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    ys : int, default: 0 or init_model_yr
        first year of the fake output timeseries. Since these simulations
        are idealized, the concept of "time" is only relative to the start of
        the simulation.
    ye : int, default: nyears
        can be used instead of "nyears"
    y0 : int
        central year of the random climate period. Has to be set!
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model (offset to add to the MB). Default is zero.
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed across glaciers can
        be useful if you want to have the same climate years for all of them
    temperature_bias : float
        add a bias to the temperature timeseries (note that this is added
        to any bias that the calibration decided is needed)
    precipitation_factor: float
        multiply a factor to the precipitation time series (note that
        this factor is multiplied to any factor that was decided during
        calibration or by global parameters)
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    mb_model_class : MassBalanceModel class
        The MassBalanceModel class to use inside the RandomMassBalance (default
        MonthlyTIModel)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    unique_samples: bool
        if true, chosen random mass balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability
    kwargs : dict
        kwargs to pass to the flowline_model_run task
    """

    mb_model = MultipleFlowlineMassBalance(gdir,
                                           mb_model_class=partial(
                                               RandomMassBalance,
                                               mb_model_class=mb_model_class),
                                           y0=y0, halfsize=halfsize,
                                           bias=bias, seed=seed,
                                           filename=climate_filename,
                                           input_filesuffix=climate_input_filesuffix,
                                           unique_samples=unique_samples)

    if temperature_bias is not None:
        mb_model.temp_bias += temperature_bias
    if precipitation_factor is not None:
        mb_model.prcp_fac *= precipitation_factor

    if ys is None:
        ys = init_model_yr if init_model_yr is not None else 0
    if ye is None:
        ye = ys + nyears

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb_model, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              store_fl_diagnostics=store_fl_diagnostics,
                              init_model_filesuffix=init_model_filesuffix,
                              init_model_yr=init_model_yr,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)


@entity_task(log)
def run_constant_climate(gdir, nyears=1000, y0=None, halfsize=15,
                         ys=None, ye=None,
                         bias=0, temperature_bias=None,
                         precipitation_factor=None,
                         store_monthly_step=False,
                         store_model_geometry=None,
                         store_fl_diagnostics=None,
                         init_model_filesuffix=None,
                         init_model_yr=None,
                         output_filesuffix='',
                         climate_filename='climate_historical',
                         mb_model_class=MonthlyTIModel,
                         climate_input_filesuffix='',
                         init_model_fls=None,
                         zero_initial_glacier=False,
                         **kwargs):
    """Runs the constant mass balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation (default: as long as needed for reaching
        equilibrium)
    ys : int, default: 0 or init_model_yr
        first year of the fake output timeseries. Since these simulations
        are idealized, the concept of "time" is only relative to the start of
        the simulation.
    ye : int, default: nyears
        can be used instead of "nyears"
    y0 : int
        central year of the random climate period. Has to be set!
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model (offset to add to the MB). Default is zero.
    temperature_bias : float
        add a bias to the temperature timeseries (note that this is added
        to any bias that the calibration decided is needed)
    precipitation_factor: float
        multiply a factor to the precipitation time series (note that
        this factor is multiplied to any factor that was decided during
        calibration or by global parameters)
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    mb_model_class : MassBalanceModel class
        The MassBalanceModel class to use inside the ConstantMassBalance
        (default MonthlyTIModel)
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_fls : []
        list of flowlines to use to initialize the model (the default is the
        present_time_glacier file from the glacier directory)
    kwargs : dict
        kwargs to pass to the flowline_model_run task
    """

    mb_model = MultipleFlowlineMassBalance(gdir,
                                           mb_model_class=partial(
                                               ConstantMassBalance,
                                               mb_model_class=mb_model_class),
                                           y0=y0, halfsize=halfsize,
                                           bias=bias,
                                           filename=climate_filename,
                                           input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mb_model.temp_bias += temperature_bias
    if precipitation_factor is not None:
        mb_model.prcp_fac *= precipitation_factor

    if ys is None:
        ys = init_model_yr if init_model_yr is not None else 0
    if ye is None:
        ye = ys + nyears

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb_model, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              store_fl_diagnostics=store_fl_diagnostics,
                              init_model_filesuffix=init_model_filesuffix,
                              init_model_yr=init_model_yr,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)


@entity_task(log)
def run_from_climate_data(gdir, ys=None, ye=None, min_ys=None, max_ys=None,
                          fixed_geometry_spinup_yr=None,
                          store_monthly_step=False,
                          store_model_geometry=None,
                          store_fl_diagnostics=None,
                          climate_filename='climate_historical',
                          mb_model=None,
                          mb_model_class=MonthlyTIModel,
                          climate_input_filesuffix='', output_filesuffix='',
                          init_model_filesuffix=None, init_model_yr=None,
                          init_model_fls=None, zero_initial_glacier=False,
                          bias=0, temperature_bias=None,
                          precipitation_factor=None, **kwargs):
    """ Runs a glacier with climate input from e.g. CRU or a GCM.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr)
    ye : int
        end year of the model run (default: last year of the provided
        climate file)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    max_ys : int
        if you want to impose a maximum start year, regardless if the glacier
        inventory date is later (e.g. if climate data does not reach).
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    mb_model : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance. Default is to use a
        mb_model_class instance (default MonthlyTIModel)
        together with the provided parameters climate_filename,
        bias and climate_input_filesuffix.
    mb_model_class : MassBalanceModel class
        the MassBalanceModel class to use, default is MonthlyTIModel
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    bias : float
        bias of the mb model (offset to add to the MB). Default is zero.
    temperature_bias : float
        add a bias to the temperature timeseries (note that this is added
        to any bias that the calibration decided is needed)
    precipitation_factor: float
        multiply a factor to the precipitation time series (note that
        this factor is multiplied to any factor that was decided during
        calibration or by global parameters)
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
    kwargs : dict
        kwargs to pass to the flowline_model_run task
    """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls
        if ys is None:
            ys = init_model_yr

    try:
        rgi_year = gdir.rgi_date.year
    except AttributeError:
        rgi_year = gdir.rgi_date

    # Take from rgi date if not set yet
    if ys is None:
        # See also: https://github.com/OGGM/oggm/issues/1020
        # Even in calendar dates, we prefer to start in the next year
        # as the rgi is often from snow free images the year before (e.g. Aug)
        ys = rgi_year + 1

    if ys <= rgi_year and init_model_filesuffix is None:
        log.warning('You are attempting to run_with_climate_data at dates '
                    'prior to the RGI inventory date. This may indicate some '
                    'problem in your workflow. Consider using '
                    '`fixed_geometry_spinup_yr` for example.')

    # Final crop
    if min_ys is not None:
        ys = ys if ys > min_ys else min_ys
    if max_ys is not None:
        ys = ys if ys < max_ys else max_ys

    if mb_model is None:
        mb_model = MultipleFlowlineMassBalance(gdir,
                                               mb_model_class=mb_model_class,
                                               filename=climate_filename,
                                               bias=bias,
                                               input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mb_model.temp_bias += temperature_bias
    if precipitation_factor is not None:
        mb_model.prcp_fac *= precipitation_factor

    if ye is None:
        # Decide from climate (we can run the last year with data as well)
        ye = mb_model.flowline_mb_models[0].ye + 1

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb_model, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              store_fl_diagnostics=store_fl_diagnostics,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
                              **kwargs)


@entity_task(log)
def run_with_hydro(gdir, run_task=None, store_monthly_hydro=False,
                   fixed_geometry_spinup_yr=None, ref_area_from_y0=False,
                   ref_area_yr=None, ref_geometry_filesuffix=None,
                   **kwargs):
    """Run the flowline model and add hydro diagnostics.

    TODOs:
        - Add the possibility to record MB during run to improve performance
          (requires change in API)
        - ...

    Parameters
    ----------
    run_task : func
        any of the `run_*`` tasks in the oggm.flowline module.
        The mass balance model used needs to have the `add_climate` output
        kwarg available though.
    store_monthly_hydro : bool
        also compute monthly hydrological diagnostics. The monthly outputs
        are stored in 2D fields (years, months)
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    ref_area_from_y0 : bool
        overwrite ref_area_yr to the first year of the timeseries
    ref_geometry_filesuffix : str
        this kwarg allows to copy the reference area from a previous simulation
        (useful for projections with historical spinup for example).
        Set to a model_geometry file filesuffix that is present in the
        current directory (e.g. `_historical` for pre-processed gdirs).
        If set, ref_area_yr and ref_area_from_y0 refer to this file instead.
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
    **kwargs : all valid kwargs for ``run_task``
    """

    # Make sure it'll return something
    kwargs['return_value'] = True

    # Check that kwargs and params are compatible
    if kwargs.get('store_monthly_step', False):
        raise InvalidParamsError('run_with_hydro only compatible with '
                                 'store_monthly_step=False.')
    if kwargs.get('mb_elev_feedback', 'annual') != 'annual':
        raise InvalidParamsError('run_with_hydro only compatible with '
                                 "mb_elev_feedback='annual' (yes, even "
                                 "when asked for monthly hydro output).")
    if not cfg.PARAMS['store_model_geometry']:
        raise InvalidParamsError('run_with_hydro only works with '
                                 "PARAMS['store_model_geometry'] = True "
                                 "for now.")

    if fixed_geometry_spinup_yr is not None:
        kwargs['fixed_geometry_spinup_yr'] = fixed_geometry_spinup_yr

    out = run_task(gdir, **kwargs)

    if out is None:
        raise InvalidWorkflowError('The run task ({}) did not run '
                                   'successfully.'.format(run_task.__name__))

    do_spinup = fixed_geometry_spinup_yr is not None
    if do_spinup:
        start_dyna_model_yr = out.y0

    # Mass balance model used during the run
    mb_mod = out.mb_model

    # Glacier geometry during the run
    suffix = kwargs.get('output_filesuffix', '')

    # We start by fetching the reference model geometry
    # The one we just computed
    fmod = FileModel(gdir.get_filepath('model_geometry', filesuffix=suffix))
    # The last one is the final state - we can't compute MB for that
    years = fmod.years[:-1]

    if ref_geometry_filesuffix:
        if not ref_area_from_y0 and ref_area_yr is None:
            raise InvalidParamsError('If `ref_geometry_filesuffix` is set, '
                                     'users need to specify `ref_area_from_y0`'
                                     ' or `ref_area_yr`')
        # User provided
        fmod_ref = FileModel(gdir.get_filepath('model_geometry',
                                               filesuffix=ref_geometry_filesuffix))
    else:
        # ours as well
        fmod_ref = fmod

    # Check input
    if ref_area_from_y0:
        ref_area_yr = fmod_ref.years[0]

    # Geometry at year yr to start with + off-glacier snow bucket
    if ref_area_yr is not None:
        if ref_area_yr not in fmod_ref.years:
            raise InvalidParamsError('The chosen ref_area_yr is not '
                                     'available!')
        fmod_ref.run_until(ref_area_yr)

    bin_area_2ds = []
    bin_elev_2ds = []
    ref_areas = []
    snow_buckets = []
    for fl in fmod_ref.fls:
        # Glacier area on bins
        bin_area = fl.bin_area_m2
        ref_areas.append(bin_area)
        snow_buckets.append(bin_area * 0)

        # Output 2d data
        shape = len(years), len(bin_area)
        bin_area_2ds.append(np.empty(shape, np.float64))
        bin_elev_2ds.append(np.empty(shape, np.float64))

    # Ok now fetch all geometry data in a first loop
    # We do that because we might want to get the largest possible area (default)
    # and we want to minimize the number of calls to run_until
    for i, yr in enumerate(years):
        fmod.run_until(yr)
        for fl_id, (fl, bin_area_2d, bin_elev_2d) in \
                enumerate(zip(fmod.fls, bin_area_2ds, bin_elev_2ds)):
            # Time varying bins
            bin_area_2d[i, :] = fl.bin_area_m2
            bin_elev_2d[i, :] = fl.surface_h

    if ref_area_yr is None:
        # Ok we get the max area instead
        for ref_area, bin_area_2d in zip(ref_areas, bin_area_2ds):
            ref_area[:] = bin_area_2d.max(axis=0)

    # Ok now we have arrays, we can work with that
    # -> second time varying loop is for mass balance
    months = [1]
    seconds = cfg.SEC_IN_YEAR
    ntime = len(years) + 1
    oshape = (ntime, 1)
    if store_monthly_hydro:
        months = np.arange(1, 13)
        seconds = cfg.SEC_IN_MONTH
        oshape = (ntime, 12)

    out = {
        'off_area': {
            'description': 'Off-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'on_area': {
            'description': 'On-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'melt_off_glacier': {
            'description': 'Off-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_on_glacier': {
            'description': 'On-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_off_glacier': {
            'description': 'Off-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_on_glacier': {
            'description': 'On-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_off_glacier': {
            'description': 'Off-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_on_glacier': {
            'description': 'On-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_off_glacier': {
            'description': 'Off-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_on_glacier': {
            'description': 'On-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snow_bucket': {
            'description': 'Off-glacier snow reservoir (state variable)',
            'unit': 'kg',
            'data': np.zeros(oshape),
        },
        'model_mb': {
            'description': 'Annual mass balance from dynamical model',
            'unit': 'kg yr-1',
            'data': np.zeros(ntime),
        },
        'residual_mb': {
            'description': 'Difference (before correction) between mb model and dyn model melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
    }

    # Initialize
    fmod.run_until(years[0])
    prev_model_vol = fmod.volume_m3

    for i, yr in enumerate(years):

        # Now the loop over the months
        for m in months:

            # A bit silly but avoid double counting in monthly ts
            off_area_out = 0
            on_area_out = 0

            for fl_id, (ref_area, snow_bucket, bin_area_2d, bin_elev_2d) in \
                    enumerate(zip(ref_areas, snow_buckets, bin_area_2ds, bin_elev_2ds)):

                bin_area = bin_area_2d[i, :]
                bin_elev = bin_elev_2d[i, :]

                # Make sure we have no negative contribution when glaciers are out
                off_area = utils.clip_min(ref_area - bin_area, 0)

                try:
                    if store_monthly_hydro:
                        flt_yr = utils.date_to_floatyear(int(yr), m)
                        mb_out = mb_mod.get_monthly_mb(bin_elev, fl_id=fl_id,
                                                       year=flt_yr,
                                                       add_climate=True)
                        mb, _, _, prcp, prcpsol = mb_out
                    else:
                        mb_out = mb_mod.get_annual_mb(bin_elev, fl_id=fl_id,
                                                      year=yr, add_climate=True)
                        mb, _, _, prcp, prcpsol = mb_out
                except ValueError as e:
                    if 'too many values to unpack' in str(e):
                        raise InvalidWorkflowError('Run with hydro needs a MB '
                                                   'model able to add climate '
                                                   'info to `get_annual_mb`.')
                    raise

                # Here we use mass (kg yr-1) not ice volume
                mb *= seconds * cfg.PARAMS['ice_density']

                # Bias of the mb model is a fake melt term that we need to deal with
                # This is here for correction purposes later
                mb_bias = mb_mod.bias * seconds / cfg.SEC_IN_YEAR

                liq_prcp_on_g = (prcp - prcpsol) * bin_area
                liq_prcp_off_g = (prcp - prcpsol) * off_area

                prcpsol_on_g = prcpsol * bin_area
                prcpsol_off_g = prcpsol * off_area

                # IMPORTANT: this does not guarantee that melt cannot be negative
                # the reason is the MB residual that here can only be understood
                # as a fake melt process.
                # In particular at the monthly scale this can lead to negative
                # or winter positive melt - we try to mitigate this
                # issue at the end of the year
                melt_on_g = (prcpsol - mb) * bin_area
                melt_off_g = (prcpsol - mb) * off_area

                if mb_mod.bias == 0:
                    # melt_on_g and melt_off_g can be negative, but the absolute
                    # values are very small. so we clip them to zero
                    melt_on_g = utils.clip_min(melt_on_g, 0)
                    melt_off_g = utils.clip_min(melt_off_g, 0)

                # This is the bad boy
                bias_on_g = mb_bias * bin_area
                bias_off_g = mb_bias * off_area

                # Update bucket with accumulation and melt
                snow_bucket += prcpsol_off_g
                # It can only melt that much
                melt_off_g = np.where((snow_bucket - melt_off_g) >= 0, melt_off_g, snow_bucket)
                # Update bucket
                snow_bucket -= melt_off_g

                # This is recomputed each month but well
                off_area_out += np.sum(off_area)
                on_area_out += np.sum(bin_area)

                # Monthly out
                out['melt_off_glacier']['data'][i, m-1] += np.sum(melt_off_g)
                out['melt_on_glacier']['data'][i, m-1] += np.sum(melt_on_g)
                out['melt_residual_off_glacier']['data'][i, m-1] += np.sum(bias_off_g)
                out['melt_residual_on_glacier']['data'][i, m-1] += np.sum(bias_on_g)
                out['liq_prcp_off_glacier']['data'][i, m-1] += np.sum(liq_prcp_off_g)
                out['liq_prcp_on_glacier']['data'][i, m-1] += np.sum(liq_prcp_on_g)
                out['snowfall_off_glacier']['data'][i, m-1] += np.sum(prcpsol_off_g)
                out['snowfall_on_glacier']['data'][i, m-1] += np.sum(prcpsol_on_g)

                # Snow bucket is a state variable - stored at end of timestamp
                if store_monthly_hydro:
                    if m == 12:
                        out['snow_bucket']['data'][i+1, 0] += np.sum(snow_bucket)
                    else:
                        out['snow_bucket']['data'][i, m] += np.sum(snow_bucket)
                else:
                    out['snow_bucket']['data'][i+1, m-1] += np.sum(snow_bucket)

        # Update the annual data
        out['off_area']['data'][i] = off_area_out
        out['on_area']['data'][i] = on_area_out

        # If monthly, put the residual where we can
        if store_monthly_hydro and mb_mod.bias != 0:
            for melt, bias in zip(
                    [
                        out['melt_on_glacier']['data'][i, :],
                        out['melt_off_glacier']['data'][i, :],
                    ],
                    [
                        out['melt_residual_on_glacier']['data'][i, :],
                        out['melt_residual_off_glacier']['data'][i, :],
                    ],
            ):

                real_melt = melt - bias
                to_correct = utils.clip_min(real_melt, 0)
                to_correct_sum = np.sum(to_correct)
                if (to_correct_sum > 1e-7) and (np.sum(melt) > 0):
                    # Ok we correct the positive melt instead
                    fac = np.sum(melt) / to_correct_sum
                    melt[:] = to_correct * fac

        if do_spinup and yr < start_dyna_model_yr:
            residual_mb = 0
            model_mb = (out['snowfall_on_glacier']['data'][i, :].sum() -
                        out['melt_on_glacier']['data'][i, :].sum())
        else:
            # Correct for mass-conservation and match the ice-dynamics model
            fmod.run_until(yr + 1)
            model_mb = (fmod.volume_m3 - prev_model_vol) * cfg.PARAMS['ice_density']
            prev_model_vol = fmod.volume_m3

            reconstructed_mb = (out['snowfall_on_glacier']['data'][i, :].sum() -
                                out['melt_on_glacier']['data'][i, :].sum())
            residual_mb = model_mb - reconstructed_mb

        # Now correct
        g_melt = out['melt_on_glacier']['data'][i, :]
        if residual_mb == 0:
            pass
        elif store_monthly_hydro:
            # We try to correct the melt only where there is some
            asum = g_melt.sum()
            if asum > 1e-7 and (residual_mb / asum < 1):
                # try to find a fac
                fac = 1 - residual_mb / asum
                # fac should always be positive, otherwise melt_on_glacier
                # gets negative. This is always true, because we only do the
                # correction if residual_mb / asum < 1
                corr = g_melt * fac
                residual_mb = g_melt - corr
                out['melt_on_glacier']['data'][i, :] = corr
            else:
                # We simply spread over the months
                residual_mb /= 12
                # residual_mb larger > melt_on_glacier
                # add absolute difference to snowfall (--=+, mass conservation)
                out['snowfall_on_glacier']['data'][i, :] -= utils.clip_max(g_melt - residual_mb, 0)
                # assure that new melt_on_glacier is non-negative
                out['melt_on_glacier']['data'][i, :] = utils.clip_min(g_melt - residual_mb, 0)
        else:
            # We simply apply the residual - no choice here
            # residual_mb larger > melt_on_glacier:
            # add absolute difference to snowfall (--=+, mass conservation)
            out['snowfall_on_glacier']['data'][i, :] -= utils.clip_max(g_melt - residual_mb, 0)
            # assure that new melt_on_glacier is non-negative
            out['melt_on_glacier']['data'][i, :] = utils.clip_min(g_melt - residual_mb, 0)

        out['model_mb']['data'][i] = model_mb
        out['residual_mb']['data'][i] = residual_mb

    # Convert to xarray
    out_vars = cfg.PARAMS['store_diagnostic_variables']
    ods = xr.Dataset()
    ods.coords['time'] = fmod.years
    if store_monthly_hydro:
        ods.coords['month_2d'] = ('month_2d', np.arange(1, 13))
        # For the user later
        sm = cfg.PARAMS['hydro_month_' + mb_mod.hemisphere]
        ods.coords['hydro_month_2d'] = ('month_2d', (np.arange(12) + 12 - sm + 1) % 12 + 1)
        ods.coords['calendar_month_2d'] = ('month_2d', np.arange(1, 13))
    for varname, d in out.items():
        data = d.pop('data')
        if varname not in out_vars:
            continue
        if len(data.shape) == 2:
            # First the annual agg
            if varname == 'snow_bucket':
                # Snowbucket is a state variable
                ods[varname] = ('time', data[:, 0])
            else:
                # Last year is never good
                data[-1, :] = np.nan
                ods[varname] = ('time', np.sum(data, axis=1))
            # Then the monthly ones
            if store_monthly_hydro:
                ods[varname + '_monthly'] = (('time', 'month_2d'), data)
        else:
            assert varname != 'snow_bucket'
            data[-1] = np.nan
            ods[varname] = ('time', data)
        for k, v in d.items():
            ods[varname].attrs[k] = v
            if store_monthly_hydro and (varname + '_monthly') in ods:
                ods[varname + '_monthly'].attrs[k] = v

    # Append the output to the existing diagnostics
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=suffix)
    ods.to_netcdf(fpath, mode='a')


def zero_glacier_stop_criterion(model, state, n_zero=5, n_years=20):
    """Stop the simulation when the glacier volume is zero for a given period

    To be passed as kwarg to `run_until_and_store`.

    Parameters
    ----------
    model : the model class
    state : a dict
    n_zero : number of 0 volume years
    n_years : number of years to consider

    Returns
    -------
    stop (True/False), new_state
    """

    if state is None:
        # OK first call
        state = {}

    if 'was_zero' not in state:
        # Maybe the state is from another criteria
        state['was_zero'] = []

    if model.yr != int(model.yr):
        # We consider only full model years
        return False, state

    if model.volume_m3 == 0:
        if len(state['was_zero']) < n_years:
            return False, state
        if np.sum(state['was_zero'][-n_years:]) >= n_zero - 1:
            return True, state
        else:
            state['was_zero'] = np.append(state['was_zero'], [True])
    else:
        state['was_zero'] = np.append(state['was_zero'], [False])

    return False, state


def spec_mb_stop_criterion(model, state, spec_mb_threshold=50, n_years=60):
    """Stop the simulation when the specific MB is close to zero for a given period.

    To be passed as kwarg to `run_until_and_store`.

    Parameters
    ----------
    model : the model class
    state : a dict
    spec_mb_threshold : the specific MB threshold (in mm w.e. per year)
    n_years : number of years to consider

    Returns
    -------
    stop (True/False), new_state
    """

    if state is None:
        # OK first call
        state = {}

    if 'spec_mb' not in state:
        # Maybe the state is from another criteria
        state['spec_mb'] = []
        state['volume_m3'] = []

    if model.yr != int(model.yr):
        # We consider only full model years
        return False, state

    area = model.area_m2
    volume = model.volume_m3
    if area < 1 or len(state['volume_m3']) == 0:
        spec_mb = np.nan
    else:
        spec_mb = (volume - state['volume_m3'][-1]) / area * cfg.PARAMS['ice_density']

    state['spec_mb'] = np.append(state['spec_mb'], [spec_mb])
    state['volume_m3'] = np.append(state['volume_m3'], [volume])

    if len(state['spec_mb']) < n_years:
        return False, state

    mbavg = np.nanmean(state['spec_mb'][-n_years:])
    if abs(mbavg) <= spec_mb_threshold:
        return True, state
    else:
        return False, state


def equilibrium_stop_criterion(model, state,
                               spec_mb_threshold=50, n_years_specmb=60,
                               n_zero=5, n_years_zero=20):
    """Stop the simulation when of og spec_mb and zero_volume criteria are met.

    To be passed as kwarg to `run_until_and_store`.

    Parameters
    ----------
    model : the model class
    state : a dict
    spec_mb_threshold : the specific MB threshold (in mm w.e. per year)
    n_years_specmb : number of years to consider for the spec_mb criterion
    n_zero : number of 0 volume years
    n_years_zero : number of years to consider for the zero volume criterion.

    Returns
    -------
    stop (True/False), new_state
    """

    if state is None:
        # OK first call
        state = {}
    s1, state = zero_glacier_stop_criterion(model, state, n_years=n_years_zero,
                                            n_zero=n_zero)
    s2, state = spec_mb_stop_criterion(model, state, n_years=n_years_specmb,
                                       spec_mb_threshold=spec_mb_threshold)
    return s1 or s2, state


def merge_to_one_glacier(main, tribs, filename='climate_historical',
                         input_filesuffix=''):
    """Merge multiple tributary glacier flowlines to a main glacier

    This function will merge multiple tributary glaciers to a main glacier
    and write modified `model_flowlines` to the main GlacierDirectory.
    The provided tributaries must have an intersecting downstream line.
    To be sure about this, use `intersect_downstream_lines` first.
    This function is mainly responsible to reproject the flowlines, set
    flowline attributes and to copy additional files, like the necessary climate
    files.

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

    # read flowlines of the Main glacier
    fls = main.read_pickle('model_flowlines')
    mfl = fls.pop(-1)  # remove main line from list and treat separately

    for trib in tribs:

        # read tributary flowlines and append to list
        tfls = trib.read_pickle('model_flowlines')

        # copy climate file and calib to new gdir
        # if we have a merge-merge situation we need to copy multiple files
        rgiids = set([fl.rgi_id for fl in tfls])

        for uid in rgiids:
            if len(rgiids) == 1:
                # we do not have a merge-merge situation
                in_id = ''
                out_id = trib.rgi_id
            else:
                in_id = '_' + uid
                out_id = uid

            climfile_in = filename + in_id + input_filesuffix + '.nc'
            climfile_out = filename + '_' + out_id + input_filesuffix + '.nc'
            shutil.copyfile(os.path.join(trib.dir, climfile_in),
                            os.path.join(main.dir, climfile_out))

            _m = os.path.basename(trib.get_filepath('mb_calib')).split('.')
            muin = _m[0] + in_id + '.' + _m[1]
            muout = _m[0] + '_' + out_id + '.' + _m[1]

            shutil.copyfile(os.path.join(trib.dir, muin),
                            os.path.join(main.dir, muout))

        # sort flowlines descending
        tfls.sort(key=lambda x: x.order, reverse=True)

        # loop over tributaries and reproject to main glacier
        for nr, tfl in enumerate(tfls):

            # 1. Step: Change projection to the main glaciers grid
            _line = salem.transform_geometry(tfl.line,
                                             crs=trib.grid, to_crs=main.grid)

            # 2. set new line
            tfl.set_line(_line)

            # 3. set map attributes
            dx = [shpg.Point(tfl.line.coords[i]).distance(
                shpg.Point(tfl.line.coords[i+1]))
                for i, pt in enumerate(tfl.line.coords[:-1])]  # get distance
            # and check if equally spaced
            if not np.allclose(dx, np.mean(dx), atol=1e-2):
                raise RuntimeError('Flowline is not evenly spaced.')

            tfl.dx = np.mean(dx).round(2)
            tfl.map_dx = mfl.map_dx
            tfl.dx_meter = tfl.map_dx * tfl.dx

            # 3. remove attributes, they will be set again later
            tfl.inflow_points = []
            tfl.inflows = []

            # 4. set flows to, mainly to update flows_to_point coordinates
            if tfl.flows_to is not None:
                tfl.set_flows_to(tfl.flows_to)

        # append tributary flowlines to list
        fls += tfls

    # add main flowline to the end
    fls = fls + [mfl]

    # Finally write the flowlines
    main.write_pickle(fls, 'model_flowlines')


def clean_merged_flowlines(gdir, buffer=None):
    """Order and cut merged flowlines to size.

    After matching flowlines were found and merged to one glacier directory
    this function makes them nice:
    There should only be one flowline per bed, so overlapping lines have to be
    cut, attributed to a another flowline and ordered.

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        The GDir of the glacier of interest
    buffer: float
        Buffer around the flowlines to find overlaps
    """

    # No buffer does not work
    if buffer is None:
        buffer = cfg.PARAMS['kbuffer']

    # Number of pixels to arbitrarily remove at junctions
    lid = int(cfg.PARAMS['flowline_junction_pix'])

    fls = gdir.read_pickle('model_flowlines')

    # separate the main main flowline
    mainfl = fls.pop(-1)

    # split fls in main and tribs
    mfls = [fl for fl in fls if fl.flows_to is None]
    tfls = [fl for fl in fls if fl not in mfls]

    # --- first treat the main flowlines ---
    # sort by order and length as a second choice
    mfls.sort(key=lambda x: (x.order, len(x.inflows), x.length_m),
              reverse=False)

    merged = []

    # for fl1 in mfls:
    while len(mfls) > 0:
        fl1 = mfls.pop(0)

        ol_index = []  # list of index from first overlap

        # loop over other main lines and main main line
        for fl2 in mfls + [mainfl]:

            # calculate overlap, maybe use larger buffer here only to find it
            _overlap = fl1.line.intersection(fl2.line.buffer(buffer*2))

            # calculate indice of first overlap if overlap length > 0
            oix = 9999
            if _overlap.length > 0 and fl1 != fl2 and fl2.flows_to != fl1:
                if isinstance(_overlap, shpg.MultiLineString):
                    if _overlap.geoms[0].coords[0] == fl1.line.coords[0]:
                        # if the head of overlap is same as the first line,
                        # best guess is, that the heads are close topgether!
                        _ov1 = _overlap.geoms[1].coords[1]
                    else:
                        _ov1 = _overlap.geoms[0].coords[1]
                else:
                    _ov1 = _overlap.coords[1]
                for _i, _p in enumerate(fl1.line.coords):
                    if _p == _ov1:
                        oix = _i
                # low indices are more likely due to an wrong overlap
                if oix < 10:
                    oix = 9999
            ol_index.append(oix)

        ol_index = np.array(ol_index)
        if np.all(ol_index == 9999):
            log.warning('Glacier %s could not be merged, removed!' %
                        fl1.rgi_id)
            # remove possible tributary flowlines
            tfls = [fl for fl in tfls if fl.rgi_id != fl1.rgi_id]
            # skip rest of this while loop
            continue

        # make this based on first overlap, but consider order and or length
        minx = ol_index[ol_index <= ol_index.min()+10][-1]
        i = np.where(ol_index == minx)[0][-1]
        _olline = (mfls + [mainfl])[i]

        # 1. cut line to size
        _line = fl1.line

        bufferuse = buffer
        while bufferuse > 0:
            _overlap = _line.intersection(_olline.line.buffer(bufferuse))
            _linediff = _line.difference(_overlap)  # cut to new line

            # if the tributary flowline is longer than the main line,
            # _line will contain multiple LineStrings: only keep the first
            if isinstance(_linediff, shpg.MultiLineString):
                _linediff = _linediff.geoms[0]

            if len(_linediff.coords) < 10:
                bufferuse -= 1
            else:
                break

        if bufferuse <= 0:
            log.warning('Glacier %s would be to short after merge, removed!' %
                        fl1.rgi_id)
            # remove possible tributary flowlines
            tfls = [fl for fl in tfls if fl.rgi_id != fl1.rgi_id]
            # skip rest of this while loop
            continue

        # remove cfg.PARAMS['flowline_junction_pix'] from the _line
        # gives a bigger gap at the junction and makes sure the last
        # point is not corrupted in terms of spacing
        _line = shpg.LineString(_linediff.coords[:-lid])

        # 2. set new line
        fl1.set_line(_line)

        # 3. set flow to attributes. This also adds inflow values to other
        fl1.set_flows_to(_olline)

        # change the array size of tributary flowline attributes
        for atr, value in fl1.__dict__.items():
            if atr in ['_ptrap', '_prec']:
                # those are indices, remove those above nx
                fl1.__setattr__(atr, value[value < fl1.nx])
            elif isinstance(value, np.ndarray) and (len(value) > fl1.nx):
                # those are actual parameters on the grid
                fl1.__setattr__(atr, value[:fl1.nx])

        merged.append(fl1)
    allfls = merged + tfls

    # now check all lines for possible cut offs
    for fl in allfls:
        try:
            fl.flows_to_indice
        except AssertionError:
            mfl = fl.flows_to
            # remove it from original
            mfl.inflow_points.remove(fl.flows_to_point)
            mfl.inflows.remove(fl)

            prdis = mfl.line.project(fl.tail)
            mfl_keep = mfl

            while mfl.flows_to is not None:
                prdis2 = mfl.flows_to.line.project(fl.tail)
                if prdis2 < prdis:
                    mfl_keep = mfl
                    prdis = prdis2
                mfl = mfl.flows_to

            # we should be good to add this line here
            fl.set_flows_to(mfl_keep.flows_to)

    allfls = allfls + [mainfl]

    for fl in allfls:
        fl.inflows = []
        fl.inflow_points = []
        if hasattr(fl, '_lazy_flows_to_indice'):
            delattr(fl, '_lazy_flows_to_indice')
        if hasattr(fl, '_lazy_inflow_indices'):
            delattr(fl, '_lazy_inflow_indices')

    for fl in allfls:
        if fl.flows_to is not None:
            fl.set_flows_to(fl.flows_to)

    for fl in allfls:
        fl.order = line_order(fl)

    # order flowlines in descending way
    allfls.sort(key=lambda x: x.order, reverse=False)

    # assert last flowline is main flowline
    assert allfls[-1] == mainfl

    # Finally write the flowlines
    gdir.write_pickle(allfls, 'model_flowlines')
