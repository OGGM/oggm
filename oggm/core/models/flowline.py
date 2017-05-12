"""Flowline modelling: bed shapes and model numerics.


"""
from __future__ import division
from six.moves import zip

# Built ins
import logging
import warnings
import copy
from functools import partial
from collections import OrderedDict

# External libs
import numpy as np
import netCDF4
from scipy.interpolate import RegularGridInterpolator
import shapely.geometry as shpg
import xarray as xr

# Locals
import oggm.cfg as cfg
from oggm import utils
import oggm.core.preprocessing.geometry
import oggm.core.preprocessing.centerlines
import oggm.core.models.massbalance as mbmods
from oggm import entity_task
from oggm.core.preprocessing.centerlines import Centerline

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, TWO_THIRDS
from oggm.cfg import RHO, G, N, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)


class ModelFlowline(Centerline):
    """The is the input flowline for the model."""

    def __init__(self, line=None, dx=1, map_dx=None,
                 surface_h=None, bed_h=None):
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

        super(ModelFlowline, self).__init__(line, dx, surface_h)

        self._thick = (surface_h - bed_h).clip(0.)
        self.map_dx = map_dx
        self.dx_meter = map_dx * self.dx
        self.bed_h = bed_h

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
        ds['coords'] = (['x', 'c'], np.asarray(self.line.coords))
        ds['surface_h'] = (['x'],  h)
        ds['bed_h'] = (['x'],  self.bed_h)
        ds.attrs['class'] = type(self).__name__
        ds.attrs['map_dx'] = self.map_dx
        ds.attrs['dx'] = self.dx
        self._add_attrs_to_dataset(ds)
        return ds


class ParabolicFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, bed_shape=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(ParabolicFlowline, self).__init__(line, dx, map_dx,
                                                surface_h, bed_h)

        assert np.all(np.isfinite(bed_shape))
        self.bed_shape = bed_shape

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return np.sqrt(4*self.thick/self.bed_shape)

    @property
    def section(self):
        return TWO_THIRDS * self.widths_m * self.thick

    @section.setter
    def section(self, val):
        self.thick = (0.75 * val * np.sqrt(self.bed_shape))**TWO_THIRDS

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['bed_shape'] = (['x'],  self.bed_shape)


class VerticalWallFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, widths=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(VerticalWallFlowline, self).__init__(line, dx, map_dx,
                                                   surface_h, bed_h)

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


class TrapezoidalFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, widths=None, lambdas=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(TrapezoidalFlowline, self).__init__(line, dx, map_dx,
                                                   surface_h, bed_h)

        self._w0_m = widths * self.map_dx - lambdas * self.thick
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
        self.thick = (np.sqrt(b**2 + 4 * a * val) - b) / a

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'],  self.widths)
        ds['lambdas'] = (['x'],  self._lambdas)


class MixedFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None,
                 section=None, bed_shape=None,
                 where_trapezoid=None, lambdas=None,
                 widths_m=None):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """

        self._do_trapeze = False
        super(MixedFlowline, self).__init__(line, dx, map_dx,
                                            surface_h.copy(), bed_h.copy())

        # The parabola is easy
        assert np.all(np.isfinite(bed_shape))
        self.bed_shape = bed_shape
        self._sqrt_bed = np.sqrt(bed_shape)

        # where to trapeze
        if len(np.asarray(where_trapezoid).shape) == 2:
            where_trapezoid = where_trapezoid[0]
        ptrap = where_trapezoid
        ref_section = section.copy()
        _w0_m = widths_m[ptrap]

        sec = section[ptrap]
        b = - 2 * widths_m[ptrap]
        det = b ** 2 - 4 * lambdas[ptrap] * 2 * sec
        h = (- b - np.sqrt(det)) / (2 * lambdas[ptrap])
        assert np.alltrue(h >= 0)
        self.bed_h[ptrap] = surface_h[ptrap] - h.clip(0)
        self._thick[ptrap] = h.clip(0)

        # This doesnt work because of interp at the tongue
        # assert np.allclose(surface_h, self.surface_h)

        # compute _w0_m
        _w0_m = _w0_m - lambdas[ptrap] * self.thick[ptrap]
        assert np.alltrue(_w0_m >= 0)

        self._w0_m = _w0_m
        self._lambdas = lambdas
        self._ptrap = ptrap
        self._do_trapeze = len(sec) > 0
        assert np.allclose(self.thick[ptrap], h.clip(0))
        assert np.allclose(ref_section, self.section)

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        out = np.sqrt(4*self.thick/self.bed_shape)
        if self._do_trapeze:
            out[self._ptrap] = self._w0_m + self._lambdas[self._ptrap] * self.thick[self._ptrap]
        return out

    @property
    def section(self):
        out = TWO_THIRDS * self.widths_m * self.thick
        if self._do_trapeze:
            out[self._ptrap] = (self.widths_m[self._ptrap] + self._w0_m) / 2 * self.thick[self._ptrap]
        return out

    @section.setter
    def section(self, val):
        out = (0.75 * val * self._sqrt_bed)**TWO_THIRDS
        if self._do_trapeze:
            b = 2 * self._w0_m
            a = 2 * self._lambdas[self._ptrap]
            out[self._ptrap] = (np.sqrt(b ** 2 + 4 * a * val[self._ptrap]) - b) / a
        self.thick = out

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""

        ds['section'] = (['x'],  self.section)
        ds['widths_m'] = (['x'],  self.widths_m)
        ds['bed_shape'] = (['x'],  self.bed_shape)
        ds['where_trapezoid'] = (['p'], self._ptrap)
        ds['lambdas'] = (['x'],  self._lambdas)


class FlowlineModel(object):
    """Interface to the actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                       fs=0., fd=None, inplace=True):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        if not inplace:
            flowlines = copy.deepcopy(flowlines)

        try:
            _ = len(flowlines)
        except TypeError:
            flowlines = [flowlines]

        self.mb = mb_model
        self.fs = fs

        # for backwards compatibility I calculate fd from glen_a and
        # throw a Deprecation warning
        if fd is not None and glen_a is None:
            self.fd_deprecated = fd
            glen_a = (N+2) * fd / 2.
            warnings.warn(DeprecationWarning('the use of fd is deprecated'))

        # Defaults
        if glen_a is None:
            glen_a = cfg.A
        self.glen_a = glen_a

        # we keep glen_a as input, but for optimisation we stick to "fd"
        self._fd = 2. / (N+2) * self.glen_a

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.fls = None
        self._trib = None
        self.reset_flowlines(flowlines)

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_flowlines(self, flowlines):
        """Reset the initial model flowlines"""

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

    def to_netcdf(self, path):
        """Creates a netcdf group file storing the state of the model."""

        flows_to_id = []
        for trib in self._trib:
            flows_to_id.append(trib[0] if trib[0] is not None else -1)

        ds = xr.Dataset()
        ds['flowlines'] = ('flowlines', np.arange(len(flows_to_id)))
        ds['flows_to_id'] = ('flowlines', flows_to_id)
        ds.to_netcdf(path)
        for i, fl in enumerate(self.fls):
            ds = fl.to_dataset()
            ds.to_netcdf(path, 'a', group='fl_{}'.format(i))

    def check_domain_end(self):
        """Returns False if the glacier reaches the domains bound."""
        return np.isclose(self.fls[-1].thick[-1], 0)

    def step(self, dt=None):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1):

        t = (y1-self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(dt=t-self.t)

        # Check for domain bounds
        if self.fls[-1].thick[-1] > 10:
            raise RuntimeError('Glacier exceeds domain boundaries.')

        # Check for NaNs
        for fl in self.fls:
            if np.any(~np.isfinite(fl.thick)):
                raise RuntimeError('NaN in numerical solution.')

    def run_until_and_store(self, y1, path=None):
        """Runs the model and returns intermediate steps in a dataset.

        You can store the whole in a netcdf file, too.
        """

        # time
        time = utils.monthly_timeseries(self.yr, y1)
        yr, mo = utils.year_to_date(time)

        # init output
        sects = [(np.zeros((len(time), fl.nx)) * np.NaN) for fl in self.fls]
        widths = [(np.zeros((len(time), fl.nx)) * np.NaN) for fl in self.fls]
        if path is not None:
            self.to_netcdf(path)

        # Run
        for i, y in enumerate(time):
            self.run_until(y)
            for s, w, fl in zip(sects, widths, self.fls):
                s[i, :] = fl.section
                w[i, :] = fl.widths_m

        # to datasets
        dss = []
        for (s, w) in zip(sects, widths):
            ds = xr.Dataset()
            ds.coords['time'] = time
            varcoords = {'time': time,
                         'year': ('time', yr),
                         'month': ('time', mo),
                         }
            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                            coords=varcoords)
            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                            coords=varcoords)
            dss.append(ds)

        # write output?
        if path is not None:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(dss):
                ds.to_netcdf(path, 'a', group='fl_{}'.format(i),
                             encoding=encode)

        return dss

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):

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


class FluxBasedModelDeprecated(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., fd=None, fixed_dt=None, min_dt=SEC_IN_DAY,
                 max_dt=31*SEC_IN_DAY, inplace=True):

        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """
        super(FluxBasedModelDeprecated, self).__init__(flowlines,
                                                       mb_model=mb_model,
                                                       y0=y0, glen_a=glen_a,
                                                       fs=fs, fd=fd,
                                                       inplace=inplace)
        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt=31*SEC_IN_DAY):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

        # Loop over tributaries to determine the flux rate
        flxs = []
        aflxs = []
        for fl, trib in zip(self.fls, self._trib):

            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            nx = fl.nx

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid points
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
                nx = nx + 1

            dx = fl.dx_meter

            # Staggered gradient
            slope_stag = np.zeros(nx+1)
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

            # Staggered thick
            thick_stag = np.zeros(nx+1)
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            rhogh = (RHO*G*slope_stag)**N
            u_stag = (thick_stag**(N+1)) * self._fd * rhogh + \
                     (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag = np.zeros(nx+1)
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flx_stag = u_stag * section_stag / dx

            # Store the results
            if is_trib:
                flxs.append(flx_stag[:-1])
                aflxs.append(np.zeros(nx-1))
                u_stag = u_stag[:-1]
            else:
                flxs.append(flx_stag)
                aflxs.append(np.zeros(nx))

            # Time crit: limit the velocity somehow
            maxu = np.max(np.abs(u_stag))
            if maxu > 0.:
                _dt = 1./60. * dx / maxu
            else:
                _dt = self.max_dt
            if _dt < dt:
                dt = _dt

        # Time step
        if dt < min_dt:
            if not self.dt_warning:
                log.warning('Unstable')
            self.dt_warning = True
        dt = np.clip(dt, min_dt, self.max_dt)

        # A second loop for the mass exchange
        for fl, flx_stag, aflx, trib in zip(self.fls, flxs, aflxs,
                                                 self._trib):

            # Mass balance
            widths = fl.widths_m
            mb = self.mb.get_mb(fl.surface_h, self.yr)
            # Allow parabolic beds to grow
            widths = np.where((mb > 0.) & (widths == 0), 10., widths)
            mb = dt * mb * widths

            # Update section with flowing and mass balance
            new_section = fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt + \
                          aflx*dt + mb

            # Keep positive values only and store
            fl.section = new_section.clip(0)

            # Add the last flux to the tributary
            # this is ok because the lines are sorted in order
            if trib[0] is not None:
                aflxs[trib[0]][trib[1]:trib[2]] += flx_stag[-1].clip(0) * \
                                                   trib[3]

        # Next step
        self.t += dt


class FluxBasedModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., fd=None, fixed_dt=None, min_dt=SEC_IN_DAY,
                 max_dt=31*SEC_IN_DAY, inplace=True):

        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """
        super(FluxBasedModel, self).__init__(flowlines, mb_model=mb_model,
                                                  y0=y0, glen_a=glen_a, fs=fs,
                                                  fd=fd, inplace=inplace)
        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

        # Optim
        self._stags = []
        for fl, trib in zip(self.fls, self._trib):
            nx = fl.nx
            if trib[0] is not None:
                nx = fl.nx + 1
            a = np.zeros(nx+1)
            b = np.zeros(nx+1)
            c = np.zeros(nx+1)
            d = np.zeros(nx-1)
            e = np.zeros(nx)
            self._stags.append((a, b, c, d, e))

    def step(self, dt=31*SEC_IN_DAY):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

        # Loop over tributaries to determine the flux rate
        flxs = []
        aflxs = []
        for fl, trib, (slope_stag, thick_stag, section_stag, znxm1, znx) \
                in zip(self.fls, self._trib, self._stags):

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

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            rhogh = (RHO*G*slope_stag)**N
            u_stag = (thick_stag**(N+1)) * self._fd * rhogh + \
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
            else:
                flxs.append(flx_stag)
                aflxs.append(znx)

            # Time crit: limit the velocity somehow
            maxu = np.max(np.abs(u_stag))
            if maxu > 0.:
                # this is arbitrary
                # 100 is quite conservative, 60 is too low
                _dt = 1./100. * dx / maxu
            else:
                _dt = self.max_dt
            if _dt < dt:
                dt = _dt

        # Time step
        if dt < min_dt:
            if not self.dt_warning:
                log.warning('Reached min dt')
            self.dt_warning = True
        dt = np.clip(dt, min_dt, self.max_dt)

        # A second loop for the mass exchange
        for fl, flx_stag, aflx, trib in zip(self.fls, flxs, aflxs,
                                                 self._trib):

            # Mass balance
            widths = fl.widths_m
            mb = self.mb.get_mb(fl.surface_h, self.yr)
            # Allow parabolic beds to grow
            widths = np.where((mb > 0.) & (widths == 0), 10., widths)
            mb = dt * mb * widths

            # Update section with flowing and mass balance
            new_section = fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt + \
                          aflx*dt + mb

            # Keep positive values only and store
            fl.section = new_section.clip(0)

            # Add the last flux to the tributary
            # this is ok because the lines are sorted in order
            if trib[0] is not None:
                aflxs[trib[0]][trib[1]:trib[2]] += flx_stag[-1].clip(0) * \
                                                   trib[3]

        # Next step
        self.t += dt


class KarthausModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 fd=None, fixed_dt=None, min_dt=SEC_IN_DAY,
                 max_dt=31*SEC_IN_DAY, inplace=True):

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
                                            fd=fd, inplace=inplace)
        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt=31*SEC_IN_DAY):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt
        dt = np.clip(dt, min_dt, self.max_dt)

        fl = self.fls[0]
        dx = fl.dx_meter
        width = fl.widths_m
        thick = fl.thick

        MassBalance = self.mb.get_mb(fl.surface_h, self.yr)

        SurfaceHeight = fl.surface_h

        # Surface gradient
        SurfaceGradient = np.zeros(fl.nx)
        SurfaceGradient[1:fl.nx-1] = (SurfaceHeight[2:] -
                                      SurfaceHeight[:fl.nx-2])/(2*dx)
        SurfaceGradient[-1] = 0
        SurfaceGradient[0] = 0

        # Diffusivity
        Diffusivity = width * (RHO*G)**3 * thick**3 * SurfaceGradient**2
        Diffusivity *= 2/(N+2) * self.glen_a * thick**2 + self.fs

        # on stagger
        DiffusivityStaggered = np.zeros(fl.nx)
        SurfaceGradientStaggered = np.zeros(fl.nx)

        DiffusivityStaggered[1:] = (Diffusivity[:fl.nx-1] + Diffusivity[1:])/2.
        DiffusivityStaggered[0] = Diffusivity[0]

        SurfaceGradientStaggered[1:] = (SurfaceHeight[1:]-SurfaceHeight[:fl.nx-1])/dx
        SurfaceGradientStaggered[0] = 0

        GradxDiff = SurfaceGradientStaggered * DiffusivityStaggered

        # Yo
        NewIceThickness = np.zeros(fl.nx)
        NewIceThickness[:fl.nx-1] = thick[:fl.nx-1] + (dt/width[0:fl.nx-1]) * \
                                    (GradxDiff[1:]-GradxDiff[:fl.nx-1])/dx + \
                                    dt * MassBalance[:fl.nx-1]

        NewIceThickness[-1] = thick[fl.nx-2]

        fl.thick = NewIceThickness.clip(0)

        # Next step
        self.t += dt

class MUSCLSuperBeeModel(FlowlineModel):
    """This model is based on Jarosch et al. 2013 (doi:10.5194/tc-7-229-2013)

       The equation references in the comments refer to the paper for clarity
    """
    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=None, fd=None,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31*SEC_IN_DAY,
                 inplace=True):

        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        if len(flowlines) > 1:
            raise ValueError('MUSCL SuperBee model does not work with tributaries.')

        super(MUSCLSuperBeeModel, self).__init__(flowlines, mb_model=mb_model,
                                                 y0=y0, glen_a=glen_a, fs=fs,
                                                 fd=fd, inplace=inplace)
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
        val_phi = np.maximum(0,np.minimum(2*r,1),np.minimum(r,2))
        
        return val_phi

    def step(self, dt=31*SEC_IN_DAY):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt
        dt = np.clip(dt, min_dt, self.max_dt)

        fl = self.fls[0]
        dx = fl.dx_meter
        width = fl.widths_m
        
        """ Switch to the notation from the MUSCL_1D example
            This is useful to ensure that the MUSCL-SuperBee code
            is working as it has been benchmarked many time"""
         

        # mass balance
        m_dot = self.mb.get_mb(fl.surface_h, self.yr)
        # get in the surface elevation
        S = fl.surface_h
        # get the bed
        B = fl.bed_h
        # define Glen's law here
        Gamma = 2.*self.glen_a*(RHO*G)**N / (N+2.) # this is the correct Gamma !!
        #Gamma = self.fd*(RHO*G)**N # this is the Gamma to be in sync with Karthaus and Flux
        # time stepping 
        c_stab = 0.165
        
        # define the finite difference indices required for the MUSCL-SuperBee scheme
        k = np.arange(0,fl.nx)
        kp = np.hstack([np.arange(1,fl.nx),fl.nx-1])
        kpp = np.hstack([np.arange(2,fl.nx),fl.nx-1,fl.nx-1])
        km = np.hstack([0,np.arange(0,fl.nx-1)])
        kmm = np.hstack([0,0,np.arange(0,fl.nx-2)])

        # I'm gonna introduce another level of adaptive time stepping here, which is probably not
        # necessary. However I keep it to be consistent with my benchmarked and tested code.
        # If the OGGM time stepping is correctly working, this loop should never run more than once
        stab_t = 0.
        while stab_t < dt:
            H = S - B
            
            # MUSCL scheme up. "up" denotes here the k+1/2 flux boundary
            r_up_m = (H[k]-H[km])/(H[kp]-H[k])                           # Eq. 27
            H_up_m = H[k] + 0.5 * self.phi(r_up_m)*(H[kp]-H[k])          # Eq. 23
            r_up_p = (H[kp]-H[k])/(H[kpp]-H[kp])                         # Eq. 27, now k+1 is used instead of k
            H_up_p = H[kp] - 0.5 * self.phi(r_up_p)*(H[kpp]-H[kp])       # Eq. 24
            
            # surface slope gradient
            s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((N-1.)/2.)
            D_up_m = Gamma * H_up_m**(N+2.) * s_grad_up                  # like Eq. 30, now using Eq. 23 instead of Eq. 24
            D_up_p = Gamma * H_up_p**(N+2.) * s_grad_up                  # Eq. 30
            
            D_up_min = np.minimum(D_up_m,D_up_p);                        # Eq. 31
            D_up_max = np.maximum(D_up_m,D_up_p);                        # Eq. 32
            D_up = np.zeros(fl.nx)
            
            # Eq. 33
            D_up[np.logical_and(S[kp]<=S[k],H_up_m<=H_up_p)] = D_up_min[np.logical_and(S[kp]<=S[k],H_up_m<=H_up_p)]
            D_up[np.logical_and(S[kp]<=S[k],H_up_m>H_up_p)] = D_up_max[np.logical_and(S[kp]<=S[k],H_up_m>H_up_p)]
            D_up[np.logical_and(S[kp]>S[k],H_up_m<=H_up_p)] = D_up_max[np.logical_and(S[kp]>S[k],H_up_m<=H_up_p)]
            D_up[np.logical_and(S[kp]>S[k],H_up_m>H_up_p)] = D_up_min[np.logical_and(S[kp]>S[k],H_up_m>H_up_p)]

            # MUSCL scheme down. "down" denotes here the k-1/2 flux boundary
            r_dn_m = (H[km]-H[kmm])/(H[k]-H[km])
            H_dn_m = H[km] + 0.5 * self.phi(r_dn_m)*(H[k]-H[km])
            r_dn_p = (H[k]-H[km])/(H[kp]-H[k])
            H_dn_p = H[k] - 0.5 * self.phi(r_dn_p)*(H[kp]-H[k])
            
            # calculate the slope gradient
            s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((N-1.)/2.)
            D_dn_m = Gamma * H_dn_m**(N+2.) * s_grad_dn
            D_dn_p = Gamma * H_dn_p**(N+2.) * s_grad_dn
            
            D_dn_min = np.minimum(D_dn_m,D_dn_p);
            D_dn_max = np.maximum(D_dn_m,D_dn_p);
            D_dn = np.zeros(fl.nx)
            
            D_dn[np.logical_and(S[k]<=S[km],H_dn_m<=H_dn_p)] = D_dn_min[np.logical_and(S[k]<=S[km],H_dn_m<=H_dn_p)]
            D_dn[np.logical_and(S[k]<=S[km],H_dn_m>H_dn_p)] = D_dn_max[np.logical_and(S[k]<=S[km],H_dn_m>H_dn_p)]
            D_dn[np.logical_and(S[k]>S[km],H_dn_m<=H_dn_p)] = D_dn_max[np.logical_and(S[k]>S[km],H_dn_m<=H_dn_p)]
            D_dn[np.logical_and(S[k]>S[km],H_dn_m>H_dn_p)] = D_dn_min[np.logical_and(S[k]>S[km],H_dn_m>H_dn_p)]
            
            dt_stab = c_stab * dx**2. / max(max(abs(D_up)),max(abs(D_dn)))      # Eq. 37
            dt_use = min(dt_stab,dt-stab_t)
            stab_t = stab_t + dt_use
            
            # check if the extra time stepping is needed [to be removed one day]
            #if dt_stab < dt:
            #    print "MUSCL extra time stepping dt: %f dt_stab: %f" % (dt, dt_stab)
            #else:
            #    print "MUSCL Scheme fine with time stepping as is"
            
            #explicit time stepping scheme
            div_q = (D_up * (S[kp] - S[k])/dx - D_dn * (S[k] - S[km])/dx)/dx    # Eq. 36
            S = S[k] + (m_dot + div_q)*dt_use                                   # Eq. 35
            
            S = np.maximum(S,B)                                                 # Eq. 7
        
        # Done with the loop, prepare output
        NewIceThickness = S-B
        
        fl.thick = NewIceThickness
        # fl.section = NewIceThickness * width
        #fl.section = NewIceThickness
        
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
                sel = ds.ts_section.isel(time=(ds.year==year) &
                                              (ds.month==month))
                fl.section = sel.values
        else:
            for fl, ds in zip(self.fls, self.dss):
                sel = ds.ts_section.sel(time=year)
                fl.section = sel.values
        self.yr = sel.time.values

    def area_m2_ts(self, rollmin=36):
        """rollmin is the numebr of months you want to smooth onto"""
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

    def length_m_ts(self, rollmin=36):
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
    line = shpg.LineString(ds['coords'].values)
    args = dict(line=line, dx=ds.dx, map_dx=ds.map_dx,
                surface_h=ds['surface_h'].values,
                bed_h=ds['bed_h'].values)

    have = {'c', 'x', 'surface_h', 'coords', 'bed_h', 'z', 'p', 'n',
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
        fl.order = oggm.core.preprocessing.centerlines._line_order(fl)

    return fls



@entity_task(log, writes=['model_flowlines'])
def init_present_time_glacier(gdir):
    """First task after inversion. Merges the data from the various
    preprocessing tasks into a stand-alone dataset ready for run.

    In a first stage we assume that all divides CAN be merged as for HEF,
    so that the concept of divide is not necessary anymore.

    This task is horribly coded and needs more work

    Parameters
    ----------
    gdir : oggm.GlacierDirectory
    """

    # Topo for heights
    with netCDF4.Dataset(gdir.get_filepath('gridded_data', div_id=0)) as nc:
        topo = nc.variables['topo_smoothed'][:]

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # Default bed shape
    defshape = 0.003

    # Map
    map_dx = gdir.grid.dx

    # OK. Dont try to solve problems you don't know about yet - i.e.
    # rethink about all this when we will have proper divides everywhere.
    # for HEF the following will work, and this is very ugly.
    major_div = gdir.read_pickle('major_divide', div_id=0)
    div_ids = list(gdir.divide_ids)
    div_ids.remove(major_div)
    div_ids = [major_div] + div_ids
    fls_list = []
    fls_per_divide = []
    inversion_per_divide = []
    for div_id in div_ids:
        fls = gdir.read_pickle('inversion_flowlines', div_id=div_id)
        fls_list.extend(fls)
        fls_per_divide.append(fls)
        invs = gdir.read_pickle('inversion_output', div_id=div_id)
        inversion_per_divide.append(invs)

    max_shape = cfg.PARAMS['max_shape_param']

    # Extend the flowlines with the downstream lines, make a new object
    new_fls = []
    flows_to_ids = []
    major_id = None
    for fls, invs, did in zip(fls_per_divide, inversion_per_divide, div_ids):
        for fl, inv in zip(fls[0:-1], invs[0:-1]):
            bed_h = fl.surface_h - inv['thick']
            w = fl.widths * map_dx
            bed_shape = (4*inv['thick'])/(w**2)
            bed_shape = bed_shape.clip(0, max_shape)
            bed_shape = np.where(inv['thick'] < 1., np.NaN, bed_shape)
            bed_shape = utils.interp_nans(bed_shape, default=defshape)
            nfl = ParabolicFlowline(fl.line, fl.dx, map_dx, fl.surface_h,
                                    bed_h, bed_shape)
            flows_to_ids.append(fls_list.index(fl.flows_to))
            new_fls.append(nfl)

        # The last one is extended with the downstream
        # TODO: copy-paste code smell
        fl = fls[-1]
        inv = invs[-1]
        dline = gdir.read_pickle('downstream_line', div_id=did)
        long_line = oggm.core.preprocessing.geometry._line_extend(fl.line,
                                                                  dline, fl.dx)

        # Interpolate heights
        x, y = long_line.xy
        hgts = interpolator((y, x))

        # Inversion stuffs
        bed_h = hgts.copy()
        bed_h[0:len(fl.surface_h)] -= inv['thick']

        # Shapes
        w = fl.widths * map_dx
        bed_shape = (4*inv['thick'])/(w**2)
        bed_shape = bed_shape.clip(0, max_shape)
        bed_shape = np.where(inv['thick'] < 1., np.NaN, bed_shape)
        bed_shape = utils.interp_nans(bed_shape, default=defshape)

        # TODO: forbid too small shape close to the end?
        # bed_shape[-4:] = bed_shape[-4:].clip(cfg.PARAMS['mixed_min_shape'])

        # Take the median of the last 30%
        ashape = np.median(bed_shape[-np.floor(len(bed_shape)/3.).astype(np.int64):])
        # But forbid too small shape
        ashape = ashape.clip(cfg.PARAMS['mixed_min_shape'])

        bed_shape = np.append(bed_shape, np.ones(len(bed_h)-len(bed_shape))*ashape)
        nfl = ParabolicFlowline(long_line, fl.dx, map_dx, hgts, bed_h, bed_shape)

        if major_id is None:
            flid = -1
            major_id = len(fls)-1
        else:
            flid = major_id
        flows_to_ids.append(flid)
        new_fls.append(nfl)

    # Finalize the linkages
    for fl, fid in zip(new_fls, flows_to_ids):
        if fid == -1:
            continue
        fl.set_flows_to(new_fls[fid])

    # Adds the line level
    for fl in new_fls:
        fl.order = oggm.core.preprocessing.centerlines._line_order(fl)

    # And sort them per order
    fls = []
    for i in np.argsort([fl.order for fl in new_fls]):
        fls.append(new_fls[i])

    # Write the data
    gdir.write_pickle(fls, 'model_flowlines')


def convert_to_mixed_flowline(fls):
    """Replaces parabolic flowlines with a mixed type."""

    new_fls = []
    flows_to_ids = []
    fls = copy.deepcopy(fls)
    for fl in fls:

        lambdas = fl.bed_shape * 0.
        lambdas[:] = cfg.PARAMS['trapezoid_lambdas']

        ptrap = np.nonzero(fl.bed_shape < cfg.PARAMS['mixed_min_shape'])

        fl_ = MixedFlowline(line=fl.line, dx=fl.dx, map_dx=fl.map_dx,
                            surface_h=fl.surface_h, bed_h=fl.bed_h,
                            section=fl.section, bed_shape=fl.bed_shape,
                            where_trapezoid=ptrap, lambdas=lambdas,
                            widths_m=fl.widths_m)

        new_fls.append(fl_)

        if fl.flows_to is None:
            flows_to_ids.append(-1)
        else:
            flows_to_ids.append(fls.index(fl.flows_to))

    for i, fid in enumerate(flows_to_ids):
        if fid != -1:
            new_fls[i].set_flows_to(new_fls[fid])

    # Adds the line level
    for fl in new_fls:
        fl.order = oggm.core.preprocessing.centerlines._line_order(fl)

    return new_fls

def _find_inital_glacier(final_model, firstguess_mb, y0, y1,
                         rtol=0.01, atol=10, max_ite=100,
                         init_bias=0., equi_rate=0.0005,
                         ref_area=None):
    """ Iterative search for a plausible starting time glacier"""

    # Objective
    if ref_area is None:
        ref_area = final_model.area_m2
    log.info('iterative_initial_glacier_search in year %d. Ref area to catch: %.3f km2. '
             'Tolerance: %.2f %%' ,
             np.int64(y0), ref_area*1e-6, rtol*100)

    # are we trying to grow or to shrink the glacier?
    prev_model = copy.deepcopy(final_model)
    prev_fls = copy.deepcopy(prev_model.fls)
    prev_model.reset_y0(y0)
    prev_model.run_until(y1)
    prev_area = prev_model.area_m2

    # Just in case we already hit the correct starting state
    if np.allclose(prev_area, ref_area, atol=atol, rtol=rtol):
        model = copy.deepcopy(final_model)
        model.reset_y0(y0)
        log.info('iterative_initial_glacier_search: inital starting glacier converges '
                 'to itself with a final dif of %.2f %%',
                 utils.rel_err(ref_area, prev_area) * 100)
        return 0, None, model

    if prev_area < ref_area:
        sign_mb = -1.
        log.info('iterative_initial_glacier_search, ite: %d. Glacier would be too '
                 'small of %.2f %%. Continue', 0,
                 utils.rel_err(ref_area, prev_area) * 100)
    else:
        log.info('iterative_initial_glacier_search, ite: %d. Glacier would be too '
                 'big of %.2f %%. Continue', 0,
                 utils.rel_err(ref_area, prev_area) * 100)
        sign_mb = 1.

    # Log prefix
    logtxt = 'iterative_initial_glacier_search'

    # Loop until 100 iterations
    c = 0
    bias_step = 0.1
    mb_bias = init_bias - bias_step
    reduce_step = 0.01

    mb = copy.deepcopy(firstguess_mb)
    mb.temp_bias = sign_mb * mb_bias
    grow_model = FluxBasedModel(copy.deepcopy(final_model.fls), mb_model=mb,
                                fs=final_model.fs,
                                glen_a=final_model.glen_a,
                                min_dt=final_model.min_dt,
                                max_dt=final_model.max_dt)
    while True and (c < max_ite):
        c += 1

        # Grow
        mb_bias += bias_step
        mb.temp_bias = sign_mb * mb_bias
        log.info(logtxt + ', ite: %d. New bias: %.2f', c, sign_mb * mb_bias)
        grow_model.reset_flowlines(copy.deepcopy(prev_fls))
        grow_model.reset_y0(0.)
        grow_model.run_until_equilibrium(rate=equi_rate)
        log.info(logtxt + ', ite: %d. Grew to equilibrium for %d years, '
                          'new area: %.3f km2', c, grow_model.yr,
                           grow_model.area_km2)

        # Shrink
        new_fls = copy.deepcopy(grow_model.fls)
        new_model = copy.deepcopy(final_model)
        new_model.reset_flowlines(copy.deepcopy(new_fls))
        new_model.reset_y0(y0)
        new_model.run_until(y1)
        new_area = new_model.area_m2

        # Maybe we done?
        if np.allclose(new_area, ref_area, atol=atol, rtol=rtol):
            new_model.reset_flowlines(new_fls)
            new_model.reset_y0(y0)
            log.info(logtxt + ', ite: %d. Converged with a '
                     'final dif of %.2f %%', c,
                     utils.rel_err(ref_area, new_area)*100)
            return c, mb_bias, new_model

        # See if we did a step to far or if we have to continue growing
        do_cont_1 = (sign_mb < 0.) and (new_area < ref_area)
        do_cont_2 = (sign_mb > 0.) and (new_area > ref_area)
        if do_cont_1 or do_cont_2:
            # Reset the previous state and continue
            prev_fls = new_fls

            log.info(logtxt + ', ite: %d. Dif of %.2f %%. '
                              'Continue', c,
                     utils.rel_err(ref_area, new_area)*100)
            continue

        # Ok. We went too far. Reduce the bias step but keep previous state
        mb_bias -= bias_step
        bias_step /= reduce_step
        log.info(logtxt + ', ite: %d. Went too far.', c)
        if bias_step < 0.1:
            break

    raise RuntimeError('Did not converge after {} iterations'.format(c))


@entity_task(log)
def random_glacier_evolution(gdir, nyears=1000, y0=None, seed=None,
                             filesuffix=''):
    """Random glacier dynamics for benchmarking purposes.

     This runs the random mass-balance model for a certain number of years.
     
     Parameters
     ----------
     nyears : length of the simulation
     y0 : central year of the random climate period
     seed : seed for the random generate
     filesuffix : for the output file
     """

    if cfg.PARAMS['use_optimized_inversion_params']:
        d = gdir.read_pickle('inversion_params')
        fs = d['fs']
        glen_a = d['glen_a']
    else:
        fs = cfg.PARAMS['flowline_fs']
        glen_a = cfg.PARAMS['flowline_glen_a']

    ys = 1
    ye = ys + nyears
    mb = mbmods.RandomMassBalanceModel(gdir, y0=y0, seed=seed)
    fls = gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb_model=mb, y0=ys, fs=fs, glen_a=glen_a)

    # run
    path = gdir.get_filepath('past_model', delete=True, filesuffix=filesuffix)
    model.run_until_and_store(ye, path=path)


@entity_task(log, writes=['past_model'])
def iterative_initial_glacier_search(gdir, y0=None, init_bias=0., rtol=0.005,
                                     write_steps=True):
    """Iterative search for the glacier in year y0.

    this doesn't really work.

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    New file::
        - past_model.p: a ModelFlowline object
    """

    if cfg.PARAMS['use_optimized_inversion_params']:
        d = gdir.read_pickle('inversion_params')
        fs = d['fs']
        glen_a = d['glen_a']
    else:
        fs = cfg.PARAMS['flowline_fs']
        glen_a = cfg.PARAMS['flowline_glen_a']

    if y0 is None:
        y0 = cfg.PARAMS['y0']
    y1 = gdir.rgi_date.year
    mb = mbmods.PastMassBalanceModel(gdir)
    fls = gdir.read_pickle('model_flowlines')
    if cfg.PARAMS['bed_shape'] == 'mixed':
        fls = convert_to_mixed_flowline(fls)

    model = FluxBasedModel(fls, mb_model=mb, y0=0., fs=fs, glen_a=glen_a)
    assert np.isclose(model.area_km2, gdir.rgi_area_km2, rtol=0.05)

    mb = mbmods.BackwardsMassBalanceModel(gdir)
    ref_area = gdir.rgi_area_m2
    ite, bias, past_model = _find_inital_glacier(model, mb, y0, y1,
                                                 rtol=rtol,
                                                 init_bias=init_bias,
                                                 ref_area=ref_area)

    # Some parameters for posterity:
    params = OrderedDict(rtol=rtol, init_bias=init_bias, ref_area=ref_area,
                         ite=ite, mb_bias=bias)

    # Write the data
    gdir.write_pickle(params, 'find_initial_glacier_params')
    path = gdir.get_filepath('past_model', delete=True)
    if write_steps:
        _ = past_model.run_until_and_store(y1, path=path)
    else:
        past_model.to_netcdf(path)
