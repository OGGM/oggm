"""Flowline modelling"""
from __future__ import division
from six.moves import zip

# Built ins
import logging
import copy
# External libs
import numpy as np
import netCDF4
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as optimization
# Locals
import oggm.conf as cfg
from oggm import utils
import oggm.prepro.geometry
import oggm.prepro.centerlines
import oggm.models.massbalance as mbmods

# Module logger
log = logging.getLogger(__name__)

# Constants
sec_in_year = 365*24*3600
sec_in_month = 31*24*3600
sec_in_day = 24*3600
sec_in_hour = 3600

rho = 900.  # ice density
g = 9.81  # gravity
n = 3.  # Glen's law's exponent
twothirds = 2./3.
fourthirds = 4./3.


class ModelFlowline(oggm.prepro.geometry.InversionFlowline):
    """A more advanced Flowline."""

    def __init__(self, line, dx, map_dx, surface_h, bed_h):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        super(ModelFlowline, self).__init__(line, dx, surface_h)

        self._thick = (surface_h - bed_h).clip(0.)
        self.map_dx = map_dx
        self.dx_meter = map_dx * self.dx
        self.bed_h = bed_h

    @oggm.prepro.geometry.InversionFlowline.widths.getter
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

    @oggm.prepro.geometry.InversionFlowline.surface_h.getter
    def surface_h(self):
        return self._thick + self.bed_h

    @property
    def length_m(self):
        pok = np.where(self.thick == 0.)[0]
        if len(pok) == 0:
            return 0.
        else:
            return (pok[0]-0.5) * self.dx_meter

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


class ParabolicFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line, dx, map_dx, surface_h, bed_h, bed_shape):
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

        self.bed_shape = bed_shape
        self._sqrt_bed = np.sqrt(bed_shape)

    @oggm.prepro.geometry.InversionFlowline.widths.getter
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return np.sqrt(4*self.thick/self.bed_shape)

    @property
    def section(self):
        return twothirds * self.widths_m * self.thick

    @section.setter
    def section(self, val):
        self.thick = (0.75 * val * self._sqrt_bed)**twothirds


class VerticalWallFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line, dx, map_dx, surface_h, bed_h, widths):
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

    @oggm.prepro.geometry.InversionFlowline.widths.getter
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


class TrapezoidalFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line, dx, map_dx, surface_h, bed_h, widths, lambdas):
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

    @oggm.prepro.geometry.InversionFlowline.widths.getter
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


class MixedFlowline(ModelFlowline):
    """A more advanced Flowline."""

    def __init__(self, line, dx, map_dx, surface_h, bed_h, bed_shape,
                 min_shape=0.0015, lambdas=0.2):
        """ Instanciate.

        Parameters
        ----------
        line: Shapely LineString

        Properties
        ----------
        #TODO: document properties
        """
        self._dot = False
        super(MixedFlowline, self).__init__(line, dx, map_dx,
                                                surface_h.copy(), bed_h.copy())

        self.bed_shape = bed_shape
        self._sqrt_bed = np.sqrt(bed_shape)

        # Where we will have to use the other
        #TODO: add these parameters to configfile
        totest = bed_shape.copy()
        if self.flows_to is None:
            totest[-np.floor(len(totest)/3.).astype(np.int64):] = min_shape + 1
        pt = np.nonzero(totest < min_shape)

        # correct bed_h
        for_assert = self.section.copy()
        for_later = self.widths_m[pt]

        sec = self.section[pt]
        b = - 2 * self.widths_m[pt]
        det = b ** 2 - 4 * lambdas * 2 * sec
        h = (- b - np.sqrt(det)) / (2 * lambdas)
        assert np.alltrue(h >= 0)
        self.bed_h[pt] = surface_h[pt] - h.clip(0)
        self._thick[pt] = h.clip(0)
        assert np.allclose(surface_h, self.surface_h)

        _w0_m = for_later - lambdas * self.thick[pt]
        assert np.alltrue(_w0_m >= 0)

        self._w0_m = _w0_m
        self._lambdas = lambdas
        self._pt = pt
        self._dot = len(pt[0]) > 0
        assert np.allclose(self.thick[pt], h.clip(0))

        assert np.allclose(for_assert, self.section)

    @oggm.prepro.geometry.InversionFlowline.widths.getter
    def widths_m(self):
        """Compute the widths out of H and shape"""
        out = np.sqrt(4*self.thick/self.bed_shape)
        if self._dot:
            out[self._pt] = self._w0_m + self._lambdas * self.thick[self._pt]
        return out

    @property
    def section(self):
        out = twothirds * self.widths_m * self.thick
        if self._dot:
            out[self._pt] = (self.widths_m[self._pt] + self._w0_m) / 2 * self.thick[self._pt]
        return out

    @section.setter
    def section(self, val):
        out = (0.75 * val * self._sqrt_bed)**twothirds
        if self._dot:
            b = 2 * self._w0_m
            a = 2 * self._lambdas
            out[self._pt] = (np.sqrt(b**2 + 4 * a * val[self._pt]) - b) / a
        self.thick = out

    @property
    def area_m2(self):
        widths = np.where(self.thick > 0., self.widths_m, 0.)
        return np.sum(widths * self.dx_meter)


class FlowlineModel(object):
    """Interface to the actual model"""

    def __init__(self, flowlines, mass_balance, y0, fs, fd):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        self.mb = mass_balance
        self.fd = fd
        self.fs = fs

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
            if fl.nx >= 9:
                gk = utils.gaussian_kernel[9]
                id0 = ide-4
                id1 = ide+5
            elif fl.nx >= 7:
                gk = utils.gaussian_kernel[7]
                id0 = ide-3
                id1 = ide+4
            elif fl.nx >= 5:
                gk = utils.gaussian_kernel[5]
                id0 = ide-2
                id1 = ide+3
            trib_ind.append((idl, id0, id1, gk))
        self._trib = trib_ind

    @property
    def yr(self):
        return self.y0 + self.t / sec_in_year

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

    def step(self, dt=None):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1):

        t = (y1-self.y0) * sec_in_year
        while self.t < t:
            self.step(dt=t-self.t)

    def run_until_equilibrium(self, rate=0.0005, ystep=5, max_ite=200):

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            t_rate = np.abs(v_af - v_bef) / v_bef
            if np.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1

        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium as expected')


class FluxBasedModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mass_balance, y0, fs, fd,
                       fixed_dt=None,
                       min_dt=sec_in_day,
                       max_dt=sec_in_month):

        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """
        super(FluxBasedModel, self).__init__(flowlines, mass_balance,
                                             y0, fs, fd)
        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt=sec_in_month):
        """Advance one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

        # Loop over tributaries to determine the flux rate
        flxs = []
        aflxs = []
        for fl in self.fls:

            dx = fl.dx_meter

            # Staggered gradient
            sh = fl.surface_h
            slope_stag = np.zeros(fl.nx+1)
            slope_stag[1:-1] = (sh[0:-1] - sh[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

            # Staggered thick
            thick = fl.thick
            thick_stag = np.zeros(fl.nx+1)
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            rhogh = (rho*g*slope_stag)**n
            u_stag = (thick_stag**(n+1)) * self.fd * rhogh + \
                     (thick_stag**(n-1)) * self.fs * rhogh

            # Staggered section
            section = fl.section
            section_stag = np.zeros(fl.nx+1)
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flx_stag = u_stag * section_stag / dx
            flxs.append(flx_stag)
            aflxs.append(np.zeros(fl.nx))

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

    def backstep(self, dt=-sec_in_month):
        """Backwards one step."""

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt > -self.min_dt else -self.min_dt
        max_dt = -self.max_dt
        dt = np.clip(dt, max_dt, min_dt)

        # Loop over tributaries to determine the flux rate
        flxs = []
        aflxs = []
        for fl in self.fls:

            dx = fl.dx_meter

            # Mass balance and "add" to section
            widths = fl.widths_m
            mb = self.mb.get_mb(fl.surface_h, self.yr)

            is_no_zero = (fl.thick[1:] > 0) | (fl.thick[:-1] > 0)
            is_no_zero = np.hstack([True, is_no_zero])
            widths = np.where(is_no_zero, widths, 0.)
            mb = dt * mb * widths
            fl.section = fl.section + mb

            # Staggered gradient
            sh = fl.surface_h
            slope_stag = np.zeros(fl.nx+1)
            slope_stag[1:-1] = (sh[0:-1] - sh[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick = fl.thick
            thick_stag = np.zeros(fl.nx+1)
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            rhogh = (rho*g*slope_stag)**n
            u_stag = (thick_stag**(n+1)) * self.fd * rhogh + \
                     (thick_stag**(n-1)) * self.fs * rhogh

            # Staggered section
            section = fl.section
            section_stag = np.zeros(fl.nx+1)
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flx_stag = u_stag * section_stag / dx
            flxs.append(flx_stag)
            aflxs.append(np.zeros(fl.nx))

        # A second loop for the mass exchange
        for fl, flx_stag, aflx, trib in zip(self.fls, flxs, aflxs,
                                                 self._trib):

            # Update section with flowing and mass balance
            new_section = fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt
                          # + aflx*dt

            # Keep positive values only and store
            fl.section = new_section.clip(0)

            # Add the last flux to the tributary
            # this is ok because the lines are sorted in order
            # if trib[0] is not None:
            #     aflxs[trib[0]][trib[1]:trib[2]] += flx_stag[-1].clip(0) * \
            #                                        trib[3]

        # Next step
        self.t += dt


class KarthausModel(FlowlineModel):
    """The actual model"""

    def __init__(self, flowlines, mass_balance, y0, fs, fd,
                       fixed_dt=None,
                       min_dt=sec_in_day,
                       max_dt=sec_in_month):

        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        if len(flowlines) > 1:
            raise ValueError('Karthaus model does not work with tributaries.')

        super(KarthausModel, self).__init__(flowlines, mass_balance,
                                            y0, fs, fd)
        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(self, dt=sec_in_month):
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
        Diffusivity = width * (rho*g)**3 * thick**3 * SurfaceGradient**2
        Diffusivity *= self.fd * thick**2 + self.fs

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

        fl.section = NewIceThickness.clip(0) * width

        # Next step
        self.t += dt


def init_present_time_glacier(gdir, min_shape=0.0012, lambdas=0.2):
    """First task after inversion. Merges the data from the various
    preprocessing tasks into a stand-alone dataset ready for run.

    In a first stage we assume that all divides CAN be merged as for HEF,
    so that the concept of divide is not necessary anymore.

    Parameters
    ----------
    gdir: GlacierDir object

    I/O
    ---
     - new file: model_flowlines.p
    """

    fn = gdir.get_filepath('major_divide', div_id=0)
    with open(fn, 'r') as text_file:
        major_div = int(text_file.read())

    # Topo for heights
    nc = netCDF4.Dataset(gdir.get_filepath('grids', div_id=0))
    topo = nc.variables['topo_smoothed'][:]
    nc.close()

    # Bilinear interpolation
    # Geometries coordinates are in "pixel centered" convention, i.e
    # (0, 0) is also located in the center of the pixel
    xy = (np.arange(0, gdir.grid.ny-0.1, 1),
          np.arange(0, gdir.grid.nx-0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo)

    # Smooth window
    sw = cfg.params['flowline_height_smooth']

    # Map
    map_dx = gdir.grid.dx

    # OK. Dont try to solve problems you don't know about yet - i.e.
    # rethink about all this when we will have proper divides everywhere.
    # for HEF the following will work, and this is very ugly.
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

    # Extend the flowlines with the downstream lines, make a new object
    # out of these

    #TODO: add this choice to config
    from functools import partial
    flobject = partial(MixedFlowline, min_shape=min_shape, lambdas=lambdas)

    new_fls = []
    flows_to_ids = []
    major_id = None
    for fls, invs, did in zip(fls_per_divide, inversion_per_divide, div_ids):
        for fl, inv in zip(fls[0:-1], invs[0:-1]):
            bed_h = fl.surface_h - inv['thick']
            # Interpolate shapes
            shape = utils.interp_nans(inv['shape'])
            nfl = flobject(fl.line, fl.dx, map_dx, fl.surface_h,
                                    bed_h, shape)
            flows_to_ids.append(fls_list.index(fl.flows_to))
            new_fls.append(nfl)

        # The last one is extended with the downstream
        # TODO: copy-paste code smell
        fl = fls[-1]
        inv = invs[-1]
        dline = gdir.read_pickle('downstream_line', div_id=did)
        long_line = oggm.prepro.geometry._line_extend(fl.line, dline, fl.dx)
        # Interpolate heights
        x, y = long_line.xy
        hgts = interpolator((y, x))
        # If smoothing, this is the moment
        hgts = gaussian_filter1d(hgts, sw)

        # Inversion stuffs
        bed_h = hgts.copy()
        bed_h[0:len(fl.surface_h)] -= inv['thick']
        # Interpolate shapes
        shape = utils.interp_nans(inv['shape'])

        # Take the median of the last 30%
        ashape = np.median(shape[-np.floor(len(shape)/3.).astype(np.int64):])
        shape = np.append(shape, np.ones(len(bed_h)-len(shape))*ashape)
        nfl = flobject(long_line, fl.dx, map_dx, hgts, bed_h, shape)

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
        fl.order = oggm.prepro.centerlines._line_order(fl)

    # And sort them per order
    fls = []
    for i in np.argsort([fl.order for fl in new_fls]):
        fls.append(new_fls[i])

    # Write the data
    gdir.write_pickle(fls, 'model_flowlines')


def _find_inital_glacier(final_model, firstguess_mb, y0, y1,
                         rtol=0.01, atol=10, max_ite=100,
                         init_bias=0., equi_rate=0.0005):
    """ Iterative search for a plausible starting time glacier"""

    # Objective
    orig_area = final_model.area_m2

    # are we trying to grow or to shrink the glacier?
    prev_model = copy.deepcopy(final_model)
    prev_fls = copy.deepcopy(prev_model.fls)
    prev_model.reset_y0(y0)
    prev_model.run_until(y1)
    prev_area = prev_model.area_m2

    # Just in case we already hit the correct starting state
    if np.allclose(prev_area, orig_area, atol=atol, rtol=rtol):
        model = copy.deepcopy(final_model)
        model.reset_y0(y0)
        log.info('find_inital_glacier, ite: %d. Converged with a final error '
                 'of %.3f', 0,
                 utils.rel_err(orig_area, prev_area))
        return model

    if prev_area < orig_area:
        sign_mb = 1.
        log.info('find_inital_glacier, ite: %d. Glacier would be too '
                  'small of %.3f. Continue', 0,
                  utils.rel_err(orig_area, prev_area))
    else:
        log.info('find_inital_glacier, ite: %d. Glacier would be too '
                  'big of %.3f. Continue', 0,
                  utils.rel_err(orig_area, prev_area))
        sign_mb = -1.

    # Loop until 100 iterations
    c = 0
    bias_step = 50.
    mb_bias = init_bias - bias_step
    reduce_step = 5.

    mb = copy.deepcopy(firstguess_mb)
    mb.set_bias(sign_mb * mb_bias)
    grow_model = FluxBasedModel(copy.deepcopy(final_model.fls), mb,
                                y0, final_model.fs, final_model.fd,
                                min_dt=final_model.min_dt,
                                max_dt=final_model.max_dt)
    while True and (c < max_ite):
        c += 1

        # Grow
        mb_bias += bias_step
        mb.set_bias(sign_mb * mb_bias)
        log.info('find_inital_glacier, ite: %d. New bias: %.0f', c,
                  sign_mb * mb_bias)
        grow_model.reset_flowlines(copy.deepcopy(prev_fls))
        grow_model.reset_y0(0.)
        grow_model.run_until_equilibrium(rate=equi_rate)
        log.info('find_inital_glacier, ite: %d. Grew for: %d years', c,
                  grow_model.yr)

        # Shrink
        new_fls = copy.deepcopy(grow_model.fls)
        new_model = copy.deepcopy(final_model)
        new_model.reset_flowlines(copy.deepcopy(new_fls))
        new_model.reset_y0(y0)
        new_model.run_until(y1)
        new_area = new_model.area_m2

        # Maybe we done?
        if np.allclose(new_area, orig_area, atol=atol, rtol=rtol):
            new_model.reset_flowlines(new_fls)
            new_model.reset_y0(y0)
            log.info('find_inital_glacier, ite: %d. Converged with a '
                     'final error of %.3f', c,
                     utils.rel_err(orig_area, new_area))
            return new_model

        # See if we did a step to far or if we have to continue growing
        do_cont_1 = (sign_mb > 0.) and (new_area < orig_area)
        do_cont_2 = (sign_mb < 0.) and (new_area > orig_area)
        if do_cont_1 or do_cont_2:
            # Reset the previous state and continue
            prev_fls = new_fls

            log.info('find_inital_glacier, ite: %d. Error of %.3f. '
                      'Continue', c,
                      utils.rel_err(orig_area, new_area))
            continue

        # Ok. We went too far. Reduce the bias step but keep previous state
        mb_bias -= bias_step
        bias_step /= reduce_step
        log.info('find_inital_glacier, ite: %d. Went too far.', c)
        if bias_step < 0.1:
            break

    raise RuntimeError('Did not converge after {} iterations'.format(c))


def find_inital_glacier(gdir, div_id=None):
    """Search for the glacier in year 1847

    Parameters
    ----------
    gdir: GlacierDir object
    div_id: the divide ID to process (should be left to None)

    I/O
    ---
    New file::
        - past_model.p: a ModelFlowline object
    """

    d = gdir.read_pickle('flowline_params')
    fs = d['fs']
    fd = d['fd']

    y0 = 1847
    y1 = 2003
    rtol = 0.005

    mb = mbmods.HistalpMassBalanceModel(gdir)
    fls = gdir.read_pickle('model_flowlines')
    model = FluxBasedModel(fls, mb, 0., fs, fd)
    vol_ref = model.volume_km3

    mb = mbmods.BackwardsMassBalanceModel(gdir)
    past_model = _find_inital_glacier(model, mb, y0, y1, rtol=rtol,
                                      init_bias=150)

    # Write the data
    gdir.write_pickle(past_model, 'past_model')
