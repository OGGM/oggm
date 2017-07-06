"""Useful plotting functions"""
from __future__ import division
from six.moves import zip

from collections import OrderedDict
import warnings
import logging
import functools

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

import shapely.geometry as shpg
import numpy as np
import netCDF4
import salem

from oggm.utils import entity_task, global_task

# Local imports
import oggm.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Remove extreme colors from colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def _plot_map(plotfunc):
    """
    Decorator for common salem.Map plotting logic
    """
    commondoc = """
    Parameters
    ----------
    ax : matplotlib axes object, optional
        If None, uses own axis
    add_scalebar : Boolean, optional, default=True
        Adds scale bar to the plot
    add_colorbar : Boolean, optional, default=True
        Adds colorbar to axis
    horizontal_colorbar : Boolean, optional, default=False
        Horizontal colorbar instead
    title : str, optional
        If left to None, the plot decides wether it writes a title or not. Set
        to '' for no title.
    title_comment : str, optional
        add something to the default title. Set to none to remove default
    lonlat_contours_kwargs: dict, optional
        pass kwargs to salem.Map.set_lonlat_contours
    cbar_ax: ax, optional
        ax where to plot the colorbar
    savefig : str, optional
        save the figure to a file instead of displaying it
    savefig_kwargs : dict, optional
        the kwargs to plt.savefig
    """

    # Build on the original docstring
    plotfunc.__doc__ = '\n'.join((plotfunc.__doc__, commondoc))

    @functools.wraps(plotfunc)
    def newplotfunc(gdir, ax=None, add_colorbar=True, title=None,
                    title_comment=None, horizontal_colorbar=False,
                    lonlat_contours_kwargs=None, cbar_ax=None,
                    add_scalebar=True, savefig=None, savefig_kwargs=None,
                    **kwargs):

        dofig = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dofig = True

        mp = salem.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
        if lonlat_contours_kwargs is not None:
            mp.set_lonlat_contours(**lonlat_contours_kwargs)

        if add_scalebar:
            mp.set_scale_bar()
        out = plotfunc(gdir, ax=ax, salemmap=mp, **kwargs)

        if add_colorbar and 'cbar_label' in out:
            cbprim = out.get('cbar_primitive', mp)
            if cbar_ax:
                cb = cbprim.colorbarbase(cbar_ax)
            else:
                if horizontal_colorbar:
                    cb = cbprim.append_colorbar(ax, "bottom", size="5%", pad=0.4)
                else:
                    cb = cbprim.append_colorbar(ax, "right", size="5%", pad=0.2)
            cb.set_label(out['cbar_label'])

        if title is None:
            if 'title' not in out:
                # Make a defaut one
                title = gdir.rgi_id
                if gdir.name is not None and gdir.name != '':
                    title += ': ' + gdir.name
                out['title'] = title

            if title_comment is None:
                title_comment = out.get('title_comment', '')

            out['title'] += title_comment
            ax.set_title(out['title'])
        else:
            ax.set_title(title)

        if dofig:
            plt.tight_layout()

        if savefig is not None:
            plt.savefig(savefig, savefig_kwargs=savefig_kwargs)
            plt.close()

    return newplotfunc

@entity_task(log)
def plot_googlemap(gdir, ax=None):
    """Plots the glacier over a googlemap."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    s = salem.read_shapefile(gdir.get_filepath('outlines'))
    gm = salem.GoogleVisibleMap(np.array(s.geometry[0].exterior.xy[0]),
                                np.array(s.geometry[0].exterior.xy[1]),
                                crs=s.crs,
                                key='AIzaSyDWG_aTgfU7CeErtIzWfdGxpStTlvDXV_o')

    img = gm.get_vardata()
    cmap = salem.Map(gm.grid, countries=False, nx=gm.grid.nx)
    cmap.set_rgb(img)

    cmap.set_shapefile(gdir.get_filepath('outlines'))

    cmap.plot(ax)
    title = gdir.rgi_id
    if gdir.name is not None and gdir.name != '':
        title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@entity_task(log)
@_plot_map
def plot_domain(gdir, ax=None, salemmap=None):  # pragma: no cover
    """Plot the glacier directory.

    """

    # Files
    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    salemmap.set_cmap(cm)
    salemmap.set_plot_params(nlevels=256)
    salemmap.set_data(topo)

    crs = gdir.grid.center_grid

    geom = gdir.read_pickle('geometries', div_id=0)

    # Plot boundaries
    poly_pix = geom['polygon_pix']

    salemmap.set_geometry(poly_pix, crs=crs, fc='white',
                          alpha=0.3, zorder=2, linewidth=.2)
    for l in poly_pix.interiors:
        salemmap.set_geometry(l, crs=crs,
                              color='black', linewidth=0.5)

    salemmap.plot(ax)

    return dict(cbar_label='Alt. [m]')


@entity_task(log)
@_plot_map
def plot_centerlines(gdir, ax=None, salemmap=None, use_flowlines=False,
                     add_downstream=False):
    """Plots the centerlines of a glacier directory.

    """

    # Files
    filename = 'centerlines'
    if use_flowlines:
        filename = 'inversion_flowlines'

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    salemmap.set_cmap(cm)
    salemmap.set_plot_params(nlevels=256)
    salemmap.set_data(topo)

    crs = gdir.grid.center_grid

    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        salemmap.set_geometry(poly_pix, crs=crs, fc='white',
                             alpha=0.3, zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            salemmap.set_geometry(l, crs=crs,
                                 color='black', linewidth=0.5)

        if add_downstream:
            continue

        # plot Centerlines
        cls = gdir.read_pickle(filename, div_id=i)

        # Go in reverse order for red always being the longuest
        cls = cls[::-1]
        color = gpd.plotting.gencolor(len(cls) + 1, colormap='Set1')
        for l, c in zip(cls, color):
            salemmap.set_geometry(l.line, crs=crs, color=c,
                                 linewidth=2.5, zorder=50)
            salemmap.set_geometry(l.head, crs=gdir.grid, marker='o',
                                  markersize=60, alpha=0.8, color=c, zorder=99)

            for j in l.inflow_points:
                salemmap.set_geometry(j, crs=crs, marker='o',
                                     markersize=40, edgecolor='k', alpha=0.8,
                                     zorder=99, facecolor='none')

    if add_downstream and not gdir.is_tidewater:
        # plot Centerlines
        cls = gdir.read_pickle(filename, div_id=0)

        # Go in reverse order for red always being the longuest
        cls = cls[::-1]
        color = gpd.plotting.gencolor(len(cls) + 1, colormap='Set1')
        for l, c in zip(cls, color):
            salemmap.set_geometry(l.line, crs=crs, color=c,
                                 linewidth=2.5, zorder=50)
            salemmap.set_geometry(l.head, crs=gdir.grid, marker='o',
                                  markersize=60, alpha=0.8, color=c, zorder=99)

            for j in l.inflow_points:
                salemmap.set_geometry(j, crs=crs, marker='o',
                                     markersize=40, edgecolor='k', alpha=0.8,
                                     zorder=99, facecolor='none')

    salemmap.plot(ax)

    return dict(cbar_label='Alt. [m]')


@entity_task(log)
@_plot_map
def plot_catchment_areas(gdir, ax=None, salemmap=None):
    """Plots the catchments out of a glacier directory.
    """

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:] * np.NaN

    salemmap.set_topography(topo)

    crs = gdir.grid.center_grid
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                             linewidth=.2)
        for l in poly_pix.interiors:
            salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('centerlines', div_id=i)[::-1]
        color = gpd.plotting.gencolor(len(cls) + 1, colormap='Set1')
        for l, c in zip(cls, color):
            salemmap.set_geometry(l.line, crs=crs, color=c,
                                 linewidth=2.5, zorder=50)

        # catchment areas
        cis = gdir.read_pickle('catchment_indices', div_id=i)
        for j, ci in enumerate(cis[::-1]):
            mask[tuple(ci.T)] = j+1
    salemmap.set_cmap('Set2')
    salemmap.set_data(mask)
    salemmap.plot(ax)

    return {}


@entity_task(log)
@_plot_map
def plot_catchment_width(gdir, ax=None, salemmap=None, corrected=False,
                         add_intersects=False, add_touches=False):
    """Plots the catchment widths out of a glacier directory.
    """

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    salemmap.set_topography(topo)

    # Maybe plot touches
    xis, yis, cis = [], [], []
    crs = gdir.grid.center_grid
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                             linewidth=.2)
        for l in poly_pix.interiors:
            salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # Plot intersects
        if add_intersects and gdir.has_file('intersects', div_id=0):
            gdf = gpd.read_file(gdir.get_filepath('intersects', div_id=0))
            salemmap.set_shapefile(gdf, color='k', linewidth=3.5, zorder=3)
        if add_intersects and gdir.has_file('divides_intersects', div_id=0):
            gdf = gpd.read_file(gdir.get_filepath('divides_intersects'))
            salemmap.set_shapefile(gdf, color='k', linewidth=3.5, zorder=3)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines', div_id=i)[::-1]
        color = gpd.plotting.gencolor(len(cls) + 1, colormap='Set1')
        for l, c in zip(cls, color):
            salemmap.set_geometry(l.line, crs=crs, color=c,
                                 linewidth=2.5, zorder=50)
            if corrected:
                for wi, cur, (n1, n2) in zip(l.widths, l.line.coords,
                                             l.normals):
                    _l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                         shpg.Point(cur + wi / 2. * n2)])

                    salemmap.set_geometry(_l, crs=crs, color=c,
                                         linewidth=0.6, zorder=50)
            else:
                for wl, wi in zip(l.geometrical_widths, l.widths):
                    col = c if np.isfinite(wi) else 'grey'
                    for w in wl:
                        salemmap.set_geometry(w, crs=crs, color=col,
                                              linewidth=0.6, zorder=50)

            if add_touches:
                pok = np.where(l.touches_border)
                xi, yi = l.line.xy
                xis.append(np.asarray(xi)[pok])
                yis.append(np.asarray(yi)[pok])
                cis.append(c)

    salemmap.plot(ax)
    for xi, yi, c in zip(xis, yis, cis):
        ax.scatter(xi, yi, color=c, s=20, zorder=51)

    return {}


@entity_task(log)
@_plot_map
def plot_inversion(gdir, ax=None, salemmap=None, linewidth=3, vmax=None):
    """Plots the result of the inversion out of a glacier directory."""

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    salemmap.set_topography(topo)

    crs = gdir.grid.center_grid

    toplot_th = np.array([])
    toplot_lines = []
    vol = []
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)
        inv = gdir.read_pickle('inversion_output', div_id=i)
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                             linewidth=.2)
        for l in poly_pix.interiors:
            salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines', div_id=i)
        for l, c in zip(cls, inv):

            salemmap.set_geometry(l.line, crs=crs, color='gray',
                                 linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                     shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(l)
            vol.extend(c['volume'])

    cm = plt.cm.get_cmap('YlOrRd')
    dl = salem.DataLevels(cmap=cm, nlevels=256, data=toplot_th,
                          vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        salemmap.set_geometry(l, crs=crs, color=c,
                             linewidth=linewidth, zorder=50)
    salemmap.plot(ax)

    return dict(cbar_label='Section thickness [m]',
                cbar_primitive=dl,
                title_comment=' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9))


@entity_task(log)
@_plot_map
def plot_distributed_thickness(gdir, ax=None, salemmap=None, how=None):
    """Plots the result of the inversion out of a glacier directory.

    Method: 'alt' or 'interp'
    """

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:]

    grids_file = gdir.get_filepath('gridded_data', div_id=0)
    with netCDF4.Dataset(grids_file) as nc:
        vn = 'thickness'
        if how is not None:
            vn += '_' + how
        thick = nc.variables[vn][:]

    thick = np.where(mask, thick, np.NaN)

    salemmap.set_topography(topo)

    crs = gdir.grid.center_grid

    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    salemmap.set_cmap(plt.get_cmap('viridis'))
    salemmap.set_plot_params(nlevels=256)
    salemmap.set_data(thick)

    salemmap.plot(ax)

    return dict(cbar_label='Glacier thickness [m]')


@entity_task(log)
@_plot_map
def plot_modeloutput_map(gdir, ax=None, salemmap=None, model=None, vmax=None,
                         linewidth=3, subset=True):
    """Plots the result of the model output."""

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    geom = gdir.read_pickle('geometries', div_id=0)
    poly_pix = geom['polygon_pix']

    ds = salem.GeoDataset(gdir.grid)

    if subset:
        mlines = shpg.GeometryCollection([l.line for l in model.fls] +
                                         [poly_pix])
        ml = mlines.bounds
        corners = ((ml[0], ml[1]), (ml[2], ml[3]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ds.set_subset(corners=corners, margin=10, crs=gdir.grid)

        salemmap = salem.Map(ds.grid, countries=False, nx=gdir.grid.nx)

    salemmap.set_topography(topo, crs=gdir.grid)

    crs = gdir.grid.center_grid
    salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
    for l in poly_pix.interiors:
        salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    toplot_th = np.array([])
    toplot_lines = []

    # plot Centerlines
    cls = model.fls
    for l in cls:
        salemmap.set_geometry(l.line, crs=crs, color='gray',
                          linewidth=1.2, zorder=50)
        toplot_th = np.append(toplot_th, l.thick)
        widths = l.widths.copy()
        widths = np.where(l.thick > 0, widths, 0.)
        for wi, cur, (n1, n2) in zip(widths, l.line.coords, l.normals):
            l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                 shpg.Point(cur + wi/2. * n2)])
            toplot_lines.append(l)

    cm = plt.cm.get_cmap('YlOrRd')
    dl = salem.DataLevels(cmap=cm, nlevels=256, data=toplot_th,
                          vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        salemmap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)
    salemmap.plot(ax)

    return dict(cbar_label='Section thickness [m]',
                cbar_primitive=dl,
                title_comment=' -- year: {:d}'.format(np.int64(model.yr)))


@entity_task(log)
def plot_modeloutput_section(gdir, model=None, ax=None, title=''):
    """Plots the result of the model output along the flowline."""

    dofig = False
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
        dofig = True
    else:
        fig = plt.gcf()

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in model.fls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick>0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0,  posax.width / 3.0,
             posax.height]
    axh = fig.add_axes(posax, frameon=False)

    axh.hist(height, orientation='horizontal', range=ylim, bins=20,
             alpha=0.3, weights=area)
    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area incl. tributaries (km$^2$)')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    cls = model.fls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Where trapezoid change color
    if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
        bed_t = cls.bed_h * np.NaN
        bed_t[cls._ptrap] = cls.bed_h[cls._ptrap]
        ax.plot(x, bed_t, color='#990000', linewidth=2.5, label='Bed (Trap.)')

    # Plot glacier
    surfh = cls.surface_h
    pok = np.where(cls.thick == 0.)[0]
    if (len(pok) > 0) and (pok[0] < (len(surfh)-1)):
        surfh[pok[0]+1:] = np.NaN
    ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

    # Plot tributaries
    for i, l in zip(cls.inflow_indices, cls.inflows):
        if l.thick[-1] > 0:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='#993399',
                    markeredgecolor='k',
                    label='Tributary (active)')
        else:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='w',
                    markeredgecolor='k',
                    label='Tributary (inactive)')

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Title
    ax.set_title(title, loc='left')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.34, 1.0),
              frameon=False)


@entity_task(log)
def plot_modeloutput_section_withtrib(gdir, model=None, title=''):  # pragma: no cover
    """Plots the result of the model output along the flowline."""

    n_tribs = len(model.fls) - 1

    axs = []
    if n_tribs == 0:
        fig = plt.figure(figsize=(8, 5))
        axmaj = fig.add_subplot(111)
    elif n_tribs <= 3:
        fig = plt.figure(figsize=(14, 10))
        axmaj = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        for i in np.arange(n_tribs):
            axs.append(plt.subplot2grid((2, 3), (0, i)))
    elif n_tribs <= 6:
        fig = plt.figure(figsize=(14, 10))
        axmaj = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        for i in np.arange(n_tribs):
            j = 0
            if i >= 3:
                i -= 3
                j = 1
            axs.append(plt.subplot2grid((3, 3), (j, i)))
    else:
        raise NotImplementedError()

    for i, cls in enumerate(model.fls):
        if i == n_tribs:
            ax = axmaj
        else:
            ax = axs[i]

        x = np.arange(cls.nx) * cls.dx * cls.map_dx

        # Plot the bed
        ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

        # Where trapezoid change color
        if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
            bed_t = cls.bed_h * np.NaN
            bed_t[cls._ptrap] = cls.bed_h[cls._ptrap]
            ax.plot(x, bed_t, color='#990000', linewidth=2.5, label='Bed (Trap.)')

        # Plot glacier
        surfh = cls.surface_h
        pok = np.where(cls.thick == 0.)[0]
        if (len(pok) > 0) and (pok[0] < (len(surfh)-1)):
            surfh[pok[0]+1:] = np.NaN
        ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

        # Plot tributaries
        for i, l in zip(cls.inflow_indices, cls.inflows):
            if l.thick[-1] > 0:
                ax.plot(x[i], cls.surface_h[i], 's', color='#993399',
                        label='Tributary (active)')
            else:
                ax.plot(x[i], cls.surface_h[i], 's', color='none',
                        label='Tributary (inactive)')

        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Distance along flowline (m)')
        ax.set_ylabel('Altitude (m)')

    # Title
    plt.title(title, loc='left')
    fig.tight_layout()


@global_task
def plot_region_inversion(gdirs, ax=None, salemmap=None):
    """Plots the result of the inversion for a larger region."""

    if ax is None:
        ax = plt.gca()

    toplot_th = np.array([])
    toplot_lines = []
    vol = []
    crs_list = []

    # loop over the directories to get the data
    for gdir in gdirs:

        crs = gdir.grid.center_grid
        for i in gdir.divide_ids:
            geom = gdir.read_pickle('geometries', div_id=i)
            inv = gdir.read_pickle('inversion_output', div_id=i)
            # Plot boundaries
            poly_pix = geom['polygon_pix']
            salemmap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                                 linewidth=.2)
            for l in poly_pix.interiors:
                salemmap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

            # plot Centerlines
            cls = gdir.read_pickle('inversion_flowlines', div_id=i)
            for l, c in zip(cls, inv):

                salemmap.set_geometry(l.line, crs=crs, color='gray',
                                     linewidth=1.2, zorder=50)
                toplot_th = np.append(toplot_th, c['thick'])
                for wi, cur, (n1, n2) in zip(l.widths, l.line.coords,
                                             l.normals):
                    l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                         shpg.Point(cur + wi / 2. * n2)])
                    toplot_lines.append(l)
                    crs_list.append(crs)
                vol.extend(c['volume'])

    # plot the data
    cm = plt.cm.get_cmap('YlOrRd')
    dl = salem.DataLevels(cmap=cm, nlevels=256, data=toplot_th, vmin=0)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, crs_list):
        salemmap.set_geometry(l, crs=crs, color=c,
                              linewidth=1, zorder=50)

    salemmap.plot(ax)
    cb = dl.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Section thickness [m]')
