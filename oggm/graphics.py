"""Useful plotting functions"""
from __future__ import division
from six.moves import zip

from collections import OrderedDict
import warnings
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from matplotlib import transforms
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

from descartes import PolygonPatch
import shapely.geometry as shpg
import glob
import os
import numpy as np
import netCDF4
import salem

from oggm.utils import entity_task

import cleo

# Local imports
import oggm.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


@entity_task(log)
def plot_googlemap(gdir, ax=None):
    """Plots the glacier over a googlemap."""

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    s = salem.utils.read_shapefile(gdir.get_filepath('outlines'))
    gm = salem.GoogleVisibleMap(np.array(s.geometry[0].exterior.xy[0]),
                                np.array(s.geometry[0].exterior.xy[1]),
                                src=s.crs)

    img = gm.get_vardata()[..., 0:3]  # sometimes there is an alpha
    cmap = cleo.Map(gm.grid, countries=False, nx=gm.grid.nx)
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
def plot_domain(gdir, ax=None):  # pragma: no cover
    """Plot the glacier directory."""

    # Files
    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    mp = cleo.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    mp.set_cmap(cm)
    mp.set_plot_params(nlevels=256)
    mp.set_data(topo)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid

    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        mp.set_geometry(poly_pix, crs=crs, fc='white',
                         alpha=0.3, zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs,
                              color='black', linewidth=0.5)

    mp.plot(ax)
    cb = mp.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Alt. [m]')
    title = gdir.rgi_id
    if gdir.name is not None and gdir.name != '':
        title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_centerlines(gdir, ax=None, use_flowlines=False,
                     add_downstream=False):
    """Plots the centerlines of a glacier directory."""

    # Files
    filename = 'centerlines'
    if use_flowlines:
        filename = 'inversion_flowlines'

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    mp = cleo.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    mp.set_cmap(cm)
    mp.set_plot_params(nlevels=256)
    mp.set_data(topo)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid

    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        mp.set_geometry(poly_pix, crs=crs, fc='white',
                         alpha=0.3, zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs,
                              color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle(filename, div_id=i)

        # Go in reverse order for red always being the longuest
        cls = cls[::-1]
        color = gpd.plotting.gencolor(len(cls)+1, colormap='Set1')
        for l, c in zip(cls, color):
            mp.set_geometry(l.line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)
            mp.set_geometry(l.head, crs=gdir.grid, marker='o',
                            markersize=60, alpha=0.8, color=c, zorder=99)

            for j in l.inflow_points:
                mp.set_geometry(j, crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='none')

        if add_downstream:
            line = gdir.read_pickle('downstream_line', div_id=i)
            mp.set_geometry(line, crs=crs, color='red', linewidth=2.5,
                              zorder=50)

            mp.set_geometry(shpg.Point(line.coords[0]), crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='w')

    mp.plot(ax)
    cb = mp.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Alt. [m]')
    title = gdir.rgi_id
    if gdir.name is not None and gdir.name != '':
        title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_catchment_width(gdir, ax=None, corrected=False):
    """Plots the catchment widths out of a glacier directory."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    mp = cleo.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
    mp.set_topography(topo)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines', div_id=i)[::-1]
        color = gpd.plotting.gencolor(len(cls)+1, colormap='Set1')
        for l, c in zip(cls, color):
            mp.set_geometry(l.line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)
            if corrected:
                for wi, cur, (n1, n2) in zip(l.widths, l.line.coords,
                                             l.normals):
                    l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                         shpg.Point(cur + wi/2. * n2)])

                    mp.set_geometry(l, crs=crs, color=c,
                                      linewidth=0.6, zorder=50)
            else:
                for wl, wi in zip(l.geometrical_widths, l.widths):
                    col = c if np.isfinite(wi) else 'grey'
                    for w in wl:
                        mp.set_geometry(w, crs=crs, color=col,
                                          linewidth=0.6, zorder=50)

    mp.plot(ax)
    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_inversion(gdir, ax=None, add_title_comment=''):
    """Plots the result of the inversion out of a glacier directory."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    mp = cleo.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
    mp.set_topography(topo)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid

    toplot_th = np.array([])
    toplot_lines = []
    vol = []
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)
        inv = gdir.read_pickle('inversion_output', div_id=i)
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines', div_id=i)
        for l, c in zip(cls, inv):

            mp.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                     shpg.Point(cur + wi/2. * n2)])
                toplot_lines.append(l)
            vol.extend(c['volume'])

    cm = plt.cm.get_cmap('YlOrRd')
    dl = cleo.DataLevels(cmap=cm, nlevels=256, data=toplot_th, vmin=0)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        mp.set_geometry(l, crs=crs, color=c,
                          linewidth=3, zorder=50)
    mp.plot(ax)
    cb = dl.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Glacier thickness [m]')
    title = gdir.rgi_id
    if gdir.name is not None and gdir.name != '':
        title += ': ' + gdir.name
    title += add_title_comment
    title += ' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9)
    ax.set_title(title)
    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_distributed_thickness(gdir, ax=None, how=None):
    """Plots the result of the inversion out of a glacier directory.

    Method: 'alt' or 'interp'
    """

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

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

    mp = cleo.Map(gdir.grid, countries=False, nx=gdir.grid.nx)
    mp.set_topography(topo)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid

    toplot_th = np.array([])
    toplot_lines = []
    for i in gdir.divide_ids:
        geom = gdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    mp.set_cmap(plt.get_cmap('viridis'))
    mp.set_plot_params(nlevels=256)
    mp.set_data(thick)

    mp.plot(ax)
    cb = mp.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Thick. [m]')
    title = gdir.rgi_id
    if gdir.name is not None and gdir.name != '':
        title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_modeloutput_map(gdir, model=None, ax=None, vmax=None):
    """Plots the result of the model output."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    geom = gdir.read_pickle('geometries', div_id=0)
    poly_pix = geom['polygon_pix']

    ds = salem.GeoDataset(gdir.grid)
    mlines = shpg.GeometryCollection([l.line for l in model.fls] + [poly_pix])
    ml = mlines.bounds
    corners = ((ml[0], ml[1]), (ml[2], ml[3]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        ds.set_subset(corners=corners, margin=10, crs=gdir.grid)

    mp = cleo.Map(ds.grid, countries=False, nx=gdir.grid.nx)
    mp.set_topography(topo, crs=gdir.grid)

    # TODO: center grid or corner grid???
    crs = gdir.grid.center_grid
    mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
    for l in poly_pix.interiors:
        mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    toplot_th = np.array([])
    toplot_lines = []

    # plot Centerlines
    cls = model.fls
    for l in cls:
        mp.set_geometry(l.line, crs=crs, color='gray',
                          linewidth=1.2, zorder=50)
        toplot_th = np.append(toplot_th, l.thick)
        widths = l.widths.copy()
        widths = np.where(l.thick > 0, widths, 0.)
        for wi, cur, (n1, n2) in zip(widths, l.line.coords, l.normals):
            l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                 shpg.Point(cur + wi/2. * n2)])
            toplot_lines.append(l)

    cm = plt.cm.get_cmap('YlOrRd')
    dl = cleo.DataLevels(cmap=cm, nlevels=256, data=toplot_th, vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        mp.set_geometry(l, crs=crs, color=c,
                          linewidth=3, zorder=50)
    mp.plot(ax)
    cb = dl.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Glacier thickness [m]')
    tit = ' -- year: {:d}'.format(np.int64(model.yr))
    ax.set_title(gdir.rgi_id + tit)

    if dofig:
        plt.tight_layout()


@entity_task(log)
def plot_modeloutput_section(gdir, model=None, ax=None, title=''):
    """Plots the result of the model output along the flowline."""

    dofig = False
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
        dofig = True

    # Compute area histo
    area = np.array([])
    height = np.array([])
    for cls in model.fls:
        area = np.concatenate((area, cls.widths_m * cls.dx_meter * 1e-6))
        height = np.concatenate((height, cls.surface_h))
    ylim = [height.min(), height.max()]

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
    if hasattr(cls, '_dot') and cls._dot:
        bed_t = cls.bed_h * np.NaN
        bed_t[cls._pt] = cls.bed_h[cls._pt]
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
        if hasattr(cls, '_dot') and cls._dot:
            bed_t = cls.bed_h * np.NaN
            bed_t[cls._pt] = cls.bed_h[cls._pt]
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
