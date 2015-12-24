"""Useful plotting functions"""
from __future__ import division
from six.moves import zip

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import matplotlib.colors as colors
import warnings

from descartes import PolygonPatch
import shapely.geometry as shpg
import glob
import os
import pickle
import numpy as np
import netCDF4
import salem


import cleo

# Local imports
import oggm.conf as cfg


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_googlemap(glacierdir, ax=None):
    """Plots the glacier over a googlemap."""

    # TODO: center grid or corner grid???
    crs = glacierdir.grid.center_grid

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    s = salem.utils.read_shapefile(glacierdir.get_filepath('outlines'))
    gm = salem.GoogleVisibleMap(np.array(s.geometry[0].exterior.xy[0]),
                                np.array(s.geometry[0].exterior.xy[1]),
                                src=s.crs)

    img = gm.get_vardata()
    cmap = cleo.Map(gm.grid, countries=False, nx=gm.grid.nx)
    cmap.set_lonlat_countours(0.02)
    cmap.set_rgb(img)

    cmap.set_shapefile(glacierdir.get_filepath('outlines'))

    cmap.plot(ax)
    ax.set_title(glacierdir.rgi_id)

    if dofig:
        plt.tight_layout()


def plot_centerlines(glacierdir, ax=None, use_flowlines=False,
                     add_downstream=False):
    """Plots the centerlines of a glacier directory."""

    # Files
    filename = 'centerlines'
    if use_flowlines:
        filename = 'inversion_flowlines'

    nc = netCDF4.Dataset(glacierdir.get_filepath('gridded_data'))
    topo = nc.variables['topo'][:]
    nc.close()

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    mp = cleo.Map(glacierdir.grid, countries=False, nx=glacierdir.grid.nx)
    mp.set_lonlat_countours(0.02)
    cm = truncate_colormap(colormap.terrain, minval=0.25, maxval=1.0, n=256)
    mp.set_cmap(cm)
    mp.set_plot_params(nlevels=256)
    mp.set_data(topo, interp='linear')

    # TODO: center grid or corner grid???
    crs = glacierdir.grid.center_grid

    for i in glacierdir.divide_ids:
        geom = glacierdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        mp.set_geometry(poly_pix, crs=crs, fc='white',
                         alpha=0.3, zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs,
                              color='black', linewidth=0.5)

        # plot Centerlines
        cls = glacierdir.read_pickle(filename, div_id=i)

        # Go in reverse order for red always being the longuest
        cls = cls[::-1]
        color = gpd.plotting.gencolor(len(cls)+1, colormap='Set1')
        for l, c in zip(cls, color):
            mp.set_geometry(l.line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)
            mp.set_geometry(l.head, crs=glacierdir.grid, marker='o',
                              markersize=60, alpha=0.8, color=c, zorder=99)

            for j in l.inflow_points:
                mp.set_geometry(j, crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='none')

        if add_downstream:
            line = glacierdir.read_pickle('downstream_line', div_id=i)
            mp.set_geometry(line, crs=crs, color='red', linewidth=2.5,
                              zorder=50)

            mp.set_geometry(shpg.Point(line.coords[0]), crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='w')

    mp.plot(ax)
    cb = mp.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Alt. [m]')
    ax.set_title(glacierdir.rgi_id)

    if dofig:
        plt.tight_layout()


def plot_catchment_width(glacierdir, ax=None, corrected=False):
    """Plots the catchment widths out of a glacier directory."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    nc = netCDF4.Dataset(glacierdir.get_filepath('gridded_data'))
    topo = nc.variables['topo'][:]
    nc.close()

    mp = cleo.Map(glacierdir.grid, countries=False, nx=glacierdir.grid.nx)
    mp.set_lonlat_countours(0.02)
    mp.set_topography(topo)

    # TODO: center grid or corner grid???
    crs = glacierdir.grid.center_grid
    for i in glacierdir.divide_ids:
        geom = glacierdir.read_pickle('geometries', div_id=i)

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = glacierdir.read_pickle('inversion_flowlines', div_id=i)[::-1]
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


def plot_inversion(glacierdir, ax=None):
    """Plots the result of the inversion out of a glacier directory."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    nc = netCDF4.Dataset(glacierdir.get_filepath('gridded_data'))
    topo = nc.variables['topo'][:]
    nc.close()

    mp = cleo.Map(glacierdir.grid, countries=False, nx=glacierdir.grid.nx)
    mp.set_lonlat_countours(0.02)
    mp.set_topography(topo)

    # TODO: center grid or corner grid???
    crs = glacierdir.grid.center_grid

    toplot_th = np.array([])
    toplot_lines = []
    for i in glacierdir.divide_ids:
        geom = glacierdir.read_pickle('geometries', div_id=i)
        inv = glacierdir.read_pickle('inversion_output', div_id=i)
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        mp.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            mp.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = glacierdir.read_pickle('inversion_flowlines', div_id=i)
        for l, c in zip(cls, inv):

            mp.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                     shpg.Point(cur + wi/2. * n2)])
                toplot_lines.append(l)


    cm = plt.cm.get_cmap('YlOrRd')
    dl = cleo.DataLevels(cmap=cm, nlevels=256, data=toplot_th, vmin=0)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        mp.set_geometry(l, crs=crs, color=c,
                          linewidth=3, zorder=50)
    mp.plot(ax)
    cb = dl.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Glacier thickness [m]')
    ax.set_title(glacierdir.rgi_id)
    if dofig:
        plt.tight_layout()


def plot_modeloutput(glacierdir, model, ax=None):  # pragma: no cover
    """Plots the result of the model output."""

    dofig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dofig = True

    nc = netCDF4.Dataset(glacierdir.get_filepath('gridded_data'))
    topo = nc.variables['topo'][:]
    nc.close()

    geom = glacierdir.read_pickle('geometries', div_id=0)
    poly_pix = geom['polygon_pix']

    ds = salem.GeoDataset(glacierdir.grid)
    mlines = shpg.GeometryCollection([l.line for l in model.fls] + [poly_pix])
    ml = mlines.bounds
    corners = ((ml[0], ml[1]), (ml[2], ml[3]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        ds.set_subset(corners=corners, margin=10, crs=glacierdir.grid)

    mp = cleo.Map(ds.grid, countries=False, nx=glacierdir.grid.nx)
    mp.set_lonlat_countours(0.02)
    mp.set_topography(topo, crs=glacierdir.grid)

    # TODO: center grid or corner grid???
    crs = glacierdir.grid.center_grid
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
        for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
            l = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                 shpg.Point(cur + wi/2. * n2)])
            toplot_lines.append(l)


    cm = plt.cm.get_cmap('YlOrRd')
    dl = cleo.DataLevels(cmap=cm, nlevels=256, data=toplot_th, vmin=0)
    colors = dl.to_rgb()
    for l, c in zip(toplot_lines, colors):
        mp.set_geometry(l, crs=crs, color=c,
                          linewidth=3, zorder=50)
    mp.plot(ax)
    cb = dl.append_colorbar(ax, "right", size="5%", pad=0.2)
    cb.set_label('Glacier thickness [m]')
    tit = ' -- year: {:d}'.format(np.int64(model.yr))
    ax.set_title(glacierdir.rgi_id + tit)

    if dofig:
        plt.tight_layout()