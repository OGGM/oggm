"""Useful plotting functions"""
import os
import functools
import logging
from collections import OrderedDict
import itertools

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import salem
import shapely.geometry as shpg
from matplotlib import cm as colormap

try:
    # optional python-colorspace dependency for better (HCL-based) colormaps
    from colorspace import sequential_hcl
    USE_HCL_CMAP = True
except ImportError:
    USE_HCL_CMAP = False

from oggm.core.flowline import FileModel
from oggm import cfg, utils

# Module logger
log = logging.getLogger(__name__)

# Set global colormaps
ALTITUDE_CMAP = colormap.terrain
if USE_HCL_CMAP:
    CMAP_DIVS = 100  # number of discrete colours from continuous colormaps
    THICKNESS_CMAP = sequential_hcl("Blue-Yellow", rev=True).cmap(CMAP_DIVS)
    SECTION_THICKNESS_CMAP = THICKNESS_CMAP
    GLACIER_THICKNESS_CMAP = THICKNESS_CMAP
else:
    SECTION_THICKNESS_CMAP = plt.cm.get_cmap('YlOrRd')
    GLACIER_THICKNESS_CMAP = plt.get_cmap('viridis')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Remove extreme colors from a colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def gencolor_generator(n, cmap='Set1'):
    """ Color generator intended to work with qualitative color scales."""
    # don't use more than 9 discrete colors
    n_colors = min(n, 9)
    cmap = colormap.get_cmap(cmap, n_colors)
    colors = cmap(range(n_colors))
    for i in range(n):
        yield colors[i % n_colors]


def gencolor(n, cmap='Set1'):

    if isinstance(cmap, str):
        return gencolor_generator(n, cmap=cmap)
    else:
        return itertools.cycle(cmap)


def _plot_map(plotfunc):
    """
    Decorator for common salem.Map plotting logic
    """
    commondoc = """

    Parameters
    ----------
    gdirs : [] or GlacierDirectory, required
        A single GlacierDirectory or a list of gdirs to plot.
    ax : matplotlib axes object, optional
        If None, uses own axis
    smap : Salem Map object, optional
        If None, makes a map from the first gdir in the list
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
    autosave : bool, optional
        set to True to override to a default savefig filename (useful
        for multiprocessing)
    figsize : tuple, optional
        size of the figure
    savefig : str, optional
        save the figure to a file instead of displaying it
    savefig_kwargs : dict, optional
        the kwargs to plt.savefig
    """

    # Build on the original docstring
    plotfunc.__doc__ = '\n'.join((plotfunc.__doc__, commondoc))

    @functools.wraps(plotfunc)
    def newplotfunc(gdirs, ax=None, smap=None, add_colorbar=True, title=None,
                    title_comment=None, horizontal_colorbar=False,
                    lonlat_contours_kwargs=None, cbar_ax=None, autosave=False,
                    add_scalebar=True, figsize=None, savefig=None,
                    savefig_kwargs=None,
                    **kwargs):

        dofig = False
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            dofig = True

        # Cast to list
        gdirs = utils.tolist(gdirs)

        if smap is None:
            mp = salem.Map(gdirs[0].grid, countries=False,
                           nx=gdirs[0].grid.nx)
        else:
            mp = smap

        if lonlat_contours_kwargs is not None:
            mp.set_lonlat_contours(**lonlat_contours_kwargs)

        if add_scalebar:
            mp.set_scale_bar()
        out = plotfunc(gdirs, ax=ax, smap=mp, **kwargs)

        if add_colorbar and 'cbar_label' in out:
            cbprim = out.get('cbar_primitive', mp)
            if cbar_ax:
                cb = cbprim.colorbarbase(cbar_ax)
            else:
                if horizontal_colorbar:
                    cb = cbprim.append_colorbar(ax, "bottom", size="5%",
                                                pad=0.4)
                else:
                    cb = cbprim.append_colorbar(ax, "right", size="5%",
                                                pad=0.2)
            cb.set_label(out['cbar_label'])

        if title is None:
            if 'title' not in out:
                # Make a defaut one
                title = ''
                if len(gdirs) == 1:
                    gdir = gdirs[0]
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

        if autosave:
            savefig = os.path.join(cfg.PATHS['working_dir'], 'plots')
            utils.mkdir(savefig)
            savefig = os.path.join(savefig, plotfunc.__name__ + '_' +
                                   gdirs[0].rgi_id + '.png')

        if savefig is not None:
            plt.savefig(savefig, **savefig_kwargs)
            plt.close()

    return newplotfunc


def plot_googlemap(gdirs, ax=None, figsize=None):
    """Plots the glacier(s) over a googlemap."""

    dofig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        dofig = True

    gdirs = utils.tolist(gdirs)

    xx, yy = [], []
    for gdir in gdirs:
        xx.extend(gdir.extent_ll[0])
        yy.extend(gdir.extent_ll[1])

    gm = salem.GoogleVisibleMap(xx, yy,
                                key='AIzaSyDWG_aTgfU7CeErtIzWfdGxpStTlvDXV_o')

    img = gm.get_vardata()
    cmap = salem.Map(gm.grid, countries=False, nx=gm.grid.nx)
    cmap.set_rgb(img)

    for gdir in gdirs:
        cmap.set_shapefile(gdir.read_shapefile('outlines'))

    cmap.plot(ax)
    title = ''
    if len(gdirs) == 1:
        title = gdir.rgi_id
        if gdir.name is not None and gdir.name != '':
            title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@_plot_map
def plot_domain(gdirs, ax=None, smap=None):
    """Plot the glacier directory."""

    # Files
    gdir = gdirs[0]

    topo = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()
    try:
        smap.set_data(topo)
    except ValueError:
        pass

    cm = truncate_colormap(ALTITUDE_CMAP, minval=0.25, maxval=1.0, n=256)
    smap.set_cmap(cm)
    smap.set_plot_params(nlevels=256)

    for gdir in gdirs:
        crs = gdir.grid.center_grid

        try:
            geom = gdir.read_pickle('geometries')

            # Plot boundaries
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='white',
                              alpha=0.3, zorder=2, linewidth=.2)
            for l in poly_pix.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'))

    smap.plot(ax)

    return dict(cbar_label='Alt. [m]')


@_plot_map
def plot_centerlines(gdirs, ax=None, smap=None, use_flowlines=False,
                     add_downstream=False, lines_cmap='Set1',
                     add_line_index=False, use_model_flowlines=False):
    """Plots the centerlines of a glacier directory."""

    if add_downstream and not use_flowlines:
        raise ValueError('Downstream lines can be plotted with flowlines only')

    # Files
    filename = 'centerlines'
    if use_model_flowlines:
        filename = 'model_flowlines'
    elif use_flowlines:
        filename = 'inversion_flowlines'

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    cm = truncate_colormap(ALTITUDE_CMAP, minval=0.25, maxval=1.0, n=256)
    smap.set_cmap(cm)
    smap.set_plot_params(nlevels=256)
    smap.set_data(topo)
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        smap.set_geometry(poly_pix, crs=crs, fc='white',
                          alpha=0.3, zorder=2, linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs,
                              color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle(filename)

        # Go in reverse order for red always being the longuest
        cls = cls[::-1]
        nl = len(cls)
        color = gencolor(len(cls) + 1, cmap=lines_cmap)
        for i, (l, c) in enumerate(zip(cls, color)):
            if add_downstream and not gdir.is_tidewater and l is cls[0]:
                line = gdir.read_pickle('downstream_line')['full_line']
            else:
                line = l.line

            smap.set_geometry(line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)

            text = '{}'.format(nl - i - 1) if add_line_index else None
            smap.set_geometry(l.head, crs=gdir.grid, marker='o',
                              markersize=60, alpha=0.8, color=c, zorder=99,
                              text=text)

            for j in l.inflow_points:
                smap.set_geometry(j, crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='none')

    smap.plot(ax)
    return dict(cbar_label='Alt. [m]')


@_plot_map
def plot_catchment_areas(gdirs, ax=None, smap=None, lines_cmap='Set1',
                         mask_cmap='Set2'):
    """Plots the catchments out of a glacier directory.
    """

    gdir = gdirs[0]
    if len(gdirs) > 1:
        raise NotImplementedError('Cannot plot a list of gdirs (yet)')

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:] * np.NaN

    smap.set_topography(topo)

    crs = gdir.grid.center_grid
    geom = gdir.read_pickle('geometries')

    # Plot boundaries
    poly_pix = geom['polygon_pix']
    smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                      linewidth=.2)
    for l in poly_pix.interiors:
        smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    # plot Centerlines
    cls = gdir.read_pickle('centerlines')[::-1]
    color = gencolor(len(cls) + 1, cmap=lines_cmap)
    for l, c in zip(cls, color):
        smap.set_geometry(l.line, crs=crs, color=c,
                          linewidth=2.5, zorder=50)

    # catchment areas
    cis = gdir.read_pickle('geometries')['catchment_indices']
    for j, ci in enumerate(cis[::-1]):
        mask[tuple(ci.T)] = j+1

    smap.set_cmap(mask_cmap)
    smap.set_data(mask)
    smap.plot(ax)

    return {}


@_plot_map
def plot_catchment_width(gdirs, ax=None, smap=None, corrected=False,
                         add_intersects=False, add_touches=False,
                         lines_cmap='Set1'):
    """Plots the catchment widths out of a glacier directory.
    """

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    # Maybe plot touches
    xis, yis, cis = [], [], []
    ogrid = smap.grid

    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # Plot intersects
        if add_intersects and gdir.has_file('intersects'):
            gdf = gdir.read_shapefile('intersects')
            smap.set_shapefile(gdf, color='k', linewidth=3.5, zorder=3)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')[::-1]
        color = gencolor(len(cls) + 1, cmap=lines_cmap)
        for l, c in zip(cls, color):
            smap.set_geometry(l.line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)
            if corrected:
                for wi, cur, (n1, n2) in zip(l.widths, l.line.coords,
                                             l.normals):
                    _l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                          shpg.Point(cur + wi / 2. * n2)])

                    smap.set_geometry(_l, crs=crs, color=c,
                                      linewidth=0.6, zorder=50)
            else:
                for wl, wi in zip(l.geometrical_widths, l.widths):
                    col = c if np.isfinite(wi) else 'grey'
                    for w in wl:
                        smap.set_geometry(w, crs=crs, color=col,
                                          linewidth=0.6, zorder=50)

            if add_touches:
                pok = np.where(l.is_rectangular)
                xi, yi = l.line.xy
                xi, yi = ogrid.transform(np.asarray(xi)[pok],
                                         np.asarray(yi)[pok], crs=crs)
                xis.append(xi)
                yis.append(yi)
                cis.append(c)

    smap.plot(ax)
    for xi, yi, c in zip(xis, yis, cis):
        ax.scatter(xi, yi, color=c, s=20, zorder=51)

    return {}


@_plot_map
def plot_inversion(gdirs, ax=None, smap=None, linewidth=3, vmax=None):
    """Plots the result of the inversion out of a glacier directory."""

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_th = np.array([])
    toplot_lines = []
    toplot_crs = []
    vol = []
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')
        inv = gdir.read_pickle('inversion_output')
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')
        for l, c in zip(cls, inv):

            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                line = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                        shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(line)
                toplot_crs.append(crs)
            vol.extend(c['volume'])

    dl = salem.DataLevels(cmap=SECTION_THICKNESS_CMAP, nlevels=256,
                          data=toplot_th, vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)

    smap.plot(ax)
    return dict(cbar_label='Section thickness [m]',
                cbar_primitive=dl,
                title_comment=' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9))


@_plot_map
def plot_distributed_thickness(gdirs, ax=None, smap=None, varname_suffix=''):
    """Plots the result of the inversion out of a glacier directory.

    Method: 'alt' or 'interp'
    """

    gdir = gdirs[0]
    if len(gdirs) > 1:
        raise NotImplementedError('Cannot plot a list of gdirs (yet)')

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:]

    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file) as nc:
        import warnings
        with warnings.catch_warnings():
            # https://github.com/Unidata/netcdf4-python/issues/766
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            vn = 'distributed_thickness' + varname_suffix
            thick = nc.variables[vn][:]

    thick = np.where(mask, thick, np.NaN)

    smap.set_topography(topo)

    crs = gdir.grid.center_grid

    geom = gdir.read_pickle('geometries')

    # Plot boundaries
    poly_pix = geom['polygon_pix']
    smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
    for l in poly_pix.interiors:
        smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    smap.set_cmap(GLACIER_THICKNESS_CMAP)
    smap.set_plot_params(nlevels=256)
    smap.set_data(thick)

    smap.plot(ax)

    return dict(cbar_label='Glacier thickness [m]')


@_plot_map
def plot_modeloutput_map(gdirs, ax=None, smap=None, model=None,
                         vmax=None, linewidth=3, filesuffix='',
                         modelyr=None):
    """Plots the result of the model output."""

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_th = np.array([])
    toplot_lines = []
    toplot_crs = []

    if model is None:
        models = []
        for gdir in gdirs:
            model = FileModel(gdir.get_filepath('model_run',
                                                filesuffix=filesuffix))
            model.run_until(modelyr)
            models.append(model)
    else:
        models = utils.tolist(model)
    for gdir, model in zip(gdirs, models):
        geom = gdir.read_pickle('geometries')
        poly_pix = geom['polygon_pix']

        crs = gdir.grid.center_grid
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)

        if isinstance(poly_pix, shpg.MultiPolygon):
            for _poly in poly_pix:
                for l in _poly.interiors:
                    smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        else:
            for l in poly_pix.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = model.fls
        for l in cls:
            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, l.thick)
            widths = l.widths.copy()
            widths = np.where(l.thick > 0, widths, 0.)
            for wi, cur, (n1, n2) in zip(widths, l.line.coords, l.normals):
                line = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                        shpg.Point(cur + wi/2. * n2)])
                toplot_lines.append(line)
                toplot_crs.append(crs)

    dl = salem.DataLevels(cmap=SECTION_THICKNESS_CMAP, nlevels=256,
                          data=toplot_th, vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)
    smap.plot(ax)
    return dict(cbar_label='Section thickness [m]',
                cbar_primitive=dl,
                title_comment=' -- year: {:d}'.format(np.int64(model.yr)))


def plot_modeloutput_section(model=None, ax=None, title=''):
    """Plots the result of the model output along the flowline."""

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    else:
        fig = plt.gcf()

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in model.fls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
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
        pt = cls.is_trapezoid & (~cls.is_rectangular)
        bed_t[pt] = cls.bed_h[pt]
        ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                label='Bed (Trap.)')
        bed_t = cls.bed_h * np.NaN
        bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
        ax.plot(x, bed_t, color='crimson', linewidth=2.5, label='Bed (Rect.)')

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
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(1.34, 1.0),
              frameon=False)


def plot_modeloutput_section_withtrib(model=None, fig=None, title=''):
    """Plots the result of the model output along the flowline."""

    n_tribs = len(model.fls) - 1

    axs = []
    if n_tribs == 0:
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
        axmaj = fig.add_subplot(111)
    elif n_tribs <= 3:
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        axmaj = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        for i in np.arange(n_tribs):
            axs.append(plt.subplot2grid((2, 3), (0, i)))
    elif n_tribs <= 6:
        if fig is None:
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
            pt = cls.is_trapezoid & (~cls.is_rectangular)
            bed_t[pt] = cls.bed_h[pt]
            ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                    label='Bed (Trap.)')
            bed_t = cls.bed_h * np.NaN
            bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
            ax.plot(x, bed_t, color='crimson', linewidth=2.5,
                    label='Bed (Rect.)')

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

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              loc='best', frameon=False)
    fig.tight_layout()
