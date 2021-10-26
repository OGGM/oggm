# Builtins
import logging
import warnings

# External libs
import numpy as np
import pandas as pd
import shapely.geometry as shpg

# Locals
from oggm import entity_task
from oggm.core.flowline import ParabolicBedFlowline


# Module logger
log = logging.getLogger(__name__)


def read_gmip_data(gdir, area_path=None, thick_path=None, width_path=None):
    """Read area, thick and width data from the GlacierMIP files.

    Also converts areas to m2.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    area_path : str
        the path to the area bins file
    thick_path : str
        the path to the thickess bins file
    width_path : str
        the path to the widths bins file

    Returns
    -------
    a dataframe with the 'area', 'thick', 'width' columns and the elevation
    as index.
    """

    keys = ['area', 'thick', 'width']
    paths = [area_path, thick_path, width_path]

    out = pd.DataFrame()
    for k, f in zip(keys, paths):
        df = pd.read_csv(f, sep=' ', skipinitialspace=True, index_col=0,
                         skiprows=1)
        df = df[[c for c in df.columns if 'Cont_range' not in c]]

        rid = gdir.rgi_id.replace('.', '-').replace('RGI60-', 'RGIv6.0.')
        s = df.loc[rid]
        pok = np.where(s > -98)
        s = s.iloc[np.min(pok) - 100:np.max(pok) + 1]
        s = s.clip(lower=0)
        out[k] = s
    # convert to m
    out['area'] = out['area'] * 1e6
    out['width'] = out['width'] * 1e3
    out.index.name = 'elevation'
    return out


@entity_task(log, writes=['model_flowlines'])
def present_time_glacier_from_bins(gdir, data=None,
                                   area_path=None,
                                   thick_path=None,
                                   width_path=None):
    """Generates a flowline object from binned data in the glacierMIP format.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    data : pd.Dataframe
        a dataframe with the 'area', 'thick', 'width' columns and the elevation
        as index. If not provided, will be read from the path files.
    area_path : str
        the path to the area bins file
    thick_path : str
        the path to the thickess bins file
    width_path : str
        the path to the widths bins file
    """

    if data is None:
        data = read_gmip_data(gdir, area_path=area_path,
                              thick_path=thick_path,
                              width_path=width_path)

    if data.index[0] < data.index[-1]:
        data = data.loc[::-1]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        nx = len(data)
        area = np.asarray(data['area'])
        width = np.asarray(data['width'])
        thick = np.asarray(data['thick'])
        elevation = np.asarray(data.index).astype(float)
        dx_meter = area / width
        dx_meter = np.where(np.isfinite(dx_meter), dx_meter, 0)
        section = width * thick
        thick = 3/2 * section / width
        thick = np.where(np.isfinite(thick), thick, 0)
        bed_shape = 4 * thick / width ** 2

    # TODO: this is a very dirty fix -> we should rather interpolate,
    # AND we should be really careful about flat bed shapes
    bed_shape = np.where(np.isfinite(bed_shape), bed_shape,
                         np.nanmean(bed_shape))

    coords = np.arange(0, nx - 0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

    nfl = ParabolicBedFlowline(line=line, dx=dx_meter, map_dx=1,
                               surface_h=elevation, bed_h=elevation-thick,
                               bed_shape=bed_shape, rgi_id=gdir.rgi_id)

    # Write the data
    gdir.write_pickle([nfl], 'model_flowlines')
    return [nfl]
