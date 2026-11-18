"""IceBoost v2.0 ice thickness (Maffezzoli et al.) for OGGM glaciers.

The data is available per glacier (RGI62, RGI70G, Zenodo record 17724512)
and per glacier complex (RGI70C, Zenodo record 21220985). We mirror these
files on the OGGM cluster, one GeoTiff per glacier.
"""
import logging
import warnings

import numpy as np

try:
    import salem
except ImportError:
    pass

try:
    import rasterio
except ImportError:
    pass

from oggm import utils
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError

# Module logger
log = logging.getLogger(__name__)

default_base_url = ('https://cluster.klima.uni-bremen.de/~oggm/'
                    'ice_thickness/iceboost_v2/')

# Bands in the IceBoost files (thickness and its uncertainty, in m)
_BANDS = {'thickness': 1, 'err': 2}


def _get_url(gdir):
    """The URL of the IceBoost file for this glacier."""
    if gdir.rgi_version[0] == '6':
        product = 'RGI62'
    elif gdir.rgi_version in ['70G', '70C']:
        product = 'RGI' + gdir.rgi_version
    else:
        raise InvalidWorkflowError('IceBoost data is only available for '
                                   'RGI6 and RGI7 (glaciers and complexes), '
                                   f'not RGI version {gdir.rgi_version} '
                                   f'({gdir.rgi_id})')
    return (f'{default_base_url}{product}/rgi{int(gdir.rgi_region)}/'
            f'{gdir.rgi_id}.tif')


@utils.entity_task(log, writes=['gridded_data'])
def iceboost_to_gdir(gdir, add_error=False, interp='linear',
                     match_volume=True):
    """Add the IceBoost v2.0 ice thickness to this glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    add_error : bool
        add the ice thickness uncertainty as well
    interp : str
        the interpolation used to reproject the data onto the glacier grid:
        'linear' (default) or 'nearest'. The IceBoost files are NaN where
        there is no ice (outside the glacier, nunataks): with 'linear',
        these NaNs are set to zero before the interpolation (otherwise
        the glacier edges would be lost). With 'nearest', they stay NaN.
    match_volume : bool
        scale the reprojected thickness (one factor for the whole glacier)
        so that its volume within the glacier mask matches the volume
        reported in the IceBoost file (default). The factor is stored
        in the ``volume_correction_factor`` attribute of the variable.
        The uncertainty is not scaled.

    Notes
    -----
    IceBoost computes the thickness on points generated inside the glacier
    outline (outside of nunataks), at 100 m spacing or finer for small
    glaciers. The glacier volume reported in the files (and in the OGGM
    reference tables) is the mean thickness of these points times the
    (geodesic) outline area. The GeoTiff files, however, keep all grid
    cells touched by the outline: their gridded area and volume are larger
    than the reported ones (globally for RGI6, about +9% area and +3%
    volume). Summing the grid therefore overestimates the volume, hence
    the ``match_volume`` option. For RGI7 glacier complexes, the reported
    volume is the sum of the volumes of the RGI7 glaciers they contain.
    See https://github.com/nmaffe/iceboost/issues/21 for details.
    """

    if interp not in ['linear', 'nearest']:
        raise InvalidParamsError(f"interp must be 'linear' or 'nearest', "
                                 f"got {interp}")

    url = _get_url(gdir)
    input_file = utils.file_downloader(url)
    if input_file is None:
        raise InvalidWorkflowError(f'No IceBoost file found for {gdir.rgi_id}'
                                   f' at {url}')

    dsb = salem.GeoTiff(input_file)
    with rasterio.open(input_file) as src:
        ref_vol_km3 = float(src.tags()['volume'])

    var_names = ['thickness', 'err'] if add_error else ['thickness']
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        for var in var_names:
            data = dsb.get_vardata(_BANDS[var]).astype(np.float64)
            data[~np.isfinite(data)] = 0 if interp == 'linear' else np.nan

            with warnings.catch_warnings():
                # This can trigger an out of bounds warning
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                                        message='.*out of bounds.*')
                r_data = gdir.grid.map_gridded_data(data, dsb.grid,
                                                    interp=interp)
            r_data = r_data.filled(np.nan)

            vn = 'iceboost_ice_thickness'
            ln = 'Ice thickness from IceBoost v2.0'
            factor = 1.
            if var == 'err':
                vn += '_err'
                ln = 'Ice thickness uncertainty from IceBoost v2.0'
            elif match_volume:
                mask = np.ma.filled(nc.variables['glacier_mask'][:], 0) == 1
                vol_km3 = np.nansum(r_data[mask]) * gdir.grid.dx**2 * 1e-9
                if vol_km3 > 0:
                    factor = ref_vol_km3 / vol_km3
                    r_data *= factor
                elif ref_vol_km3 > 0:
                    log.warning(f'({gdir.rgi_id}) no IceBoost thickness '
                                f'within the glacier mask: cannot match '
                                f'the IceBoost volume.')

            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True,
                                      fill_value=np.nan)
            v.units = 'm'
            v.long_name = ln
            v.data_source = url
            if var == 'thickness':
                v.iceboost_volume_km3 = ref_vol_km3
                v.volume_correction_factor = factor
            v[:] = r_data.astype(np.float32)
