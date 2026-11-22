import numpy as np

from oggm.exceptions import InvalidWorkflowError


def _raise_if_no_data(gdir, data, product):
    """Raise if no valid value of a product lands on the glacier grid.

    A product file covering the glacier region but with nothing valid over
    this glacier is treated like a missing file: the shop tasks raise
    instead of writing an all-NaN variable.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory being processed
    data : ndarray, masked array or None
        the data reprojected to the glacier grid (None if no file had data)
    product : str
        the product name, for the error message
    """
    if data is None or not np.isfinite(np.ma.filled(data, np.nan)).any():
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no valid {product} data '
                                   f'on the glacier grid.')
