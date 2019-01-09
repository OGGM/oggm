"""Fallback functions which can be passed to the entity decorator."""

# External libs
import numpy as np


def fb_local_t_star(gdir):
    """A Fallback function if climate.local_t_star raises an Error.

    This function will still write a `local_mustar.json`, filled with NANs,
    if climate.local_t_star fails and cfg.PARAMS['continue_on_error'] = True.

    As `local_mustar.json` will be read, extended and rewritten by
    climate.mu_star_calibration, those entries will be written here as well.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = np.nan
    df['bias'] = np.nan
    df['mu_star_glacierwide'] = np.nan
    df['mu_star_per_flowline'] = [np.nan]
    df['mu_star_flowline_avg'] = np.nan
    df['mu_star_allsame'] = np.nan
    gdir.write_json(df, 'local_mustar')
