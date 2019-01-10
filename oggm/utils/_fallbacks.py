"""Fallback functions which can be passed to the entity decorator."""

# External libs
import numpy as np


def fb_local_t_star(gdir):
    """A Fallback function if climate.local_t_star raises an Error.

    This function will still write a `local_mustar.json`, filled with NANs,
    if climate.local_t_star fails and cfg.PARAMS['continue_on_error'] = True.

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
    gdir.write_json(df, 'local_mustar')


def fb_mu_star_calibration(gdir):
    """A Fallback function if climate.mu_star_calibration raises an Error.

    This function will still read, expand and write a `local_mustar.json`,
    filled with NANs, if climate.mu_star_calibration fails
    and if cfg.PARAMS['continue_on_error'] = True.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # read json
    df = gdir.read_json('local_mustar')
    # add these keys which mu_star_calibration would add
    df['mu_star_per_flowline'] = [np.nan]
    df['mu_star_flowline_avg'] = np.nan
    df['mu_star_allsame'] = np.nan
    # write
    gdir.write_json(df, 'local_mustar')
