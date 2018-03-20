import logging
import warnings
import copy
from collections import OrderedDict
from time import gmtime, strftime

# External libs
import numpy as np
import pandas as pd
from scipy import optimize as optimization
import shapely.geometry as shpg
import xarray as xr

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
import oggm.core.massbalance as mbmods
from oggm.core.flowline import robust_model_run

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, TWO_THIRDS, SEC_IN_HOUR
from oggm.cfg import RHO, G, N, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)


class UncertainRandomMassBalance(mbmods.MassBalanceModel):
    def __init__(self, basis_model,
                 rdn_temp_bias_seed=None, rdn_temp_bias_sigma=1,
                 rdn_prcp_bias_seed=None, rdn_prcp_bias_sigma=0.2,
                 rdn_bias_seed=None, rdn_bias_sigma=350):
        super(UncertainRandomMassBalance, self).__init__()
        self.mbmod = basis_model
        self.valid_bounds = self.mbmod.valid_bounds
        self.rng_temp = np.random.RandomState(rdn_temp_bias_seed)
        self.rng_prcp = np.random.RandomState(rdn_prcp_bias_seed)
        self.rng_bias = np.random.RandomState(rdn_bias_seed)
        self._temp_sigma = rdn_temp_bias_sigma
        self._prcp_sigma = rdn_prcp_bias_sigma
        self._bias_sigma = rdn_bias_sigma
        self._state_temp = dict()
        self._state_prcp = dict()
        self._state_bias = dict()

    def _get_state_temp(self, year):
        year = int(year)
        if year not in self._state_temp:
            self._state_temp[year] = self.rng_temp.randn() * self._temp_sigma
        return self._state_temp[year]

    def _get_state_prcp(self, year):
        year = int(year)
        if year not in self._state_prcp:
            self._state_prcp[year] = self.rng_prcp.randn() * self._prcp_sigma
        return self._state_prcp[year]

    def _get_state_bias(self, year):
        year = int(year)
        if year not in self._state_bias:
            self._state_bias[year] = self.rng_bias.randn() * self._bias_sigma
        return self._state_bias[year]

    def get_annual_mb(self, heights, year=None):
        _t = self.mbmod.mbmod.temp_bias
        _p = self.mbmod.mbmod.prcp_bias
        _b = self.mbmod.mbmod.bias
        self.mbmod.mbmod.temp_bias = self._get_state_temp(year) + _t
        self.mbmod.mbmod.prcp_bias = self._get_state_prcp(year) + _p
        self.mbmod.mbmod.bias = self._get_state_bias(year) + _b
        out = self.mbmod.get_annual_mb(heights, year=year)
        self.mbmod.mbmod.temp_bias = _t
        self.mbmod.mbmod.prcp_bias = _p
        self.mbmod.mbmod.bias = _b
        return out


@entity_task(log)
def run_uncertain_random_climate(gdir, nyears=700,
                                 output_filesuffix='',
                                 sigma_t=None,
                                 sigma_p=None,
                                 sigma_smb=None,
                                 rdn_temp_bias_seed=None,
                                 rdn_prcp_bias_seed=None,
                                 rdn_bias_seed=None,
                                 orig_mb_tbias=0,
                                 orig_mb_pbias=1,
                                 orig_mb_sbias=0,
                                 **kwargs):
    """Test stuff

    Parameters
    ----------
    gdir
    nyears
    output_filesuffix
    rdn_temp_bias_seed
    rdn_prcp_bias_seed
    rdn_bias_seed
    kwargs

    Returns
    -------
    """

    # Find the optimal bias for a balanced MB
    mbref = mbmods.PastMassBalance(gdir)
    h, w = gdir.get_inversion_flowline_hw()
    cyrs = np.unique(mbref.years)

    def to_minimize(x):
        mbref.bias = x
        return np.mean(mbref.get_specific_mb(h, w, cyrs))

    opti_bias = optimization.brentq(to_minimize, -10000, 10000, xtol=0.01)
    mbref.bias = opti_bias
    assert np.allclose(np.mean(mbref.get_specific_mb(h, w, cyrs)),
                       0, atol=0.01)

    rdf = pd.DataFrame(index=cyrs)
    for y in cyrs:
        rdf.loc[y, 'SMB'] = mbref.get_specific_mb(h, w, year=y)
        t, _, p, _ = mbref.get_annual_climate([np.mean(h)], year=y)
        rdf.loc[y, 'TEMP'] = t
        rdf.loc[y, 'PRCP'] = p

    if sigma_smb is None:
        sigma_smb = rdf.std()['SMB'] / 2
    if sigma_t is None:
        sigma_t = rdf.std()['TEMP'] / 2
    if sigma_p is None:
        sigma_p = (rdf.std() / rdf.mean())['PRCP'] / 2

    mb = mbmods.RandomMassBalance(gdir, all_years=True, seed=0)
    mb.mbmod.temp_bias = orig_mb_tbias
    mb.mbmod.prcp_bias = orig_mb_pbias
    mb.mbmod.bias = orig_mb_sbias + opti_bias

    rmb = UncertainRandomMassBalance(mb,
                                     rdn_temp_bias_seed=rdn_temp_bias_seed,
                                     rdn_temp_bias_sigma=sigma_t,
                                     rdn_prcp_bias_seed=rdn_prcp_bias_seed,
                                     rdn_prcp_bias_sigma=sigma_p,
                                     rdn_bias_seed=rdn_bias_seed,
                                     rdn_bias_sigma=sigma_smb)

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=rmb, ys=0, ye=nyears,
                            **kwargs)
