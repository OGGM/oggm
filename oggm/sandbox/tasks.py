import logging

# External libs
import numpy as np
import pandas as pd
from scipy import optimize as optimization

# Locals
from oggm import entity_task
import oggm.core.massbalance as mbmods
from oggm.core.flowline import robust_model_run

# Module logger
log = logging.getLogger(__name__)


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

    rmb = mbmods.UncertainMassBalance(mb,
                                      rdn_temp_bias_seed=rdn_temp_bias_seed,
                                      rdn_temp_bias_sigma=sigma_t,
                                      rdn_prcp_bias_seed=rdn_prcp_bias_seed,
                                      rdn_prcp_bias_sigma=sigma_p,
                                      rdn_bias_seed=rdn_bias_seed,
                                      rdn_bias_sigma=sigma_smb)

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=rmb, ys=0, ye=nyears,
                            **kwargs)
