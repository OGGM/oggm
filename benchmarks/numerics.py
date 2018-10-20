from scipy import stats
import numpy as np
from oggm import cfg
from oggm.core import massbalance, flowline
from oggm.core.sia2d import Upstream2D
from oggm.tests.funcs import dummy_constant_bed

cfg.initialize()


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2.,
                    kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def time_1d_flux_simple_bed_fixed_dt():

        fls = dummy_constant_bed()
        mb = massbalance.LinearMassBalance(2600.)

        model = flowline.FluxBasedModel(fls, mb_model=mb, y0=0.,
                                        fixed_dt=10 * cfg.SEC_IN_DAY)
        model.run_until(800)


def time_1d_flux_simple_bed_adaptive_dt():

        fls = dummy_constant_bed()
        mb = massbalance.LinearMassBalance(2600.)

        model = flowline.FluxBasedModel(fls, mb_model=mb, y0=0.)
        model.run_until(800)


def time_2d_sia_small():

    bed_2d = gkern() * 1e5
    mb = massbalance.LinearMassBalance(450., grad=3)
    sdmodel = Upstream2D(bed_2d, dx=200, mb_model=mb, y0=0.)
    sdmodel.run_until(2000)


def time_2d_sia_large():

    bed_2d = gkern(kernlen=51) * 5e5
    mb = massbalance.LinearMassBalance(450., grad=3)
    sdmodel = Upstream2D(bed_2d, dx=200, mb_model=mb, y0=0.)
    sdmodel.run_until(2000)
