from oggm import utils
from oggm.core.massbalance import RandomMassBalance, MultipleFlowlineMassBalance
from oggm.core.flowline import (robust_model_run, simple_model_run,
                                FluxBasedModel)
from functools import partial
import logging

# Module logger
log = logging.getLogger(__name__)


@utils.entity_task(log)
def default_run(gdir, nyears=300, y0=None, temp_bias=None, seed=None,
                output_filesuffix=None, flux_limiter=False):
    """A simple doc"""
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias
    robust_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears, flux_limiter=flux_limiter)


@utils.entity_task(log)
def better_run(gdir, nyears=300, y0=None, temp_bias=None, seed=None,
               min_dt=0, output_filesuffix=None, cfl_number=0.05,
               flux_limiter=False):
    """A simple doc"""
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias

    supermodel = partial(FluxBasedModel, min_dt=min_dt, raise_on_cfl=True,
                         cfl_number=cfl_number, flux_limiter=flux_limiter)
    supermodel.__name__ = 'Model' + output_filesuffix
    simple_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears,
                     numerical_model_class=supermodel)
