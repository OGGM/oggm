from oggm import utils, cfg
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

    bias = 0 if y0 is None else None
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     bias=bias, y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias
    robust_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears, flux_limiter=flux_limiter)


@utils.entity_task(log)
def better_run(gdir, nyears=300, y0=None, temp_bias=None, seed=None,
               min_dt=0, max_dt=10*cfg.SEC_IN_DAY, output_filesuffix=None,
               cfl_number=0.05, fixed_dt=None, flux_limiter=False,
               monthly_steps=False):
    """A simple doc"""

    bias = 0 if y0 is None else None
    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     bias=bias, y0=y0, seed=seed)
    if temp_bias is not None:
        mb.temp_bias = temp_bias

    supermodel = partial(FluxBasedModel, min_dt=min_dt, max_dt=max_dt,
                         raise_on_cfl=True, cfl_number=cfl_number,
                         flux_limiter=flux_limiter,
                         fixed_dt=fixed_dt, monthly_steps=monthly_steps)
    supermodel.__name__ = 'Model' + output_filesuffix
    simple_model_run(gdir, output_filesuffix=output_filesuffix,
                     mb_model=mb, ys=0, ye=nyears,
                     numerical_model_class=supermodel)
