# Builtins
import logging
import copy

# External libs
import numpy as np

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
import oggm.core.massbalance as mbmods
from oggm.core.flowline import FluxBasedModel

# Constants

# Module logger
log = logging.getLogger(__name__)


def _find_inital_glacier(final_model, firstguess_mb, y0, y1,
                         rtol=0.01, atol=10, max_ite=100,
                         init_bias=0., equi_rate=0.0005,
                         ref_area=None):
    """ Iterative search for a plausible starting time glacier"""

    # Objective
    if ref_area is None:
        ref_area = final_model.area_m2
    log.info('iterative_initial_glacier_search '
             'in year %d. Ref area to catch: %.3f km2. '
             'Tolerance: %.2f %%',
             np.int64(y0), ref_area * 1e-6, rtol * 100)

    # are we trying to grow or to shrink the glacier?
    prev_model = copy.deepcopy(final_model)
    prev_fls = copy.deepcopy(prev_model.fls)
    prev_model.reset_y0(y0)
    prev_model.run_until(y1)
    prev_area = prev_model.area_m2

    # Just in case we already hit the correct starting state
    if np.allclose(prev_area, ref_area, atol=atol, rtol=rtol):
        model = copy.deepcopy(final_model)
        model.reset_y0(y0)
        log.info('iterative_initial_glacier_search: inital '
                 'starting glacier converges '
                 'to itself with a final dif of %.2f %%',
                 utils.rel_err(ref_area, prev_area) * 100)
        return 0, None, model

    if prev_area < ref_area:
        sign_mb = -1.
        log.info('iterative_initial_glacier_search, ite: %d. '
                 'Glacier would be too '
                 'small of %.2f %%. Continue', 0,
                 utils.rel_err(ref_area, prev_area) * 100)
    else:
        log.info('iterative_initial_glacier_search, ite: %d. '
                 'Glacier would be too '
                 'big of %.2f %%. Continue', 0,
                 utils.rel_err(ref_area, prev_area) * 100)
        sign_mb = 1.

    # Log prefix
    logtxt = 'iterative_initial_glacier_search'

    # Loop until 100 iterations
    c = 0
    bias_step = 0.1
    mb_bias = init_bias - bias_step
    reduce_step = 0.01

    mb = copy.deepcopy(firstguess_mb)
    mb.temp_bias = sign_mb * mb_bias
    grow_model = FluxBasedModel(copy.deepcopy(final_model.fls), mb_model=mb,
                                fs=final_model.fs,
                                glen_a=final_model.glen_a,
                                min_dt=final_model.min_dt,
                                max_dt=final_model.max_dt)
    while True and (c < max_ite):
        c += 1

        # Grow
        mb_bias += bias_step
        mb.temp_bias = sign_mb * mb_bias
        log.info(logtxt + ', ite: %d. New bias: %.2f', c, sign_mb * mb_bias)
        grow_model.reset_flowlines(copy.deepcopy(prev_fls))
        grow_model.reset_y0(0.)
        grow_model.run_until_equilibrium(rate=equi_rate)
        log.info(logtxt + ', ite: %d. Grew to equilibrium for %d years, '
                          'new area: %.3f km2', c, grow_model.yr,
                 grow_model.area_km2)

        # Shrink
        new_fls = copy.deepcopy(grow_model.fls)
        new_model = copy.deepcopy(final_model)
        new_model.reset_flowlines(copy.deepcopy(new_fls))
        new_model.reset_y0(y0)
        new_model.run_until(y1)
        new_area = new_model.area_m2

        # Maybe we done?
        if np.allclose(new_area, ref_area, atol=atol, rtol=rtol):
            new_model.reset_flowlines(new_fls)
            new_model.reset_y0(y0)
            log.info(logtxt + ', ite: %d. Converged with a '
                     'final dif of %.2f %%', c,
                     utils.rel_err(ref_area, new_area)*100)
            return c, mb_bias, new_model

        # See if we did a step to far or if we have to continue growing
        do_cont_1 = (sign_mb < 0.) and (new_area < ref_area)
        do_cont_2 = (sign_mb > 0.) and (new_area > ref_area)
        if do_cont_1 or do_cont_2:
            # Reset the previous state and continue
            prev_fls = new_fls

            log.info(logtxt + ', ite: %d. Dif of %.2f %%. '
                              'Continue', c,
                     utils.rel_err(ref_area, new_area)*100)
            continue

        # Ok. We went too far. Reduce the bias step but keep previous state
        mb_bias -= bias_step
        bias_step /= reduce_step
        log.info(logtxt + ', ite: %d. Went too far.', c)
        if bias_step < 0.1:
            break

    raise RuntimeError('Did not converge after {} iterations'.format(c))


@entity_task(log, writes=['model_run'])
def iterative_initial_glacier_search(gdir, y0=None, init_bias=0., rtol=0.005,
                                     write_steps=True):
    """Iterative search for the glacier in year y0.

    this is outdated and doesn't really work.
    """

    fs = cfg.PARAMS['fs']
    glen_a = cfg.PARAMS['glen_a']

    if y0 is None:
        y0 = cfg.PARAMS['y0']
    y1 = gdir.rgi_date.year
    mb = mbmods.PastMassBalance(gdir)
    fls = gdir.read_pickle('model_flowlines')

    model = FluxBasedModel(fls, mb_model=mb, y0=0., fs=fs, glen_a=glen_a)
    assert np.isclose(model.area_km2, gdir.rgi_area_km2, rtol=0.05)

    mb = mbmods.BackwardsMassBalanceModel(gdir)
    ref_area = gdir.rgi_area_m2
    ite, bias, past_model = _find_inital_glacier(model, mb, y0, y1,
                                                 rtol=rtol,
                                                 init_bias=init_bias,
                                                 ref_area=ref_area)

    path = gdir.get_filepath('model_run', delete=True)
    if write_steps:
        past_model.run_until_and_store(y1, path=path)
    else:
        past_model.to_netcdf(path)


def test_find_t0(self):

    from oggm.tests.funcs import init_hef
    from oggm.core import flowline
    import pandas as pd
    import matplotlib.pyplot as plt
    do_plot = True

    gdir = init_hef(border=80)

    flowline.init_present_time_glacier(gdir)
    glacier = gdir.read_pickle('model_flowlines')
    df = pd.read_csv(utils.get_demo_file('hef_lengths.csv'), index_col=0)
    df.columns = ['Leclercq']
    df = df.loc[1950:]

    vol_ref = flowline.FlowlineModel(glacier).volume_km3

    init_bias = 94.  # so that "went too far" comes once on travis
    rtol = 0.005

    flowline.iterative_initial_glacier_search(gdir, y0=df.index[0],
                                              init_bias=init_bias,
                                              rtol=rtol, write_steps=True)

    past_model = flowline.FileModel(gdir.get_filepath('model_run'))

    vol_start = past_model.volume_km3
    bef_fls = copy.deepcopy(past_model.fls)

    mylen = past_model.length_m_ts()
    df['oggm'] = mylen[12::12].values
    df = df-df.iloc[-1]

    past_model.run_until(2003)

    vol_end = past_model.volume_km3
    np.testing.assert_allclose(vol_ref, vol_end, rtol=0.05)

    rmsd = utils.rmsd(df.Leclercq, df.oggm)
    self.assertTrue(rmsd < 1000.)

    if do_plot:  # pragma: no cover
        df.plot()
        plt.ylabel('Glacier length (relative to 2003)')
        plt.show()
        plt.figure()
        lab = 'ref (vol={:.2f}km3)'.format(vol_ref)
        plt.plot(glacier[-1].surface_h, 'k', label=lab)
        lab = 'oggm start (vol={:.2f}km3)'.format(vol_start)
        plt.plot(bef_fls[-1].surface_h, 'b', label=lab)
        lab = 'oggm end (vol={:.2f}km3)'.format(vol_end)
        plt.plot(past_model.fls[-1].surface_h, 'r', label=lab)

        plt.plot(glacier[-1].bed_h, 'gray', linewidth=2)
        plt.legend(loc='best')
        plt.show()
