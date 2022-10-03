"""Dynamic spinup functions for model initialisation"""
# Builtins
import logging
import copy
import os

# External libs
import numpy as np
from scipy import interpolate

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance)
from oggm.core.climate import apparent_mb_from_any_mb
from oggm.core.flowline import (FluxBasedModel, FileModel,
                                init_present_time_glacier,
                                run_from_climate_data)

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def run_dynamic_spinup(gdir, init_model_filesuffix=None, init_model_yr=None,
                       init_model_fls=None,
                       climate_input_filesuffix='',
                       evolution_model=FluxBasedModel,
                       mb_model_historical=None, mb_model_spinup=None,
                       spinup_period=20, spinup_start_yr=None,
                       min_spinup_period=10, spinup_start_yr_max=None,
                       yr_rgi=None, minimise_for='area', precision_percent=1,
                       precision_absolute=1, min_ice_thickness=None,
                       first_guess_t_bias=-2, t_bias_max_step_length=2,
                       maxiter=30, output_filesuffix='_dynamic_spinup',
                       store_model_geometry=True, store_fl_diagnostics=None,
                       store_model_evolution=True, ignore_errors=False,
                       return_t_bias_best=False, ye=None,
                       model_flowline_filesuffix='', make_compatible=False,
                       **kwargs):
    """Dynamically spinup the glacier to match area or volume at the RGI date.

    This task allows to do simulations in the recent past (before the glacier
    inventory date), when the state of the glacier was unknown. This is a
    very difficult task the longer further back in time one goes
    (see publications by Eis et al. for a theoretical background), but can
    work quite well for short periods. Note that the solution is not unique.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    init_model_filesuffix : str or None
        if you want to start from a previous model run state. This state
        should be at time yr_rgi_date.
    init_model_yr : int or None
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    climate_input_filesuffix : str
        filesuffix for the input climate file
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: FluxBasedModel
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a PastMassBalance model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "yr_rgi - spinup_period". Minimum allowed value is 10. If
        the provided climate data starts at year later than
        (yr_rgi - spinup_period) the spinup_period is set to
        (yr_rgi - yr_climate_start). Caution if spinup_start_yr is set the
        spinup_period is ignored.
        Default is 20
    spinup_start_yr : int or None
        The start year of the dynamic spinup. If the provided year is before
        the provided climate data starts the year of the climate data is used.
        If set it overrides the spinup_period.
        Default is None
    min_spinup_period : int
        If the dynamic spinup function fails with the initial 'spinup_period'
        a shorter period is tried. Here you can define the minimum period to
        try.
        Default is 10
    spinup_start_yr_max : int or None
        Possibility to provide a maximum year where the dynamic spinup must
        start from at least. If set, this overrides the min_spinup_period if
        yr_rgi - spinup_start_yr_max > min_spinup_period.
        Default is None
    yr_rgi : int
        The rgi date, at which we want to match area or volume.
        If None, gdir.rgi_date + 1 is used (the default).
    minimise_for : str
        The variable we want to match at yr_rgi. Options are 'area' or 'volume'.
        Default is 'area'
    precision_percent : float
        Gives the precision we want to match in percent. The algorithm makes
        sure that the resulting relative mismatch is smaller than
        precision_percent, but also that the absolute value is smaller than
        precision_absolute.
        Default is 1., meaning the difference must be within 1% of the given
        value (area or volume).
    precision_absolute : float
        Gives an minimum absolute value to match. The algorithm makes sure that
        the resulting relative mismatch is smaller than precision_percent, but
        also that the absolute value is smaller than precision_absolute.
        The unit of precision_absolute depends on minimise_for (if 'area' in
        km2, if 'volume' in km3)
        Default is 1.
    min_ice_thickness : float
        Gives an minimum ice thickness for model grid points which are counted
        to the total model value. This could be useful to filter out seasonal
        'glacier growth', as OGGM do not differentiate between snow and ice in
        the forward model run. Therefore you could see quite fast changes
        (spikes) in the time-evolution (especially visible in length and area).
        If you set this value to 0 the filtering can be switched off.
        Default is 10.
    first_guess_t_bias : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_bias_max_step_length : float
        Defines the maximums allowed change of t_bias between two iterations. Is
        needed to avoid to large changes.
        Default is 2.
    maxiter : int
        Maximum number of minimisation iterations per spinup period. If reached
        and 'ignore_errors=False' an error is raised.
        Default is 30
    output_filesuffix : str
        for the output file
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        Default is True
    store_fl_diagnostics : bool or None
        whether to store the model flowline diagnostics to disk or not.
        Default is None
    store_model_evolution : bool
        if True the complete dynamic spinup run is saved (complete evolution
        of the model during the dynamic spinup), if False only the final model
        state after the dynamic spinup run is saved. (Hint: if
        store_model_evolution = True and ignore_errors = True and an Error
        during the dynamic spinup occurs the stored model evolution is only one
        year long)
        Default is True
    ignore_errors : bool
        If True the function saves the model without a dynamic spinup using
        the 'output_filesuffix', if an error during the dynamic spinup occurs.
        This is useful if you want to keep glaciers for the following tasks.
        Default is True
    return_t_bias_best : bool
        If True the used temperature bias for the spinup is returned in
        addition to the final model. If an error occurs and ignore_error=True,
        the returned value is np.nan.
        Default is False
    ye : int
        end year of the model run, must be larger than yr_rgi. If nothing is
        given it is set to yr_rgi. It is not recommended to use it if only data
        until yr_rgi is needed for calibration as this increases the run time
        of each iteration during the iterative minimisation. Instead use
        run_from_climate_data afterwards and merge both outputs using
        merge_consecutive_run_outputs.
        Default is None
    model_flowline_filesuffix : str
        suffix to the model_flowlines filename to use (if no other flowlines
        are provided with init_model_filesuffix or init_model_fls).
        Default is ''
    make_compatible : bool
        if set to true this will add all variables to the resulting dataset
        so it could be combined with any other one. This is necessary if
        different spinup methods are used. For example if using the dynamic
        spinup and setting fixed geometry spinup as fallback, the variable
        'is_fixed_geometry_spinup' must be added to the dynamic spinup so
        it is possible to compile both glaciers together.
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final dynamically spined-up model. Type depends on the selected
        evolution_model, by default a FluxBasedModel.
    """

    if yr_rgi is None:
        yr_rgi = gdir.rgi_date + 1  # + 1 converted to hydro years

    if ye is None:
        ye = yr_rgi

    if ye < yr_rgi:
        raise RuntimeError(f'The provided end year (ye = {ye}) must be larger'
                           f'than the rgi date (yr_rgi = {yr_rgi}!')

    yr_min = gdir.get_climate_info()['baseline_hydro_yr_0']

    if min_ice_thickness is None:
        min_ice_thickness = cfg.PARAMS['dynamic_spinup_min_ice_thick']

    # check provided maximum start year here, and change min_spinup_period
    if spinup_start_yr_max is not None:
        if spinup_start_yr_max < yr_min:
            raise RuntimeError(f'The provided maximum start year (= '
                               f'{spinup_start_yr_max}) must be larger than '
                               f'the start year of the provided climate data '
                               f'(= {yr_min})!')
        if spinup_start_yr is not None:
            if spinup_start_yr_max <= spinup_start_yr:
                raise RuntimeError(f'The provided start year (= '
                                   f'{spinup_start_yr} must be smaller than '
                                   f'the maximum start year '
                                   f'{spinup_start_yr_max}!')
        if (yr_rgi - spinup_start_yr_max) > min_spinup_period:
            min_spinup_period = (yr_rgi - spinup_start_yr_max)

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls

    if init_model_fls is None:
        fls_spinup = gdir.read_pickle('model_flowlines',
                                      filesuffix=model_flowline_filesuffix)
    else:
        fls_spinup = copy.deepcopy(init_model_fls)

    # MassBalance for actual run from yr_spinup to yr_rgi
    if mb_model_historical is None:
        mb_model_historical = MultipleFlowlineMassBalance(
            gdir, mb_model_class=PastMassBalance,
            filename='climate_historical',
            input_filesuffix=climate_input_filesuffix)

    # here we define the file-paths for the output
    if store_model_geometry:
        geom_path = gdir.get_filepath('model_geometry',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    else:
        geom_path = False

    if store_fl_diagnostics is None:
        store_fl_diagnostics = cfg.PARAMS['store_fl_diagnostics']

    if store_fl_diagnostics:
        fl_diag_path = gdir.get_filepath('fl_diagnostics',
                                         filesuffix=output_filesuffix,
                                         delete=True)
    else:
        fl_diag_path = False

    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)

    if cfg.PARAMS['use_inversion_params_for_run']:
        diag = gdir.get_diagnostics()
        fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
        glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
    else:
        fs = cfg.PARAMS['fs']
        glen_a = cfg.PARAMS['glen_a']

    kwargs.setdefault('fs', fs)
    kwargs.setdefault('glen_a', glen_a)

    mb_elev_feedback = kwargs.get('mb_elev_feedback', 'annual')
    if mb_elev_feedback != 'annual':
        raise InvalidParamsError('Only use annual mb_elev_feedback with the '
                                 'dynamic spinup function!')

    if cfg.PARAMS['use_kcalving_for_run']:
        raise InvalidParamsError('Dynamic spinup not tested with '
                                 "cfg.PARAMS['use_kcalving_for_run'] is `True`!")

    # this function saves a model without conducting a dynamic spinup, but with
    # the provided output_filesuffix, so following tasks can find it.
    # This is necessary if yr_rgi < yr_min + 10 or if the dynamic spinup failed.
    def save_model_without_dynamic_spinup():
        gdir.add_to_diagnostics('run_dynamic_spinup_success', False)
        yr_use = np.clip(yr_rgi, yr_min, None)
        model_dynamic_spinup_end = evolution_model(fls_spinup,
                                                   mb_model_historical,
                                                   y0=yr_use,
                                                   **kwargs)

        with np.warnings.catch_warnings():
            if ye < yr_use:
                yr_run = yr_use
            else:
                yr_run = ye
            # For operational runs we ignore the warnings
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_dynamic_spinup_end.run_until_and_store(
                yr_run,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path,
                make_compatible=make_compatible)

        return model_dynamic_spinup_end

    if yr_rgi < yr_min + min_spinup_period:
        log.warning('The provided rgi_date is smaller than yr_climate_start + '
                    'min_spinup_period, therefore no dynamic spinup is '
                    'conducted and the original flowlines are saved at the '
                    'provided rgi_date or the start year of the provided '
                    'climate data (if yr_climate_start > yr_rgi)')
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_bias_best:
                return model_dynamic_spinup_end, np.nan
            else:
                return model_dynamic_spinup_end
        else:
            raise RuntimeError('The difference between the rgi_date and the '
                               'start year of the climate data is to small to '
                               'run a dynamic spinup!')

    # here we define the flowline we want to match, it is assumed that during
    # the inversion the volume was calibrated towards the consensus estimate
    # (as it is by default), but this means the volume is matched on a
    # regional scale, maybe we could use here the individual glacier volume
    fls_ref = copy.deepcopy(fls_spinup)
    if minimise_for == 'area':
        unit = 'km2'
        other_variable = 'volume'
        other_unit = 'km3'
    elif minimise_for == 'volume':
        unit = 'km3'
        other_variable = 'area'
        other_unit = 'km2'
    else:
        raise NotImplementedError
    cost_var = f'{minimise_for}_{unit}'
    reference_value = np.sum([getattr(f, cost_var) for f in fls_ref])
    other_reference_value = np.sum([getattr(f, f'{other_variable}_{other_unit}')
                                    for f in fls_ref])

    # if reference value is zero no dynamic spinup is possible
    if reference_value == 0.:
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_bias_best:
                return model_dynamic_spinup_end, np.nan
            else:
                return model_dynamic_spinup_end
        else:
            raise RuntimeError('The given reference value is Zero, no dynamic '
                               'spinup possible!')

    # here we adjust the used precision_percent to make sure the resulting
    # absolute mismatch is smaller than precision_absolute
    precision_percent = min(precision_percent,
                            precision_absolute / reference_value * 100)

    # only used to check performance of function
    forward_model_runs = [0]

    # the actual spinup run
    def run_model_with_spinup_to_rgi_date(t_bias):
        forward_model_runs.append(forward_model_runs[-1] + 1)

        # with t_bias the glacier state after spinup is changed between iterations
        mb_model_spinup.temp_bias = t_bias
        # run the spinup
        model_spinup = evolution_model(copy.deepcopy(fls_spinup),
                                       mb_model_spinup,
                                       y0=0,
                                       **kwargs)
        model_spinup.run_until(2 * halfsize_spinup)

        # if glacier is completely gone return information in ice-free
        ice_free = False
        if np.isclose(model_spinup.volume_km3, 0.):
            ice_free = True

        # Now conduct the actual model run to the rgi date
        model_historical = evolution_model(model_spinup.fls,
                                           mb_model_historical,
                                           y0=yr_spinup,
                                           **kwargs)
        if store_model_evolution:
            ds = model_historical.run_until_and_store(
                ye,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path,
                dynamic_spinup_min_ice_thick=min_ice_thickness,
                make_compatible=make_compatible)
            if type(ds) == tuple:
                ds = ds[0]
            model_area_km2 = ds.area_m2_min_h.loc[yr_rgi].values * 1e-6
            model_volume_km3 = ds.volume_m3_min_h.loc[yr_rgi].values * 1e-9
        else:
            # only run to rgi date and extract values
            model_historical.run_until(yr_rgi)
            fls = model_historical.fls
            model_area_km2 = np.sum(
                [np.sum(fl.bin_area_m2[fl.thick > min_ice_thickness])
                 for fl in fls]) * 1e-6
            model_volume_km3 = np.sum(
                [np.sum((fl.section * fl.dx_meter)[fl.thick > min_ice_thickness])
                 for fl in fls]) * 1e-9
            # afterwards finish the complete run
            model_historical.run_until(ye)

        if cost_var == 'area_km2':
            return model_area_km2, model_volume_km3, model_historical, ice_free
        elif cost_var == 'volume_km3':
            return model_volume_km3, model_area_km2, model_historical, ice_free
        else:
            raise NotImplementedError(f'{cost_var}')

    def cost_fct(t_bias, model_dynamic_spinup_end_loc, other_variable_mismatch_loc):
        # actual model run
        model_value, other_value, model_dynamic_spinup, ice_free = \
            run_model_with_spinup_to_rgi_date(t_bias)

        # save the final model for later
        model_dynamic_spinup_end_loc.append(copy.deepcopy(model_dynamic_spinup))

        # calculate the mismatch in percent
        cost = (model_value - reference_value) / reference_value * 100
        other_variable_mismatch_loc.append(
            (other_value - other_reference_value) / other_reference_value * 100)

        return cost, ice_free

    def init_cost_fct():
        model_dynamic_spinup_end_loc = []
        other_variable_mismatch_loc = []

        def c_fct(t_bias):
            return cost_fct(t_bias, model_dynamic_spinup_end_loc,
                            other_variable_mismatch_loc)

        return c_fct, model_dynamic_spinup_end_loc, other_variable_mismatch_loc

    def minimise_with_spline_fit(fct_to_minimise):
        # defines limits of t_bias in accordance to maximal allowed change
        # between iterations
        t_bias_limits = [first_guess_t_bias - t_bias_max_step_length,
                         first_guess_t_bias + t_bias_max_step_length]
        t_bias_guess = []
        mismatch = []
        # this two variables indicate that the limits were already adapted to
        # avoid an ice_free or out_of_domain error
        was_ice_free = False
        was_out_of_domain = False
        was_errors = [was_out_of_domain, was_ice_free]

        def get_mismatch(t_bias):
            t_bias = copy.deepcopy(t_bias)
            # first check if the new t_bias is in limits
            if t_bias < t_bias_limits[0]:
                # was the smaller limit already executed, if not first do this
                if t_bias_limits[0] not in t_bias_guess:
                    t_bias = copy.deepcopy(t_bias_limits[0])
                else:
                    # smaller limit was already used, check if it was
                    # already newly defined with glacier exceeding domain
                    if was_errors[0]:
                        raise RuntimeError('Not able to minimise without '
                                           'exceeding the domain! Best '
                                           f'mismatch '
                                           f'{np.min(np.abs(mismatch))}%')
                    else:
                        # ok we set a new lower limit
                        t_bias_limits[0] = t_bias_limits[0] - \
                            t_bias_max_step_length
            elif t_bias > t_bias_limits[1]:
                # was the larger limit already executed, if not first do this
                if t_bias_limits[1] not in t_bias_guess:
                    t_bias = copy.deepcopy(t_bias_limits[1])
                else:
                    # larger limit was already used, check if it was
                    # already newly defined with ice free glacier
                    if was_errors[1]:
                        raise RuntimeError('Not able to minimise without ice '
                                           'free glacier after spinup! Best '
                                           'mismatch '
                                           f'{np.min(np.abs(mismatch))}%')
                    else:
                        # ok we set a new upper limit
                        t_bias_limits[1] = t_bias_limits[1] + \
                            t_bias_max_step_length

            # now clip t_bias with limits
            t_bias = np.clip(t_bias, t_bias_limits[0], t_bias_limits[1])

            # now start with mismatch calculation

            # if error during spinup (ice_free or out_of_domain) this defines
            # how much t_bias is changed to look for an error free glacier spinup
            t_bias_search_change = t_bias_max_step_length / 10
            # maximum number of changes to look for an error free glacier
            max_iterations = int(t_bias_max_step_length / t_bias_search_change)
            is_out_of_domain = True
            is_ice_free_spinup = True
            is_ice_free_end = True
            is_first_guess_ice_free = False
            is_first_guess_out_of_domain = False
            doing_first_guess = (len(mismatch) == 0)
            define_new_lower_limit = False
            define_new_upper_limit = False
            iteration = 0

            while ((is_out_of_domain | is_ice_free_spinup | is_ice_free_end) &
                   (iteration < max_iterations)):
                try:
                    tmp_mismatch, is_ice_free_spinup = fct_to_minimise(t_bias)

                    # no error occurred, so we are not outside the domain
                    is_out_of_domain = False

                    # check if we are ice_free after spinup, if so we search
                    # for a new upper limit for t_bias
                    if is_ice_free_spinup:
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_bias, first_guess_t_bias) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_bias = copy.deepcopy(t_bias_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_bias = t_bias - t_bias_max_step_length
                        else:
                            t_bias = np.round(t_bias - t_bias_search_change,
                                              decimals=1)
                        if np.isclose(t_bias, t_bias_guess).any():
                            iteration = copy.deepcopy(max_iterations)

                    # check if we are ice_free at the end of the model run, if
                    # so we use the lower t_bias limit and change the limit if
                    # needed
                    elif np.isclose(tmp_mismatch, -100.):
                        is_ice_free_end = True
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_bias, first_guess_t_bias) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_bias = copy.deepcopy(t_bias_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_bias = t_bias - t_bias_max_step_length
                        else:
                            # if lower limit was already used change it and use
                            if t_bias == t_bias_limits[0]:
                                t_bias_limits[0] = t_bias_limits[0] - \
                                    t_bias_max_step_length
                                t_bias = copy.deepcopy(t_bias_limits[0])
                            else:
                                # otherwise just try with a colder t_bias
                                t_bias = np.round(t_bias - t_bias_search_change,
                                                  decimals=1)

                    else:
                        is_ice_free_end = False

                except RuntimeError as e:
                    # check if glacier grow to large
                    if 'Glacier exceeds domain boundaries, at year:' in f'{e}':
                        # ok we where outside the domain, therefore we search
                        # for a new lower limit for t_bias in 0.1 °C steps
                        is_out_of_domain = True
                        define_new_lower_limit = True
                        was_errors[0] = True
                        # special treatment if it is the first guess
                        if np.isclose(t_bias, first_guess_t_bias) & \
                                doing_first_guess:
                            is_first_guess_out_of_domain = True
                            # here directly jump to the larger limit
                            t_bias = t_bias_limits[1]
                        elif is_first_guess_out_of_domain & doing_first_guess:
                            # make large steps if it is first guess
                            t_bias = t_bias + t_bias_max_step_length
                        else:
                            t_bias = np.round(t_bias + t_bias_search_change,
                                              decimals=1)
                        if np.isclose(t_bias, t_bias_guess).any():
                            iteration = copy.deepcopy(max_iterations)

                    else:
                        # otherwise this error can not be handled here
                        raise RuntimeError(e)

                iteration += 1

            if iteration >= max_iterations:
                # ok we were not able to find an mismatch without error
                # (ice_free or out of domain), so we try to raise an descriptive
                # RuntimeError
                if len(mismatch) == 0:
                    # unfortunately we were not able conduct one single error
                    # free run
                    msg = 'Not able to conduct one error free run. Error is '
                    if is_first_guess_ice_free:
                        msg += f'"ice_free" with last t_bias of {t_bias}.'
                    elif is_first_guess_out_of_domain:
                        msg += f'"out_of_domain" with last t_bias of {t_bias}.'
                    else:
                        raise RuntimeError('Something unexpected happened!')

                    raise RuntimeError(msg)

                elif define_new_lower_limit:
                    raise RuntimeError('Not able to minimise without '
                                       'exceeding the domain! Best '
                                       f'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif define_new_upper_limit:
                    raise RuntimeError('Not able to minimise without ice '
                                       'free glacier after spinup! Best mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif is_ice_free_end:
                    raise RuntimeError('Not able to find a t_bias so that '
                                       'glacier is not ice free at the end! '
                                       '(Last t_bias '
                                       f'{t_bias + t_bias_max_step_length} °C)')
                else:
                    raise RuntimeError('Something unexpected happened during '
                                       'definition of new t_bias limits!')
            else:
                # if we found a new limit set it
                if define_new_upper_limit & define_new_lower_limit:
                    # we can end here if we are at the first guess and took
                    # a to large step
                    was_errors[0] = False
                    was_errors[1] = False
                    if t_bias <= t_bias_limits[0]:
                        t_bias_limits[0] = t_bias
                        t_bias_limits[1] = t_bias_limits[0] + \
                            t_bias_max_step_length
                    elif t_bias >= t_bias_limits[1]:
                        t_bias_limits[1] = t_bias
                        t_bias_limits[0] = t_bias_limits[1] - \
                            t_bias_max_step_length
                    else:
                        if is_first_guess_ice_free:
                            t_bias_limits[1] = t_bias
                        elif is_out_of_domain:
                            t_bias_limits[0] = t_bias
                        else:
                            raise RuntimeError('I have not expected to get here!')
                elif define_new_lower_limit:
                    t_bias_limits[0] = copy.deepcopy(t_bias)
                    if t_bias >= t_bias_limits[1]:
                        # this happens when the first guess was out of domain
                        was_errors[0] = False
                        t_bias_limits[1] = t_bias_limits[0] + \
                            t_bias_max_step_length
                elif define_new_upper_limit:
                    t_bias_limits[1] = copy.deepcopy(t_bias)
                    if t_bias <= t_bias_limits[0]:
                        # this happens when the first guess was ice free
                        was_errors[1] = False
                        t_bias_limits[0] = t_bias_limits[1] - \
                            t_bias_max_step_length

            return float(tmp_mismatch), float(t_bias)

        # first guess
        new_mismatch, new_t_bias = get_mismatch(first_guess_t_bias)
        t_bias_guess.append(new_t_bias)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_bias_guess, mismatch

        # second (arbitrary) guess is given depending on the outcome of first
        # guess, when mismatch is 100% t_bias is changed for
        # t_bias_max_step_length, but at least the second guess is 0.2 °C away
        # from the first guess
        step = np.sign(mismatch[-1]) * max(np.abs(mismatch[-1]) *
                                           t_bias_max_step_length / 100,
                                           0.2)
        new_mismatch, new_t_bias = get_mismatch(t_bias_guess[0] + step)
        t_bias_guess.append(new_t_bias)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_bias_guess, mismatch

        # Now start with splin fit for guessing
        while len(t_bias_guess) < maxiter:
            # get next guess from splin (fit partial linear function to previously
            # calculated (mismatch, t_bias) pairs and get t_bias value where
            # mismatch=0 from this fitted curve)
            sort_index = np.argsort(np.array(mismatch))
            tck = interpolate.splrep(np.array(mismatch)[sort_index],
                                     np.array(t_bias_guess)[sort_index],
                                     k=1)
            # here we catch interpolation errors (two different t_bias with
            # same mismatch), could happen if one t_bias was close to a newly
            # defined limit
            if np.isnan(tck[1]).any():
                if was_errors[0]:
                    raise RuntimeError('Not able to minimise without '
                                       'exceeding the domain! Best '
                                       f'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif was_errors[1]:
                    raise RuntimeError('Not able to minimise without ice '
                                       'free glacier! Best mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                else:
                    raise RuntimeError('Not able to minimise! Problem is '
                                       'unknown, need to check by hand! Best '
                                       'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
            new_mismatch, new_t_bias = get_mismatch(float(interpolate.splev(0,
                                                                            tck)
                                                          ))
            t_bias_guess.append(new_t_bias)
            mismatch.append(new_mismatch)

            if abs(mismatch[-1]) < precision_percent:
                return t_bias_guess, mismatch

        # Ok when we end here the spinup could not find satisfying match after
        # maxiter(ations)
        raise RuntimeError(f'Could not find mismatch smaller '
                           f'{precision_percent}% (only '
                           f'{np.min(np.abs(mismatch))}%) in {maxiter}'
                           f'Iterations!')

    # define function for the actual minimisation
    c_fun, model_dynamic_spinup_end, other_variable_mismatch = init_cost_fct()

    # define the MassBalanceModels for different spinup periods and try to
    # minimise, if minimisation fails a shorter spinup period is used
    # (first a spinup period between initial period and 'min_spinup_period'
    # years and the second try is to use a period of 'min_spinup_period' years,
    # if it still fails the actual error is raised)
    if spinup_start_yr is not None:
        spinup_period_initial = min(yr_rgi - spinup_start_yr, yr_rgi - yr_min)
    else:
        spinup_period_initial = min(spinup_period, yr_rgi - yr_min)
    if spinup_period_initial <= min_spinup_period:
        spinup_periods_to_try = [min_spinup_period]
    else:
        # try out a maximum of three different spinup_periods
        spinup_periods_to_try = [spinup_period_initial,
                                 int((spinup_period_initial + min_spinup_period) / 2),
                                 min_spinup_period]

    # check if the user provided an mb_model_spinup, otherwise we must define a
    # new one each iteration
    provided_mb_model_spinup = False
    if mb_model_spinup is not None:
        provided_mb_model_spinup = True

    for spinup_period in spinup_periods_to_try:
        yr_spinup = yr_rgi - spinup_period

        if not provided_mb_model_spinup:
            # define spinup MassBalance
            # spinup is running for 'yr_rgi - yr_spinup' years, using a
            # ConstantMassBalance
            y0_spinup = (yr_spinup + yr_rgi) / 2
            halfsize_spinup = yr_rgi - y0_spinup
            mb_model_spinup = MultipleFlowlineMassBalance(
                gdir, mb_model_class=ConstantMassBalance,
                filename='climate_historical',
                input_filesuffix=climate_input_filesuffix, y0=y0_spinup,
                halfsize=halfsize_spinup)

        # try to conduct minimisation, if an error occurred try shorter spinup
        # period
        try:
            final_t_bias_guess, final_mismatch = minimise_with_spline_fit(c_fun)
            # ok no error occurred so we succeeded
            break
        except RuntimeError as e:
            # if the last spinup period was min_spinup_period the dynamic
            # spinup failed
            if spinup_period == min_spinup_period:
                log.warning('No dynamic spinup could be conducted and the '
                            'original model with no spinup is saved using the '
                            f'provided output_filesuffix "{output_filesuffix}". '
                            f'The error message of the dynamic spinup is: {e}')
                if ignore_errors:
                    model_dynamic_spinup_end = save_model_without_dynamic_spinup()
                    if return_t_bias_best:
                        return model_dynamic_spinup_end, np.nan
                    else:
                        return model_dynamic_spinup_end
                else:
                    # delete all files which could be saved during the previous
                    # iterations
                    if geom_path and os.path.exists(geom_path):
                        os.remove(geom_path)

                    if fl_diag_path and os.path.exists(fl_diag_path):
                        os.remove(fl_diag_path)

                    if diag_path and os.path.exists(diag_path):
                        os.remove(diag_path)

                    raise RuntimeError(e)

    # hurray, dynamic spinup successfully
    gdir.add_to_diagnostics('run_dynamic_spinup_success', True)

    # also save some other stuff
    gdir.add_to_diagnostics('temp_bias_dynamic_spinup',
                            float(final_t_bias_guess[-1]))
    gdir.add_to_diagnostics('dynamic_spinup_period',
                            int(spinup_period))
    gdir.add_to_diagnostics('dynamic_spinup_forward_model_iterations',
                            int(forward_model_runs[-1]))
    gdir.add_to_diagnostics(f'{minimise_for}_mismatch_dynamic_spinup_{unit}_'
                            f'percent',
                            float(final_mismatch[-1]))
    gdir.add_to_diagnostics(f'reference_{minimise_for}_dynamic_spinup_{unit}',
                            float(reference_value))
    gdir.add_to_diagnostics('dynamic_spinup_other_variable_reference',
                            float(other_reference_value))
    gdir.add_to_diagnostics('dynamic_spinup_mismatch_other_variable_percent',
                            float(other_variable_mismatch[-1]))

    # here only save the final model state if store_model_evolution = False
    if not store_model_evolution:
        with np.warnings.catch_warnings():
            # For operational runs we ignore the warnings
            np.warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_dynamic_spinup_end[-1].run_until_and_store(
                yr_rgi,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path, )

    if return_t_bias_best:
        return model_dynamic_spinup_end[-1], final_t_bias_guess[-1]
    else:
        return model_dynamic_spinup_end[-1]


def define_new_mu_star_in_gdir(gdir, new_mu_star, bias=0):
    """
    Helper function to define a new mu star in an gdir. Is used inside the run
    functions of the dynamic mu star calibration.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to change the mu star
    new_mu_star : float
        the new mu_star to save in the gdir
    bias : float
        an additional bias to add to the mass balance at the end (was
        originally used for the static mu star calibration)

    Returns
    -------

    """
    df = gdir.read_json('local_mustar')
    df['bias'] = bias
    df['mu_star_per_flowline'] = [new_mu_star] * len(df['mu_star_per_flowline'])
    df['mu_star_glacierwide'] = new_mu_star
    df['mu_star_flowline_avg'] = new_mu_star
    df['mu_star_allsame'] = True
    gdir.write_json(df, 'local_mustar')


def dynamic_mu_star_run_with_dynamic_spinup(
        gdir, mu_star, yr0_ref_mb, yr1_ref_mb, fls_init, ys, ye,
        output_filesuffix='', evolution_model=FluxBasedModel,
        mb_model_historical=None, mb_model_spinup=None,
        minimise_for='area', climate_input_filesuffix='', spinup_period=20,
        min_spinup_period=10, yr_rgi=None, precision_percent=1,
        precision_absolute=1, min_ice_thickness=None,
        first_guess_t_bias=-2, t_bias_max_step_length=2, maxiter=30,
        store_model_geometry=True, store_fl_diagnostics=None,
        local_variables=None, set_local_variables=False, do_inversion=True,
        **kwargs):
    """
    This function is one option for a 'run_function' for the
    'run_dynamic_mu_star_calibration' function (the corresponding
    'fallback_function' is
    'dynamic_mu_star_run_with_dynamic_spinup_fallback'). This
    function defines a new mu_star in the glacier directory and conducts an
    inversion calibrating A to match '_vol_m3_ref' with this new mu_star
    ('calibrate_inversion_from_consensus'). Afterwards a dynamic spinup is
    conducted to match 'minimise_for' (for more info look at docstring of
    'run_dynamic_spinup'). And in the end the geodetic mass balance of the
    current run is calculated (between the period [yr0_ref_mb, yr1_ref_mb]) and
    returned.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mu_star : float
        the mu_star used for this run
    yr0_ref_mb : int
        the start year of the geodetic mass balance
    yr1_ref_mb : int
        the end year of the geodetic mass balance
    fls_init : []
        List of flowlines to use to initialise the model
    ys : int
        start year of the complete run, must by smaller or equal y0_ref_mb
    ye : int
        end year of the complete run, must be smaller or equal y1_ref_mb
    output_filesuffix : str
        For the output file.
        Default is ''
    evolution_model : class:oggm.core.FlowlineModel
        Evolution model to use.
        Default is FluxBasedModel
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a PastMassBalance model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    minimise_for : str
        The variable we want to match at yr_rgi. Options are 'area' or 'volume'.
        Default is 'area'.
    climate_input_filesuffix : str
        filesuffix for the input climate file
        Default is ''
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "yr_rgi - spinup_period". Minimum allowed value is defined
        with 'min_spinup_period'. If the provided climate data starts at year
        later than (yr_rgi - spinup_period) the spinup_period is set to
        (yr_rgi - yr_climate_start). Caution if spinup_start_yr is set the
        spinup_period is ignored.
        Default is 20
    min_spinup_period : int
        If the dynamic spinup function fails with the initial 'spinup_period'
        a shorter period is tried. Here you can define the minimum period to
        try.
        Default is 10
    yr_rgi : int or None
        The rgi date, at which we want to match area or volume.
        If None, gdir.rgi_date + 1 is used (the default).
        Default is None
    precision_percent : float
        Gives the precision we want to match for the selected variable
        ('minimise_for') at rgi_date in percent. The algorithm makes sure that
        the resulting relative mismatch is smaller than precision_percent, but
        also that the absolute value is smaller than precision_absolute.
        Default is 1, meaning the difference must be within 1% of the given
        value (area or volume).
    precision_absolute : float
        Gives an minimum absolute value to match. The algorithm makes sure that
        the resulting relative mismatch is smaller than
        precision_percent, but also that the absolute value is
        smaller than precision_absolute.
        The unit of precision_absolute depends on minimise_for (if 'area' in
        km2, if 'volume' in km3)
        Default is 1.
    min_ice_thickness : float
        Gives an minimum ice thickness for model grid points which are counted
        to the total model value. This could be useful to filter out seasonal
        'glacier growth', as OGGM do not differentiate between snow and ice in
        the forward model run. Therefore you could see quite fast changes
        (spikes) in the time-evolution (especially visible in length and area).
        If you set this value to 0 the filtering can be switched off.
        Default is 10.
    first_guess_t_bias : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_bias_max_step_length : float
        Defines the maximums allowed change of t_bias between two iterations. Is
        needed to avoid to large changes.
        Default is 2
    maxiter : int
        Maximum number of minimisation iterations per dynamic spinup where area
        or volume is tried to be matched. If reached and 'ignore_errors=False'
        an error is raised.
        Default is 30
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        Default is True
    store_fl_diagnostics : bool or None
        Whether to store the model flowline diagnostics to disk or not.
        Default is None (-> cfg.PARAMS['store_fl_diagnostics'])
    local_variables : dict
        User MUST provide a dictionary here. This dictionary is used to save
        the last temperature bias from the previous dynamic spinup run (for
        optimisation) and the initial glacier volume. User must take care that
        this variables are not changed outside this function!
        Default is None (-> raise an error if no dict is provided)
    set_local_variables : bool
        If True this resets the local_variables to their initial state. It
        sets the first_guess_t_bias with the key 't_bias' AND sets the current
        glacier volume with the key 'vol_m3_ref' which is later used in the
        calibration during inversion.
        Default is False
    do_inversion : bool
        If True a complete inversion is conducted using the provided mu_star
        before the actual calibration run.
        Default is False
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`, float
        The final model after the run and the calculated geodetic mass balance
    """
    if not isinstance(local_variables, dict):
        raise ValueError('You must provide a dict for local_variables!')

    from oggm.workflow import calibrate_inversion_from_consensus

    if set_local_variables:
        # clear the provided dictionary and set the first elements
        local_variables.clear()
        local_variables['t_bias'] = [first_guess_t_bias]
        # ATTENTION: it is assumed that the flowlines in gdir have the volume
        # we want to match during calibrate_inversion_from_consensus when we
        # set_local_variables
        fls_ref = gdir.read_pickle('model_flowlines')
        local_variables['vol_m3_ref'] = np.sum([f.volume_m3 for f in fls_ref])

        # we are done with preparing the local_variables for the upcoming iterations
        return None

    if yr_rgi is None:
        yr_rgi = gdir.rgi_date + 1  # + 1 converted to hydro years
    if min_spinup_period > yr_rgi - ys:
        log.workflow('The RGI year is closer to ys as the minimum spinup '
                     'period -> therefore the minimum spinup period is '
                     'adapted and it is the only period which is tried by the '
                     'dynamic spinup function!')
        min_spinup_period = yr_rgi - ys
        spinup_period = yr_rgi - ys
        min_ice_thickness = 0

    # check that inversion is only possible without providing own fls
    if do_inversion:
        if not np.all([np.all(getattr(fl_prov, 'surface_h') ==
                              getattr(fl_orig, 'surface_h')) and
                       np.all(getattr(fl_prov, 'bed_h') ==
                              getattr(fl_orig, 'bed_h'))
                       for fl_prov, fl_orig in
                       zip(fls_init, gdir.read_pickle('model_flowlines'))]):
            raise InvalidWorkflowError('If you want to perform a dynamic '
                                       'mu_star calibration including an '
                                       'inversion, it is not possible to '
                                       'provide your own flowlines! (fls_init '
                                       'should be None or '
                                       'the original model_flowlines)')

    # Here we start with the actual model run
    if mu_star == gdir.read_json('local_mustar')['mu_star_glacierwide']:
        # we do not need to define a new mu_star or do an inversion
        do_inversion = False
    else:
        define_new_mu_star_in_gdir(gdir, mu_star)

    if do_inversion:
        with utils.DisableLogger():
            apparent_mb_from_any_mb(gdir,
                                    add_to_log_file=False,  # dont write to log
                                    )
            # do inversion with A calibration to current volume
            calibrate_inversion_from_consensus(
                [gdir], apply_fs_on_mismatch=True, error_on_mismatch=False,
                filter_inversion_output=True,
                volume_m3_reference=local_variables['vol_m3_ref'],
                add_to_log_file=False)

    # this is used to keep the original model_flowline unchanged (-> to be able
    # to conduct different dynamic calibration runs in the same gdir)
    model_flowline_filesuffix = '_dyn_mu_calib'
    init_present_time_glacier(gdir, filesuffix=model_flowline_filesuffix,
                              add_to_log_file=False)

    # Now do a dynamic spinup to match area
    # do not ignore errors in dynamic spinup, so all 'bad' files are
    # deleted in run_dynamic_spinup function
    try:
        model, last_best_t_bias = run_dynamic_spinup(
            gdir,
            continue_on_error=False,  # force to raise an error in @entity_task
            add_to_log_file=False,  # dont write to log file in @entity_task
            init_model_fls=fls_init,
            climate_input_filesuffix=climate_input_filesuffix,
            evolution_model=evolution_model,
            mb_model_historical=mb_model_historical,
            mb_model_spinup=mb_model_spinup,
            spinup_period=spinup_period,
            spinup_start_yr=ys,
            spinup_start_yr_max=yr0_ref_mb,
            min_spinup_period=min_spinup_period, yr_rgi=yr_rgi,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            t_bias_max_step_length=t_bias_max_step_length,
            maxiter=maxiter,
            minimise_for=minimise_for,
            first_guess_t_bias=local_variables['t_bias'][-1],
            output_filesuffix=output_filesuffix,
            store_model_evolution=True, ignore_errors=False,
            return_t_bias_best=True, ye=ye,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            model_flowline_filesuffix=model_flowline_filesuffix,
            make_compatible=True,
            **kwargs)
        # save the temperature bias which was successful in the last iteration
        # as we expect we are not so far away in the next iteration (only
        # needed for optimisation, potentially need less iterations in
        # run_dynamic_spinup)
        local_variables['t_bias'].append(last_best_t_bias)
    except RuntimeError as e:
        raise RuntimeError(f'Dynamic spinup raised error! (Message: {e})')

    # calculate dmdtda from previous simulation here
    with utils.DisableLogger():
        ds = utils.compile_run_output(gdir, input_filesuffix=output_filesuffix,
                                      path=False)
    dmdtda_mdl = ((ds.volume.loc[yr1_ref_mb].values -
                   ds.volume.loc[yr0_ref_mb].values) /
                  gdir.rgi_area_m2 /
                  (yr1_ref_mb - yr0_ref_mb) *
                  cfg.PARAMS['ice_density'])

    return model, dmdtda_mdl


def dynamic_mu_star_run_with_dynamic_spinup_fallback(
        gdir, mu_star, fls_init, ys, ye, local_variables, output_filesuffix='',
        evolution_model=FluxBasedModel, minimise_for='area',
        mb_model_historical=None, mb_model_spinup=None,
        climate_input_filesuffix='', spinup_period=20, min_spinup_period=10,
        yr_rgi=None, precision_percent=1,
        precision_absolute=1, min_ice_thickness=10,
        first_guess_t_bias=-2, t_bias_max_step_length=2, maxiter=30,
        store_model_geometry=True, store_fl_diagnostics=None,
        do_inversion=True, **kwargs):
    """
    This is the fallback function corresponding to the function
    'dynamic_mu_star_run_with_dynamic_spinup', which are provided
    to 'run_dynamic_mu_star_calibration'. It is used if the run_function fails and
    if 'ignore_error == True' in 'run_dynamic_mu_star_calibration'. First it resets
    mu_star of gdir. Afterwards it tries to conduct a dynamic spinup. If this
    also fails the last thing is to just do a run without a dynamic spinup.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mu_star : float
        the mu_star used for this run
    fls_init : []
        List of flowlines to use to initialise the model
    ys : int
        start year of the run
    ye : int
        end year of the run
    local_variables : dict
        Dict in which under the key 'vol_m3_ref' the volume which is used in
        'calibrate_inversion_from_consensus'
    output_filesuffix : str
        For the output file.
        Default is ''
    evolution_model : class:oggm.core.FlowlineModel
        Evolution model to use.
        Default is FluxBasedModel
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a PastMassBalance model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    minimise_for : str
        The variable we want to match at yr_rgi. Options are 'area' or 'volume'.
        Default is 'area'.
    climate_input_filesuffix : str
        filesuffix for the input climate file
        Default is ''
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "yr_rgi - spinup_period". Minimum allowed value is defined
        with 'min_spinup_period'. If the provided climate data starts at year
        later than (yr_rgi - spinup_period) the spinup_period is set to
        (yr_rgi - yr_climate_start). Caution if spinup_start_yr is set the
        spinup_period is ignored.
        Default is 20
    min_spinup_period : int
        If the dynamic spinup function fails with the initial 'spinup_period'
        a shorter period is tried. Here you can define the minimum period to
        try.
        Default is 10
    yr_rgi : int or None
        The rgi date, at which we want to match area or volume.
        If None, gdir.rgi_date + 1 is used (the default).
        Default is None
    precision_percent : float
        Gives the precision we want to match for the selected variable
        ('minimise_for') at rgi_date in percent. The algorithm makes sure that
        the resulting relative mismatch is smaller than precision_percent, but
        also that the absolute value is smaller than precision_absolute.
        Default is 1, meaning the difference must be within 1% of the given
        value (area or volume).
    precision_absolute : float
        Gives an minimum absolute value to match. The algorithm makes sure that
        the resulting relative mismatch is smaller than
        precision_percent, but also that the absolute value is
        smaller than precision_absolute.
        The unit of precision_absolute depends on minimise_for (if 'area' in
        km2, if 'volume' in km3)
        Default is 1.
    min_ice_thickness : float
        Gives an minimum ice thickness for model grid points which are counted
        to the total model value. This could be useful to filter out seasonal
        'glacier growth', as OGGM do not differentiate between snow and ice in
        the forward model run. Therefore you could see quite fast changes
        (spikes) in the time-evolution (especially visible in length and area).
        If you set this value to 0 the filtering can be switched off.
        Default is 10.
    first_guess_t_bias : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_bias_max_step_length : float
        Defines the maximums allowed change of t_bias between two iterations. Is
        needed to avoid to large changes.
        Default is 2
    maxiter : int
        Maximum number of minimisation iterations per dynamic spinup where area
        or volume is tried to be matched. If reached and 'ignore_errors=False'
        an error is raised.
        Default is 30
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        Default is True
    store_fl_diagnostics : bool or None
        Whether to store the model flowline diagnostics to disk or not.
        Default is None (-> cfg.PARAMS['store_fl_diagnostics'])
    do_inversion : bool
        If True a complete inversion is conducted using the provided mu_star
        before the actual fallback run.
        Default is False
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final model after the run.
    """
    from oggm.workflow import calibrate_inversion_from_consensus

    if local_variables is None:
        raise RuntimeError('Need the volume to do'
                           'calibrate_inversion_from_consensus provided in '
                           'local_variables!')

    # revert gdir to original state if necessary
    if mu_star != gdir.read_json('local_mustar')['mu_star_glacierwide']:
        define_new_mu_star_in_gdir(gdir, mu_star)
        if do_inversion:
            with utils.DisableLogger():
                apparent_mb_from_any_mb(gdir,
                                        add_to_log_file=False)
                calibrate_inversion_from_consensus(
                    [gdir], apply_fs_on_mismatch=True, error_on_mismatch=False,
                    filter_inversion_output=True,
                    volume_m3_reference=local_variables['vol_m3_ref'],
                    add_to_log_file=False)
    if os.path.isfile(os.path.join(gdir.dir,
                                   'model_flowlines_dyn_mu_calib.pkl')):
        os.remove(os.path.join(gdir.dir,
                               'model_flowlines_dyn_mu_calib.pkl'))

    if yr_rgi is None:
        yr_rgi = gdir.rgi_date + 1  # + 1 converted to hydro years
    if min_spinup_period > yr_rgi - ys:
        log.workflow('The RGI year is closer to ys as the minimum spinup '
                     'period -> therefore the minimum spinup period is '
                     'adapted and it is the only period which is tried by the '
                     'dynamic spinup function!')
        min_spinup_period = yr_rgi - ys
        spinup_period = yr_rgi - ys
        min_ice_thickness = 0

    yr_clim_min = gdir.get_climate_info()['baseline_hydro_yr_0']
    try:
        model_end = run_dynamic_spinup(
            gdir,
            continue_on_error=False,  # force to raise an error in @entity_task
            add_to_log_file=False,
            init_model_fls=fls_init,
            climate_input_filesuffix=climate_input_filesuffix,
            evolution_model=evolution_model,
            mb_model_historical=mb_model_historical,
            mb_model_spinup=mb_model_spinup,
            spinup_period=spinup_period,
            spinup_start_yr=ys,
            min_spinup_period=min_spinup_period,
            yr_rgi=yr_rgi,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            first_guess_t_bias=first_guess_t_bias,
            t_bias_max_step_length=t_bias_max_step_length,
            maxiter=maxiter,
            output_filesuffix=output_filesuffix,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            ignore_errors=False,
            ye=ye,
            make_compatible=True,
            **kwargs)

        gdir.add_to_diagnostics('used_spinup_option', 'dynamic spinup only')

    except RuntimeError:
        log.warning('No dynamic spinup could be conducted by using the '
                    f'original mu* ({mu_star}). Therefore the last '
                    'try is to conduct a run until ye without a dynamic '
                    'spinup.')
        model_end = run_from_climate_data(
            gdir,
            add_to_log_file=False,
            min_ys=yr_clim_min, ye=ye,
            output_filesuffix=output_filesuffix,
            climate_input_filesuffix=climate_input_filesuffix,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            init_model_fls=fls_init, evolution_model=evolution_model,
            fixed_geometry_spinup_yr=ys)

        gdir.add_to_diagnostics('used_spinup_option', 'fixed geometry spinup')

        # set all dynamic diagnostics to None if there where some successful
        # runs
        diag = gdir.get_diagnostics()
        if minimise_for == 'area':
            unit = 'km2'
        elif minimise_for == 'volume':
            unit = 'km3'
        else:
            raise NotImplementedError
        for key in ['temp_bias_dynamic_spinup', 'dynamic_spinup_period',
                    'dynamic_spinup_forward_model_iterations',
                    f'{minimise_for}_mismatch_dynamic_spinup_{unit}_percent',
                    f'reference_{minimise_for}_dynamic_spinup_{unit}',
                    'dynamic_spinup_other_variable_reference',
                    'dynamic_spinup_mismatch_other_variable_percent']:
            if key in diag:
                gdir.add_to_diagnostics(key, None)

        gdir.add_to_diagnostics('run_dynamic_spinup_success', False)
    return model_end


def dynamic_mu_star_run(
        gdir, mu_star, yr0_ref_mb, yr1_ref_mb, fls_init, ys, ye,
        output_filesuffix='', evolution_model=FluxBasedModel,
        local_variables=None, set_local_variables=False, yr_rgi=None,
        **kwargs):
    """
    This function is one option for a 'run_function' for the
    'run_dynamic_mu_star_calibration' function (the corresponding
    'fallback_function' is 'dynamic_mu_star_run_fallback'). It is meant to
    define a new mu_star in the given gdir and conduct a
    'run_from_climate_data' run between ys and ye and return the geodetic mass
    balance (units: kg m-2 yr-1) of the period yr0_ref_mb and yr1_ref_mb.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mu_star : float
        the mu_star used for this run
    yr0_ref_mb : int
        the start year of the geodetic mass balance
    yr1_ref_mb : int
        the end year of the geodetic mass balance
    fls_init : []
        List of flowlines to use to initialise the model
    ys : int
        start year of the complete run, must by smaller or equal y0_ref_mb
    ye : int
        end year of the complete run, must be smaller or equal y1_ref_mb
    output_filesuffix : str
        For the output file.
        Default is ''
    evolution_model : class:oggm.core.FlowlineModel
        Evolution model to use.
        Default is FluxBasedModel
    local_variables : None
        Not needed in this function, just here to match with the function
        call in run_dynamic_mu_star_calibration.
    set_local_variables : bool
        Not needed in this function. Only here to be confirm with the use of
        this function in 'run_dynamic_mu_star_calibration'.
    yr_rgi : int or None
        The rgi year of the gdir.
        Default is None
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`, float
        The final model after the run and the calculated geodetic mass balance
    """

    if set_local_variables:
        # No local variables needed in this function
        return None

    # Here we start with the actual model run
    define_new_mu_star_in_gdir(gdir, mu_star)

    # conduct model run
    try:
        model = run_from_climate_data(gdir,
                                      # force to raise an error in @entity_task
                                      continue_on_error=False,
                                      add_to_log_file=False,
                                      ys=ys, ye=ye,
                                      output_filesuffix=output_filesuffix,
                                      init_model_fls=fls_init,
                                      evolution_model=evolution_model,
                                      **kwargs)
    except RuntimeError as e:
        raise RuntimeError(f'run_from_climate_data raised error! '
                           f'(Message: {e})')

    # calculate dmdtda from previous simulation here
    with utils.DisableLogger():
        ds = utils.compile_run_output(gdir, input_filesuffix=output_filesuffix,
                                      path=False)
    dmdtda_mdl = ((ds.volume.loc[yr1_ref_mb].values -
                   ds.volume.loc[yr0_ref_mb].values) /
                  gdir.rgi_area_m2 /
                  (yr1_ref_mb - yr0_ref_mb) *
                  cfg.PARAMS['ice_density'])

    return model, dmdtda_mdl


def dynamic_mu_star_run_fallback(
        gdir, mu_star, fls_init, ys, ye, local_variables, output_filesuffix='',
        evolution_model=FluxBasedModel, yr_rgi=None, **kwargs):
    """
    This is the fallback function corresponding to the function
    'dynamic_mu_star_run', which are provided to
    'run_dynamic_mu_star_calibration'. It is used if the run_function fails and
    if 'ignore_error=True' in 'run_dynamic_mu_star_calibration'. It sets
    mu_star and conduct a run_from_climate_data run from ys to ye.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    mu_star : float
        the mu_star used for this run
    fls_init : []
        List of flowlines to use to initialise the model
    ys : int
        start year of the run
    ye : int
        end year of the run
    local_variables : dict
        Not needed in this function, just here to match with the function
        call in run_dynamic_mu_star_calibration.
    output_filesuffix : str
        For the output file.
        Default is ''
    evolution_model : class:oggm.core.FlowlineModel
        Evolution model to use.
        Default is FluxBasedModel
    yr_rgi : int or None
        The rgi year of the gdir.
        Default is None
    kwargs : dict
        kwargs to pass to the evolution_model instance

     Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final model after the run.
    """

    define_new_mu_star_in_gdir(gdir, mu_star)

    # conduct model run
    try:
        model = run_from_climate_data(gdir,
                                      # force to raise an error in @entity_task
                                      continue_on_error=False,
                                      add_to_log_file=False,
                                      ys=ys, ye=ye,
                                      output_filesuffix=output_filesuffix,
                                      init_model_fls=fls_init,
                                      evolution_model=evolution_model,
                                      **kwargs)
        gdir.add_to_diagnostics('used_spinup_option', 'no spinup')
    except RuntimeError as e:
        raise RuntimeError(f'In fallback function run_from_climate_data '
                           f'raised error! (Message: {e})')

    return model


@entity_task(log, writes=['inversion_flowlines'])
def run_dynamic_mu_star_calibration(
        gdir, ref_dmdtda=None, err_ref_dmdtda=None, err_dmdtda_scaling_factor=1,
        ref_period='', ignore_hydro_months=False, min_mu_star=None,
        max_mu_star=None, mu_star_max_step_length=5, maxiter=20,
        ignore_errors=False, output_filesuffix='_dynamic_mu_star',
        ys=None, ye=None,
        run_function=dynamic_mu_star_run_with_dynamic_spinup,
        kwargs_run_function=None,
        fallback_function=dynamic_mu_star_run_with_dynamic_spinup_fallback,
        kwargs_fallback_function=None, init_model_filesuffix=None,
        init_model_yr=None, init_model_fls=None,
        first_guess_diagnostic_msg='dynamic spinup only'):
    """Calibrate mu_star to match a geodetic mass balance incorporating a
    dynamic model run.

    This task iteratively search for a mu_star to match a provided geodetic
    mass balance. How one model run looks like is defined in the 'run_function'.
    This function should take a new mu_star guess, conducts a dynamic run and
    calculate the geodetic mass balance. The goal is to match the geodetic mass
    blanance 'ref_dmdtda' inside the provided error 'err_ref_dmdtda'. If the
    minimisation of the mismatch between the provided and modeled geodetic mass
    balance is not working the 'fallback_function' is called. In there it is
    decided what run should be conducted in such a failing case. Further if
    'ignore_error' is set to True and we could not find a satisfying mismatch
    the best run so far is saved (if not one successful run with 'run_function'
    the 'fallback_function' is called).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_dmdtda : float or None
        The reference geodetic mass balance to match (units: kg m-2 yr-1). If
        None the data from Hugonnet 2021 is used.
        Default is None
    err_ref_dmdtda : float or None
        The error of the reference geodetic mass balance to match (unit: kg m-2
        yr-1). Must always be a positive number. If None the data from Hugonett
        2021 is used.
        Default is None
    err_dmdtda_scaling_factor : float
        The error of the geodetic mass balance is multiplied by this factor.
        When looking at more glaciers you should set this factor smaller than
        1 (Default), but the smaller this factor the more glaciers will fail
        during calibration. The factor is only used if ref_dmdtda = None and
        err_ref_dmdtda = None.
        The idea is that we reduce the uncertainty of individual observations
        to count for correlated uncertainties when looking at regional or
        global scales. If err_scaling_factor is 1 (Default) and you look at the
        results of more than one glacier this equals that all errors are
        uncorrelated. Therefore the result will be outside the uncertainty
        boundaries given in Hugonett 2021 e.g. for the global estimate, because
        some correlation of the individual errors is assumed during aggregation
        of glaciers to regions (for more details see paper Hugonett 2021).
    ref_period : str
        If ref_dmdtda is None one of '2000-01-01_2010-01-01',
        '2010-01-01_2020-01-01', '2000-01-01_2020-01-01'. If ref_dmdtda is
        set, this should still match the same format but can be any date.
        Default is '' (-> PARAMS['geodetic_mb_period'])
    ignore_hydro_months : bool
        Do not raise an  error if we are working on calendar years.
        Default is False
    min_mu_star : float or None
        Lower absolute limit for mu*.
        Default is None (-> cfg.PARAMS['min_mu_star'])
    max_mu_star : float or None
        Upper absolute limit for mu*
        Default is None (-> cfg.PARAMS['max_mu_star'])
    mu_star_max_step_length : float
        Defines the maximum allowed change of mu_star between two iterations.
        Is needed to avoid to large changes.
        Default is 5
    maxiter : int
        Maximum number of minimisation iterations of minimising mismatch to
        dmdtda by changing mu_star. Each of this iterations conduct a complete
        run defined in the 'run_function'. If maxiter_mu_star reached and
        'ignore_errors=False' an error is raised.
        Default is 20
    ignore_errors : bool
        If True and the 'run_function' with mu* star calibration is not working
        to match dmdtda inside the provided uncertainty first, if their where
        some successful runs with the 'run_function' they are saved as part
        success, and if not a single run was successful the 'fallback_function'
        is called. If False and the 'run_function' with mu* star calibration is
        not working an Error is raised.
        Default is True
    output_filesuffix : str
        For the output file.
        Default is '_historical_dynamic_mu_star'
    ys : int or None
        The start year of the conducted run. If None the first year of the
        provided climate file.
        Default is None
    ye : int or None
        The end year of the conducted run. If None the last year of the
        provided climate file.
        Default is None
    run_function : function
        This function defines how a new defined mu_star is used to conduct the
        next model run. This function must contain the arguments 'gdir',
        'mu_star', 'yr0_ref_mb', 'yr1_ref_mb', 'fls_init', 'ys', 'ye' and
        'output_filesuffix'. Further this function must return the final model
        and the calculated geodetic mass balance dmdtda in kg m-2 yr-1.
    kwargs_run_function : None or dict
        Can provide additional keyword arguments to the run_function as a
        dictionary.
    fallback_function : function
        This is a fallback function if the calibration is not working using
        'run_function' it is called. This function must contain the arguments
        'gdir', 'mu_star', 'fls_init', 'ys', 'ye', 'local_variables' and
        'output_filesuffix'. Further this function should return the final
        model.
    kwargs_fallback_function : None or dict
        Can provide additional keyword arguments to the fallback_function as a
        dictionary.
    init_model_filesuffix : str or None
        If you want to start from a previous model run state. This state
        should be at time yr_rgi_date.
        Default is None
    init_model_yr : int or None
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        List of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set.
    first_guess_diagnostic_msg : str
        This message will be added to the glacier diagnostics if only the
        default mu* resulted in a successful 'run_function' run.
        Default is 'dynamic spinup only'

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final dynamically spined-up model. Type depends on the selected
        evolution_model, by default a FluxBasedModel.
    """
    # mu* constraints
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']

    if kwargs_run_function is None:
        kwargs_run_function = {}
    if kwargs_fallback_function is None:
        kwargs_fallback_function = {}

    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    if sm != 1 and not ignore_hydro_months:
        raise InvalidParamsError('run_dynamic_mu_star_calibration makes '
                                 'more sense when applied on calendar years '
                                 "(PARAMS['hydro_month_nh']=1 and "
                                 "`PARAMS['hydro_month_sh']=1). If you want "
                                 "to ignore this error, set "
                                 "ignore_hydro_months to True")

    if max_mu_star > 1000:
        raise InvalidParamsError('You seem to have set a very high '
                                 'max_mu_star for this run. This is not '
                                 'how this task is supposed to work, and '
                                 'we recommend a value lower than 1000 '
                                 '(or even 600). You can directly provide a '
                                 'value for max_mu_star or set '
                                 "cfg.PARAMS['max_mu_star'] to a smaller value!")

    # geodetic mb stuff
    if not ref_period:
        ref_period = cfg.PARAMS['geodetic_mb_period']
    # if a reference geodetic mb is specified also the error of it must be
    # specified, and vice versa
    if ((ref_dmdtda is None and err_ref_dmdtda is not None) or
            (ref_dmdtda is not None and err_ref_dmdtda is None)):
        raise RuntimeError('If you provide a reference geodetic mass balance '
                           '(ref_dmdtda) you must also provide an error for it '
                           '(err_ref_dmdtda), and vice versa!')
    # Get the reference geodetic mb and error if not given
    if ref_dmdtda is None:
        df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        # reference geodetic mass balance from Hugonnet 2021
        ref_dmdtda = float(
            df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period]['dmdtda'])
        # dmdtda: in meters water-equivalent per year -> we convert
        ref_dmdtda *= 1000  # kg m-2 yr-1
        # error of reference geodetic mass balance from Hugonnet 2021
        err_ref_dmdtda = float(df_ref_dmdtda.loc[df_ref_dmdtda['period'] ==
                                                 ref_period]['err_dmdtda'])
        err_ref_dmdtda *= 1000  # kg m-2 yr-1
        err_ref_dmdtda *= err_dmdtda_scaling_factor

    if err_ref_dmdtda <= 0:
        raise RuntimeError('The provided error for the geodetic mass-balance '
                           '(err_ref_dmdtda) must be positive and non zero! But'
                           f'given was {err_ref_dmdtda}!')
    # get start and end year of geodetic mb
    yr0_ref_mb, yr1_ref_mb = ref_period.split('_')
    yr0_ref_mb = int(yr0_ref_mb.split('-')[0])
    yr1_ref_mb = int(yr1_ref_mb.split('-')[0])

    if ye is None:
        # One adds 1 because the run ends at the end of the year
        ye = gdir.get_climate_info()['baseline_hydro_yr_1'] + 1

    if ye < yr1_ref_mb:
        raise RuntimeError('The provided ye is smaller than the end year of '
                           'the given geodetic_mb_period!')

    if ys is None:
        ys = gdir.get_climate_info()['baseline_hydro_yr_0']

    if ys > yr0_ref_mb:
        raise RuntimeError('The provided ys is larger than the start year of '
                           'the given geodetic_mb_period!')

    yr_rgi = gdir.rgi_date + 1  # + 1 converted to hydro years
    if yr_rgi < ys:
        if ignore_errors:
            log.workflow('The rgi year is smaller than the provided start year '
                         'ys -> setting the rgi year to ys to continue!')
            yr_rgi = ys
        else:
            raise RuntimeError('The rgi year is smaller than the provided '
                               'start year ys!')
    kwargs_run_function['yr_rgi'] = yr_rgi
    kwargs_fallback_function['yr_rgi'] = yr_rgi

    # get initial flowlines from which we want to start from
    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls

    if init_model_fls is None:
        fls_init = gdir.read_pickle('model_flowlines')
    else:
        fls_init = copy.deepcopy(init_model_fls)

    # save original mu star for later to be able to recreate original gdir
    # (using the fallback function) if an error occurs
    mu_star_initial = gdir.read_json('local_mustar')['mu_star_glacierwide']

    # only used to check performance of minimisation
    dynamic_mu_star_calibration_runs = [0]

    # this function is called if the actual dynamic mu star calibration fails
    def fallback_run(mu_star, reset, best_mismatch=None, initial_mismatch=None,
                     only_first_guess=None):
        if reset:
            # unfortunately we could not conduct an error free run using the
            # provided run_function, so we us the fallback_function

            # this diagnostics should be overwritten inside the fallback_function
            gdir.add_to_diagnostics('used_spinup_option', 'fallback_function')

            model = fallback_function(gdir=gdir, mu_star=mu_star,
                                      fls_init=fls_init, ys=ys, ye=ye,
                                      local_variables=local_variables_run_function,
                                      output_filesuffix=output_filesuffix,
                                      **kwargs_fallback_function)
        else:
            # we were not able to reach the desired precision during the
            # minimisation, but at least we conducted a few error free runs
            # using the run_function, and therefore we save the best guess we
            # found so far
            if only_first_guess:
                gdir.add_to_diagnostics('used_spinup_option',
                                        first_guess_diagnostic_msg)
            else:
                gdir.add_to_diagnostics('used_spinup_option',
                                        'dynamic mu_star calibration (part '
                                        'success)')
            model, dmdtda_mdl = run_function(gdir=gdir, mu_star=mu_star,
                                             yr0_ref_mb=yr0_ref_mb,
                                             yr1_ref_mb=yr1_ref_mb,
                                             fls_init=fls_init, ys=ys, ye=ye,
                                             output_filesuffix=output_filesuffix,
                                             local_variables=local_variables_run_function,
                                             **kwargs_run_function)

            gdir.add_to_diagnostics(
                'dmdtda_mismatch_dynamic_calibration_reference',
                float(ref_dmdtda))
            gdir.add_to_diagnostics(
                'dmdtda_dynamic_calibration_given_error',
                float(err_ref_dmdtda))
            gdir.add_to_diagnostics(
                'dmdtda_mismatch_dynamic_calibration',
                float(best_mismatch))
            gdir.add_to_diagnostics(
                'dmdtda_mismatch_with_initial_mu_star',
                float(initial_mismatch))
            gdir.add_to_diagnostics('mu_star_dynamic_calibration',
                                    float(mu_star))
            gdir.add_to_diagnostics('mu_star_before_dynamic_calibration',
                                    float(mu_star_initial))
            gdir.add_to_diagnostics('run_dynamic_mu_star_calibration_iterations',
                                    int(dynamic_mu_star_calibration_runs[-1]))

        return model

    # here we define the local variables which are used in the run_function,
    # for some run_functions this is useful to save parameters from a previous
    # run to be faster in the upcoming runs
    local_variables_run_function = {}
    run_function(gdir=gdir, mu_star=None, yr0_ref_mb=None, yr1_ref_mb=None,
                 fls_init=None, ys=None, ye=None,
                 local_variables=local_variables_run_function,
                 set_local_variables=True, **kwargs_run_function)

    # this is the actual model run which is executed each iteration in order to
    # minimise the mismatch of dmdtda of model and observation
    def model_run(mu_star):
        # to check performance of minimisation
        dynamic_mu_star_calibration_runs.append(
            dynamic_mu_star_calibration_runs[-1] + 1)

        model, dmdtda_mdl = run_function(gdir=gdir, mu_star=mu_star,
                                         yr0_ref_mb=yr0_ref_mb,
                                         yr1_ref_mb=yr1_ref_mb,
                                         fls_init=fls_init, ys=ys, ye=ye,
                                         output_filesuffix=output_filesuffix,
                                         local_variables=local_variables_run_function,
                                         **kwargs_run_function)
        return model, dmdtda_mdl

    def cost_fct(mu_star, model_dynamic_spinup_end):

        # actual model run
        model_dynamic_spinup, dmdtda_mdl = model_run(mu_star)

        # save final model for later
        model_dynamic_spinup_end.append(copy.deepcopy(model_dynamic_spinup))

        # calculate the mismatch of dmdtda
        cost = float(dmdtda_mdl - ref_dmdtda)

        return cost

    def init_cost_fun():
        model_dynamic_spinup_end = []

        def c_fun(mu_star):
            return cost_fct(mu_star, model_dynamic_spinup_end)

        return c_fun, model_dynamic_spinup_end

    # Here start with own spline minimisation algorithm
    def minimise_with_spline_fit(fct_to_minimise, mu_star_guess, mismatch):
        # defines limits of mu in accordance to maximal allowed change
        # between iterations
        mu_star_limits = [mu_star_initial - mu_star_max_step_length,
                          mu_star_initial + mu_star_max_step_length]

        # this two variables indicate that the limits were already adapted to
        # avoid an error
        was_min_error = False
        was_max_error = False
        was_errors = [was_min_error, was_max_error]

        def get_mismatch(mu_star):
            mu_star = copy.deepcopy(mu_star)
            # first check if the mu_star is inside limits
            if mu_star < mu_star_limits[0]:
                # was the smaller limit already executed, if not first do this
                if mu_star_limits[0] not in mu_star_guess:
                    mu_star = copy.deepcopy(mu_star_limits[0])
                else:
                    # smaller limit was already used, check if it was
                    # already newly defined with error
                    if was_errors[0]:
                        raise RuntimeError('Not able to minimise without '
                                           'raising an error at lower limit of '
                                           'mu star!')
                    else:
                        # ok we set a new lower limit, consider also minimum
                        # limit
                        mu_star_limits[0] = max(min_mu_star,
                                                mu_star_limits[0] -
                                                mu_star_max_step_length)
            elif mu_star > mu_star_limits[1]:
                # was the larger limit already executed, if not first do this
                if mu_star_limits[1] not in mu_star_guess:
                    mu_star = copy.deepcopy(mu_star_limits[1])
                else:
                    # larger limit was already used, check if it was
                    # already newly defined with ice free glacier
                    if was_errors[1]:
                        raise RuntimeError('Not able to minimise without '
                                           'raising an error at upper limit of '
                                           'mu star!')
                    else:
                        # ok we set a new upper limit, consider also maximum
                        # limit
                        mu_star_limits[1] = min(max_mu_star,
                                                mu_star_limits[1] +
                                                mu_star_max_step_length)

            # now clip mu_star with limits (to be sure)
            mu_star = np.clip(mu_star, mu_star_limits[0], mu_star_limits[1])
            if mu_star in mu_star_guess:
                raise RuntimeError('This mu star was already tried. Probably '
                                   'we are at one of the max or min limit and '
                                   'still have no satisfactory mismatch '
                                   'found!')

            # if error during dynamic calibration this defines how much
            # mu_star is changed in the upcoming iteratoins to look for an
            # error free run
            mu_star_search_change = mu_star_max_step_length / 10
            # maximum number of changes to look for an error free run
            max_iterations = int(mu_star_max_step_length /
                                 mu_star_search_change)

            current_min_error = False
            current_max_error = False
            doing_first_guess = (len(mismatch) == 0)
            iteration = 0
            current_mu_star = copy.deepcopy(mu_star)

            # in this loop if an error at the limits is raised we go step by
            # step away from the limits until we are at the initial guess or we
            # found an error free run
            tmp_mismatch = None
            while ((current_min_error | current_max_error | iteration == 0) &
                   (iteration < max_iterations)):
                try:
                    tmp_mismatch = fct_to_minimise(mu_star)
                except RuntimeError as e:
                    # check if we are at the lower limit
                    if mu_star == mu_star_limits[0]:
                        # check if there was already an error at the lower limit
                        if was_errors[0]:
                            raise RuntimeError('Second time with error at '
                                               'lower limit of mu star! '
                                               'Error message of model run: '
                                               f'{e}')
                        else:
                            was_errors[0] = True
                            current_min_error = True

                    # check if we are at the upperlimit
                    elif mu_star == mu_star_limits[1]:
                        # check if there was already an error at the lower limit
                        if was_errors[1]:
                            raise RuntimeError('Second time with error at '
                                               'upper limit of mu star! '
                                               'Error message of model run: '
                                               f'{e}')
                        else:
                            was_errors[1] = True
                            current_max_error = True

                    if current_min_error:
                        # currently we searching for a new lower limit with no
                        # error
                        mu_star = np.round(mu_star + mu_star_search_change,
                                           decimals=1)
                    elif current_max_error:
                        # currently we searching for a new upper limit with no
                        # error
                        mu_star = np.round(mu_star - mu_star_search_change,
                                           decimals=1)

                    # if we end close to an already executed guess while
                    # searching for a new limit we quite
                    if np.isclose(mu_star, mu_star_guess).any():
                        raise RuntimeError('Not able to further minimise, '
                                           'return the best we have so far!'
                                           f'Error message: {e}')

                    if doing_first_guess:
                        # unfortunately first guess is not working
                        raise RuntimeError('Dynamic calibration is not working '
                                           'with first guess! Error '
                                           f'message: {e}')

                    if np.isclose(mu_star, current_mu_star):
                        # something unexpected happen so we end here
                        raise RuntimeError('Unexpected error not at the limits'
                                           f' of mu star. Error Message: {e}')

                iteration += 1

            if iteration >= max_iterations:
                # ok we were not able to find an mismatch without error
                if current_min_error:
                    raise RuntimeError('Not able to find new lower limit for '
                                       'mu star!')
                elif current_max_error:
                    raise RuntimeError('Not able to find new upper limit for '
                                       'mu star!')
                else:
                    raise RuntimeError('Something unexpected happened during '
                                       'definition of new mu star limits!')
            else:
                # if we found a new limit set it
                if current_min_error:
                    mu_star_limits[0] = copy.deepcopy(mu_star)
                elif current_max_error:
                    mu_star_limits[1] = copy.deepcopy(mu_star)

            if tmp_mismatch is None:
                raise RuntimeError('Not able to find a new mismatch for '
                                   'dmdtda!')

            return float(tmp_mismatch), float(mu_star)

        # first guess
        new_mismatch, new_mu_star = get_mismatch(mu_star_initial)
        mu_star_guess.append(new_mu_star)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < err_ref_dmdtda:
            return mismatch[-1], new_mu_star

        # second (arbitrary) guess is given depending on the outcome of first
        # guess, mu_star is changed for percent of mismatch relative to
        # err_ref_dmdtda times mu_star_max_step_length (if
        # mismatch = 2 * err_ref_dmdtda this corresponds to 100%; for 100% or
        # 150% the next step is (-1) * mu_star_max_step_length; if mismatch
        # -40%, next step is 0.4 * mu_star_max_step_length; but always at least
        # a change of 0.5 is imposed to prevent too close guesses). (-1) as if
        # mismatch is negative we need a larger mu_star to get closer to 0
        step = (-1) * np.sign(mismatch[-1]) * \
            max((np.abs(mismatch[-1]) - err_ref_dmdtda) / err_ref_dmdtda *
                mu_star_max_step_length, 0.5)
        new_mismatch, new_mu_star = get_mismatch(mu_star_guess[0] + step)
        mu_star_guess.append(new_mu_star)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < err_ref_dmdtda:
            return mismatch[-1], new_mu_star

        # Now start with splin fit for guessing
        while len(mu_star_guess) < maxiter:
            # get next guess from splin (fit partial linear function to
            # previously calculated (mismatch, mu_star) pairs and get mu_star
            # value where mismatch=0 from this fitted curve)
            sort_index = np.argsort(np.array(mismatch))
            tck = interpolate.splrep(np.array(mismatch)[sort_index],
                                     np.array(mu_star_guess)[sort_index],
                                     k=1)
            # here we catch interpolation errors (two different mu_star with
            # same mismatch), could happen if one mu_star was close to a newly
            # defined limit
            if np.isnan(tck[1]).any():
                if was_errors[0]:
                    raise RuntimeError('Second time with error at lower '
                                       'limit of mu star! (nan in splin fit)')
                elif was_errors[1]:
                    raise RuntimeError('Second time with error at upper '
                                       'limit of mu star! (nan in splin fit)')
                else:
                    raise RuntimeError('Not able to minimise! Problem is '
                                       'unknown. (nan in splin fit)')
            new_mismatch, new_mu_star = get_mismatch(
                float(interpolate.splev(0, tck)))
            mu_star_guess.append(new_mu_star)
            mismatch.append(new_mismatch)

            if abs(mismatch[-1]) < err_ref_dmdtda:
                return mismatch[-1], new_mu_star

        # Ok when we end here the spinup could not find satisfying match after
        # maxiter(ations)
        raise RuntimeError(f'Could not find mismatch smaller '
                           f'{err_ref_dmdtda} kg m-2 yr-1 (only '
                           f'{np.min(np.abs(mismatch))} kg m-2 yr-1) in '
                           f'{maxiter} Iterations!')

    # wrapper to get values for intermediate (mismatch, mu_star) guesses if an
    # error is raised
    def init_minimiser():
        mu_star_guess = []
        mismatch = []

        def minimiser(fct_to_minimise):
            return minimise_with_spline_fit(fct_to_minimise, mu_star_guess,
                                            mismatch)

        return minimiser, mu_star_guess, mismatch

    # define function for the actual minimisation
    c_fun, models_dynamic_spinup_end = init_cost_fun()

    # define minimiser
    minimise_given_fct, mu_star_guesses, mismatch_dmdtda = init_minimiser()

    try:
        final_mismatch, final_mu_star = minimise_given_fct(c_fun)
    except RuntimeError as e:
        # something happened during minimisation, if there where some
        # successful runs we return the one with the best mismatch, otherwise
        # we conduct just a run with no dynamic spinup
        if len(mismatch_dmdtda) == 0:
            # we conducted no successful run, so run without dynamic spinup
            if ignore_errors:
                log.workflow('Dynamic mu star calibration not successful. '
                             f'Error message: {e}')
                model_return = fallback_run(mu_star=mu_star_initial,
                                            reset=True)
                return model_return
            else:
                raise RuntimeError('Dynamic mu star calibration was not '
                                   f'successful! Error Message: {e}')
        else:
            if ignore_errors:
                log.workflow('Dynamic mu star calibration not successful. Error '
                             f'message: {e}')

                # there where some successful runs so we return the one with the
                # smallest mismatch of dmdtda
                min_mismatch_index = np.argmin(np.abs(mismatch_dmdtda))
                mu_star_best = np.array(mu_star_guesses)[min_mismatch_index]

                # check if the first guess was the best guess
                only_first_guess = False
                if min_mismatch_index == 1:
                    only_first_guess = True

                model_return = fallback_run(
                    mu_star=mu_star_best, reset=False,
                    best_mismatch=np.array(mismatch_dmdtda)[min_mismatch_index],
                    initial_mismatch=mismatch_dmdtda[0],
                    only_first_guess=only_first_guess)

                return model_return
            else:
                raise RuntimeError('Dynamic mu star calibration not successful. '
                                   f'Error message: {e}')

    # check that new mu star is correctly saved in gdir
    assert final_mu_star == gdir.read_json('local_mustar')['mu_star_glacierwide']

    # hurray, dynamic mu star calibration successful
    gdir.add_to_diagnostics('used_spinup_option',
                            'dynamic mu_star calibration (full success)')
    gdir.add_to_diagnostics('dmdtda_mismatch_dynamic_calibration_reference',
                            float(ref_dmdtda))
    gdir.add_to_diagnostics('dmdtda_dynamic_calibration_given_error',
                            float(err_ref_dmdtda))
    gdir.add_to_diagnostics('dmdtda_mismatch_dynamic_calibration',
                            float(final_mismatch))
    gdir.add_to_diagnostics('dmdtda_mismatch_with_initial_mu_star',
                            float(mismatch_dmdtda[0]))
    gdir.add_to_diagnostics('mu_star_dynamic_calibration', float(final_mu_star))
    gdir.add_to_diagnostics('mu_star_before_dynamic_calibration',
                            float(mu_star_initial))
    gdir.add_to_diagnostics('run_dynamic_mu_star_calibration_iterations',
                            int(dynamic_mu_star_calibration_runs[-1]))

    log.workflow(f'Dynamic mu star calibration worked for {gdir.rgi_id}!')

    return models_dynamic_spinup_end[-1]
