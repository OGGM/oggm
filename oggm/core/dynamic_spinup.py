"""Dynamic spinup functions for model initialisation"""
# Builtins
import logging
import copy
import os
import warnings

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
                                   MonthlyTIModel,
                                   apparent_mb_from_any_mb)
from oggm.core.flowline import (decide_evolution_model, FileModel,
                                init_present_time_glacier,
                                run_from_climate_data)

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def run_dynamic_spinup(gdir, init_model_filesuffix=None, init_model_yr=None,
                       init_model_fls=None,
                       climate_input_filesuffix='',
                       evolution_model=None,
                       mb_model_historical=None, mb_model_spinup=None,
                       spinup_period=20, spinup_start_yr=None,
                       min_spinup_period=10, spinup_start_yr_max=None,
                       target_yr=None, target_value=None,
                       minimise_for='area', precision_percent=1,
                       precision_absolute=1, min_ice_thickness=None,
                       first_guess_t_spinup=-2, t_spinup_max_step_length=2,
                       maxiter=30, output_filesuffix='_dynamic_spinup',
                       store_model_geometry=True, store_fl_diagnostics=None,
                       store_model_evolution=True, ignore_errors=False,
                       return_t_spinup_best=False, ye=None,
                       model_flowline_filesuffix='',
                       add_fixed_geometry_spinup=False, **kwargs):
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
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a MonthlyTIModel model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "target_yr - spinup_period". Minimum allowed value is 10. If
        the provided climate data starts at year later than
        (target_yr - spinup_period) the spinup_period is set to
        (target_yr - yr_climate_start). Caution if spinup_start_yr is set the
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
        target_yr - spinup_start_yr_max > min_spinup_period.
        Default is None
    target_yr : int or None
        The year at which we want to match area or volume.
        If None, gdir.rgi_date + 1 is used (the default).
        Default is None
    target_value : float or None
        The value we want to match at target_yr. Depending on minimise_for this
        value is interpreted as an area in km2 or a volume in km3. If None the
        total area or volume from the provided initial flowlines is used.
        Default is None
    minimise_for : str
        The variable we want to match at target_yr. Options are 'area' or
        'volume'.
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
        Default is cfg.PARAMS['dynamic_spinup_min_ice_thick'].
    first_guess_t_spinup : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_spinup_max_step_length : float
        Defines the maximums allowed change of t_spinup between two iterations.
        Is needed to avoid to large changes.
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
    return_t_spinup_best : bool
        If True the used temperature bias for the spinup is returned in
        addition to the final model. If an error occurs and ignore_error=True,
        the returned value is np.nan.
        Default is False
    ye : int
        end year of the model run, must be larger than target_yr. If nothing is
        given it is set to target_yr. It is not recommended to use it if only
        data until target_yr is needed for calibration as this increases the
        run time of each iteration during the iterative minimisation. Instead
        use run_from_climate_data afterwards and merge both outputs using
        merge_consecutive_run_outputs.
        Default is None
    model_flowline_filesuffix : str
        suffix to the model_flowlines filename to use (if no other flowlines
        are provided with init_model_filesuffix or init_model_fls).
        Default is ''
    add_fixed_geometry_spinup : bool
        If True and the original spinup_period must be shortened (due to
        ice-free or out-of-boundary error) a fixed geometry spinup is added at
        the beginning so that the resulting model run always starts from the
        defined start year (could be defined through spinup_period or
        spinup_start_yr). Only has an effect if store_model_evolution is True.
        Default is False
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final dynamically spined-up model. Type depends on the selected
        evolution_model.
    """

    evolution_model = decide_evolution_model(evolution_model)

    if target_yr is None:
        # Even in calendar dates, we prefer to set rgi_year in the next year
        # as the rgi is often from snow free images the year before (e.g. Aug)
        target_yr = gdir.rgi_date + 1

    if ye is None:
        ye = target_yr

    if ye < target_yr:
        raise RuntimeError(f'The provided end year (ye = {ye}) must be larger'
                           f'than the target year (target_yr = {target_yr}!')

    yr_min = gdir.get_climate_info()['baseline_yr_0']

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
            if spinup_start_yr_max < spinup_start_yr:
                raise RuntimeError(f'The provided start year (= '
                                   f'{spinup_start_yr}) must be smaller than '
                                   f'the maximum start year '
                                   f'{spinup_start_yr_max}!')
        if (target_yr - spinup_start_yr_max) > min_spinup_period:
            min_spinup_period = (target_yr - spinup_start_yr_max)

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

    # MassBalance for actual run from yr_spinup to target_yr
    if mb_model_historical is None:
        mb_model_historical = MultipleFlowlineMassBalance(
            gdir, mb_model_class=MonthlyTIModel,
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
    # This is necessary if target_yr < yr_min + 10 or if the dynamic spinup failed.
    def save_model_without_dynamic_spinup():
        gdir.add_to_diagnostics('run_dynamic_spinup_success', False)
        yr_use = np.clip(target_yr, yr_min, None)
        model_dynamic_spinup_end = evolution_model(fls_spinup,
                                                   mb_model_historical,
                                                   y0=yr_use,
                                                   **kwargs)

        with warnings.catch_warnings():
            if ye < yr_use:
                yr_run = yr_use
            else:
                yr_run = ye
            # For operational runs we ignore the warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_dynamic_spinup_end.run_until_and_store(
                yr_run,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path)

        return model_dynamic_spinup_end

    if target_yr < yr_min + min_spinup_period:
        log.warning('The provided rgi_date is smaller than yr_climate_start + '
                    'min_spinup_period, therefore no dynamic spinup is '
                    'conducted and the original flowlines are saved at the '
                    'provided target year or the start year of the provided '
                    'climate data (if yr_climate_start > target_yr)')
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_spinup_best:
                return model_dynamic_spinup_end, np.nan
            else:
                return model_dynamic_spinup_end
        else:
            raise RuntimeError('The difference between the rgi_date and the '
                               'start year of the climate data is too small to '
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
    if target_value is None:
        reference_value = np.sum([getattr(f, cost_var) for f in fls_ref])
    else:
        reference_value = target_value
    other_reference_value = np.sum([getattr(f, f'{other_variable}_{other_unit}')
                                    for f in fls_ref])

    # if reference value is zero no dynamic spinup is possible
    if reference_value == 0.:
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_spinup_best:
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
    def run_model_with_spinup_to_target_year(t_spinup):
        forward_model_runs.append(forward_model_runs[-1] + 1)

        # with t_spinup the glacier state after spinup is changed between iterations
        mb_model_spinup.temp_bias = t_spinup
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
            # check if we need to add the min_h variable (done inplace)
            delete_area_min_h = False
            ovars = cfg.PARAMS['store_diagnostic_variables']
            if 'area_min_h' not in ovars:
                ovars += ['area_min_h']
                delete_area_min_h = True

            ds = model_historical.run_until_and_store(
                ye,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path,
                dynamic_spinup_min_ice_thick=min_ice_thickness,
                fixed_geometry_spinup_yr=fixed_geometry_spinup_yr)

            # now we delete the min_h variable again if it was not
            # included before (inplace)
            if delete_area_min_h:
                ovars.remove('area_min_h')

            if type(ds) == tuple:
                ds = ds[0]
            model_area_km2 = ds.area_m2_min_h.loc[target_yr].values * 1e-6
            model_volume_km3 = ds.volume_m3.loc[target_yr].values * 1e-9
        else:
            # only run to rgi date and extract values
            model_historical.run_until(target_yr)
            fls = model_historical.fls
            model_area_km2 = np.sum(
                [np.sum(fl.bin_area_m2[fl.thick > min_ice_thickness])
                 for fl in fls]) * 1e-6
            model_volume_km3 = model_historical.volume_km3
            # afterwards finish the complete run
            model_historical.run_until(ye)

        if cost_var == 'area_km2':
            return model_area_km2, model_volume_km3, model_historical, ice_free
        elif cost_var == 'volume_km3':
            return model_volume_km3, model_area_km2, model_historical, ice_free
        else:
            raise NotImplementedError(f'{cost_var}')

    def cost_fct(t_spinup, model_dynamic_spinup_end_loc, other_variable_mismatch_loc):
        # actual model run
        model_value, other_value, model_dynamic_spinup, ice_free = \
            run_model_with_spinup_to_target_year(t_spinup)

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

        def c_fct(t_spinup):
            return cost_fct(t_spinup, model_dynamic_spinup_end_loc,
                            other_variable_mismatch_loc)

        return c_fct, model_dynamic_spinup_end_loc, other_variable_mismatch_loc

    def minimise_with_spline_fit(fct_to_minimise):
        # defines limits of t_spinup in accordance to maximal allowed change
        # between iterations
        t_spinup_limits = [first_guess_t_spinup - t_spinup_max_step_length,
                           first_guess_t_spinup + t_spinup_max_step_length]
        t_spinup_guess = []
        mismatch = []
        # this two variables indicate that the limits were already adapted to
        # avoid an ice_free or out_of_domain error
        was_ice_free = False
        was_out_of_domain = False
        was_errors = [was_out_of_domain, was_ice_free]

        def get_mismatch(t_spinup):
            t_spinup = copy.deepcopy(t_spinup)
            # first check if the new t_spinup is in limits
            if t_spinup < t_spinup_limits[0]:
                # was the smaller limit already executed, if not first do this
                if t_spinup_limits[0] not in t_spinup_guess:
                    t_spinup = copy.deepcopy(t_spinup_limits[0])
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
                        t_spinup_limits[0] = (t_spinup_limits[0] -
                                              t_spinup_max_step_length)
            elif t_spinup > t_spinup_limits[1]:
                # was the larger limit already executed, if not first do this
                if t_spinup_limits[1] not in t_spinup_guess:
                    t_spinup = copy.deepcopy(t_spinup_limits[1])
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
                        t_spinup_limits[1] = (t_spinup_limits[1] +
                                              t_spinup_max_step_length)

            # now clip t_spinup with limits
            t_spinup = np.clip(t_spinup, t_spinup_limits[0], t_spinup_limits[1])

            # now start with mismatch calculation

            # if error during spinup (ice_free or out_of_domain) this defines
            # how much t_spinup is changed to look for an error free glacier spinup
            t_spinup_search_change = t_spinup_max_step_length / 10
            # maximum number of changes to look for an error free glacier
            max_iterations = int(t_spinup_max_step_length / t_spinup_search_change)
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
                    tmp_mismatch, is_ice_free_spinup = fct_to_minimise(t_spinup)

                    # no error occurred, so we are not outside the domain
                    is_out_of_domain = False

                    # check if we are ice_free after spinup, if so we search
                    # for a new upper limit for t_spinup
                    if is_ice_free_spinup:
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_spinup = copy.deepcopy(t_spinup_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup - t_spinup_max_step_length
                        else:
                            t_spinup = np.round(t_spinup - t_spinup_search_change,
                                                decimals=1)
                        if np.isclose(t_spinup, t_spinup_guess).any():
                            iteration = copy.deepcopy(max_iterations)

                    # check if we are ice_free at the end of the model run, if
                    # so we use the lower t_spinup limit and change the limit if
                    # needed
                    elif np.isclose(tmp_mismatch, -100.):
                        is_ice_free_end = True
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_spinup = copy.deepcopy(t_spinup_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup - t_spinup_max_step_length
                        else:
                            # if lower limit was already used change it and use
                            if t_spinup == t_spinup_limits[0]:
                                t_spinup_limits[0] = (t_spinup_limits[0] -
                                                      t_spinup_max_step_length)
                                t_spinup = copy.deepcopy(t_spinup_limits[0])
                            else:
                                # otherwise just try with a colder t_spinup
                                t_spinup = np.round(t_spinup - t_spinup_search_change,
                                                    decimals=1)

                    else:
                        is_ice_free_end = False

                except RuntimeError as e:
                    # check if glacier grow to large
                    if 'Glacier exceeds domain boundaries, at year:' in f'{e}':
                        # ok we where outside the domain, therefore we search
                        # for a new lower limit for t_spinup in 0.1 °C steps
                        is_out_of_domain = True
                        define_new_lower_limit = True
                        was_errors[0] = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_out_of_domain = True
                            # here directly jump to the larger limit
                            t_spinup = t_spinup_limits[1]
                        elif is_first_guess_out_of_domain & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup + t_spinup_max_step_length
                        else:
                            t_spinup = np.round(t_spinup + t_spinup_search_change,
                                                decimals=1)
                        if np.isclose(t_spinup, t_spinup_guess).any():
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
                        msg += f'"ice_free" with last t_spinup of {t_spinup}.'
                    elif is_first_guess_out_of_domain:
                        msg += f'"out_of_domain" with last t_spinup of {t_spinup}.'
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
                    raise RuntimeError('Not able to find a t_spinup so that '
                                       'glacier is not ice free at the end! '
                                       '(Last t_spinup '
                                       f'{t_spinup + t_spinup_max_step_length} °C)')
                else:
                    raise RuntimeError('Something unexpected happened during '
                                       'definition of new t_spinup limits!')
            else:
                # if we found a new limit set it
                if define_new_upper_limit & define_new_lower_limit:
                    # we can end here if we are at the first guess and took
                    # a to large step
                    was_errors[0] = False
                    was_errors[1] = False
                    if t_spinup <= t_spinup_limits[0]:
                        t_spinup_limits[0] = t_spinup
                        t_spinup_limits[1] = (t_spinup_limits[0] +
                                              t_spinup_max_step_length)
                    elif t_spinup >= t_spinup_limits[1]:
                        t_spinup_limits[1] = t_spinup
                        t_spinup_limits[0] = (t_spinup_limits[1] -
                                              t_spinup_max_step_length)
                    else:
                        if is_first_guess_ice_free:
                            t_spinup_limits[1] = t_spinup
                        elif is_out_of_domain:
                            t_spinup_limits[0] = t_spinup
                        else:
                            raise RuntimeError('I have not expected to get here!')
                elif define_new_lower_limit:
                    t_spinup_limits[0] = copy.deepcopy(t_spinup)
                    if t_spinup >= t_spinup_limits[1]:
                        # this happens when the first guess was out of domain
                        was_errors[0] = False
                        t_spinup_limits[1] = (t_spinup_limits[0] +
                                              t_spinup_max_step_length)
                elif define_new_upper_limit:
                    t_spinup_limits[1] = copy.deepcopy(t_spinup)
                    if t_spinup <= t_spinup_limits[0]:
                        # this happens when the first guess was ice free
                        was_errors[1] = False
                        t_spinup_limits[0] = (t_spinup_limits[1] -
                                              t_spinup_max_step_length)

            return float(tmp_mismatch), float(t_spinup)

        # first guess
        new_mismatch, new_t_spinup = get_mismatch(first_guess_t_spinup)
        t_spinup_guess.append(new_t_spinup)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_spinup_guess, mismatch

        # second (arbitrary) guess is given depending on the outcome of first
        # guess, when mismatch is 100% t_spinup is changed for
        # t_spinup_max_step_length, but at least the second guess is 0.2 °C away
        # from the first guess
        step = np.sign(mismatch[-1]) * max(np.abs(mismatch[-1]) *
                                           t_spinup_max_step_length / 100,
                                           0.2)
        new_mismatch, new_t_spinup = get_mismatch(t_spinup_guess[0] + step)
        t_spinup_guess.append(new_t_spinup)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_spinup_guess, mismatch

        # Now start with splin fit for guessing
        while len(t_spinup_guess) < maxiter:
            # get next guess from splin (fit partial linear function to previously
            # calculated (mismatch, t_spinup) pairs and get t_spinup value where
            # mismatch=0 from this fitted curve)
            sort_index = np.argsort(np.array(mismatch))
            tck = interpolate.splrep(np.array(mismatch)[sort_index],
                                     np.array(t_spinup_guess)[sort_index],
                                     k=1)
            # here we catch interpolation errors (two different t_spinup with
            # same mismatch), could happen if one t_spinup was close to a newly
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
            new_mismatch, new_t_spinup = get_mismatch(float(interpolate.splev(0,
                                                                            tck)
                                                          ))
            t_spinup_guess.append(new_t_spinup)
            mismatch.append(new_mismatch)

            if abs(mismatch[-1]) < precision_percent:
                return t_spinup_guess, mismatch

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
        spinup_period_initial = min(target_yr - spinup_start_yr,
                                    target_yr - yr_min)
    else:
        spinup_period_initial = min(spinup_period, target_yr - yr_min)
    if spinup_period_initial <= min_spinup_period:
        spinup_periods_to_try = [min_spinup_period]
    else:
        # try out a maximum of three different spinup_periods
        spinup_periods_to_try = [spinup_period_initial,
                                 int((spinup_period_initial +
                                      min_spinup_period) / 2),
                                 min_spinup_period]
    # after defining the initial spinup period we can define the year for the
    # fixed_geometry_spinup
    if add_fixed_geometry_spinup:
        fixed_geometry_spinup_yr = target_yr - spinup_period_initial
    else:
        fixed_geometry_spinup_yr = None

    # check if the user provided an mb_model_spinup, otherwise we must define a
    # new one each iteration
    provided_mb_model_spinup = False
    if mb_model_spinup is not None:
        provided_mb_model_spinup = True

    for spinup_period in spinup_periods_to_try:
        yr_spinup = target_yr - spinup_period

        if not provided_mb_model_spinup:
            # define spinup MassBalance
            # spinup is running for 'target_yr - yr_spinup' years, using a
            # ConstantMassBalance
            y0_spinup = (yr_spinup + target_yr) / 2
            halfsize_spinup = target_yr - y0_spinup
            mb_model_spinup = MultipleFlowlineMassBalance(
                gdir, mb_model_class=ConstantMassBalance,
                filename='climate_historical',
                input_filesuffix=climate_input_filesuffix, y0=y0_spinup,
                halfsize=halfsize_spinup)

        # try to conduct minimisation, if an error occurred try shorter spinup
        # period
        try:
            final_t_spinup_guess, final_mismatch = minimise_with_spline_fit(c_fun)
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
                    if return_t_spinup_best:
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
                            float(final_t_spinup_guess[-1]))
    gdir.add_to_diagnostics('dynamic_spinup_target_year',
                            int(target_yr))
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
        with warnings.catch_warnings():
            # For operational runs we ignore the warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_dynamic_spinup_end[-1].run_until_and_store(
                target_yr,
                geom_path=geom_path,
                diag_path=diag_path,
                fl_diag_path=fl_diag_path, )

    if return_t_spinup_best:
        return model_dynamic_spinup_end[-1], final_t_spinup_guess[-1]
    else:
        return model_dynamic_spinup_end[-1]


def define_new_melt_f_in_gdir(gdir, new_melt_f):
    """
    Helper function to define a new melt_f in a gdir. Is used inside the run
    functions of the dynamic melt_f calibration.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to change the melt_f
    new_melt_f : float
        the new melt_f to save in the gdir

    Returns
    -------

    """
    try:
        df = gdir.read_json('mb_calib')
    except FileNotFoundError:
        raise InvalidWorkflowError('`mb_calib.json` does not exist in gdir. '
                                   'You first need to calibrate the whole '
                                   'MassBalanceModel before changing melt_f '
                                   'alone!')
    df['melt_f'] = new_melt_f
    gdir.write_json(df, 'mb_calib')


def dynamic_melt_f_run_with_dynamic_spinup(
        gdir, melt_f, yr0_ref_mb, yr1_ref_mb, fls_init, ys, ye,
        output_filesuffix='', evolution_model=None,
        mb_model_historical=None, mb_model_spinup=None,
        minimise_for='area', climate_input_filesuffix='', spinup_period=20,
        min_spinup_period=10, target_yr=None, precision_percent=1,
        precision_absolute=1, min_ice_thickness=None,
        first_guess_t_spinup=-2, t_spinup_max_step_length=2, maxiter=30,
        store_model_geometry=True, store_fl_diagnostics=None,
        local_variables=None, set_local_variables=False, do_inversion=True,
        spinup_start_yr_max=None, add_fixed_geometry_spinup=True,
        **kwargs):
    """
    This function is one option for a 'run_function' for the
    'run_dynamic_melt_f_calibration' function (the corresponding
    'fallback_function' is
    'dynamic_melt_f_run_with_dynamic_spinup_fallback'). This
    function defines a new melt_f in the glacier directory and conducts an
    inversion calibrating A to match '_vol_m3_ref' with this new melt_f
    ('calibrate_inversion_from_consensus'). Afterwards a dynamic spinup is
    conducted to match 'minimise_for' (for more info look at docstring of
    'run_dynamic_spinup'). And in the end the geodetic mass balance of the
    current run is calculated (between the period [yr0_ref_mb, yr1_ref_mb]) and
    returned.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    melt_f : float
        the melt_f used for this run
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
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a MonthlyTIModel model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    minimise_for : str
        The variable we want to match at target_yr. Options are 'area' or
        'volume'.
        Default is 'area'.
    climate_input_filesuffix : str
        filesuffix for the input climate file
        Default is ''
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "target_yr - spinup_period". Minimum allowed value is
        defined with 'min_spinup_period'. If the provided climate data starts
        at year later than (target_yr - spinup_period) the spinup_period is set
        to (target_yr - yr_climate_start). Caution if spinup_start_yr is set
        the spinup_period is ignored.
        Default is 20
    min_spinup_period : int
        If the dynamic spinup function fails with the initial 'spinup_period'
        a shorter period is tried. Here you can define the minimum period to
        try.
        Default is 10
    target_yr : int or None
        The target year at which we want to match area or volume.
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
        Default is cfg.PARAMS['dynamic_spinup_min_ice_thick'].
    first_guess_t_spinup : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_spinup_max_step_length : float
        Defines the maximums allowed change of t_spinup between two iterations.
        Is needed to avoid to large changes.
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
        sets the first_guess_t_spinup with the key 't_spinup' AND sets the
        current glacier volume with the key 'vol_m3_ref' which is later used in
        the calibration during inversion.
        Default is False
    do_inversion : bool
        If True a complete inversion is conducted using the provided melt_f
        before the actual calibration run.
        Default is False
    spinup_start_yr_max : int or None
        Possibility to provide a maximum year where the dynamic spinup must
        start from at least. If set, this overrides the min_spinup_period if
        target_yr - spinup_start_yr_max > min_spinup_period. If None it is set
        to yr0_ref_mb.
        Default is None
    add_fixed_geometry_spinup : bool
        If True and the original spinup_period of the dynamical spinup must be
        shortened (due to ice-free or out-of-boundary error) a
        fixed-geometry-spinup is added at the beginning so that the resulting
        model run always starts from ys.
        Default is True
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`, float
        The final model after the run and the calculated geodetic mass balance
    """

    evolution_model = decide_evolution_model(evolution_model)

    if not isinstance(local_variables, dict):
        raise ValueError('You must provide a dict for local_variables!')

    from oggm.workflow import calibrate_inversion_from_consensus

    if set_local_variables:
        # clear the provided dictionary and set the first elements
        local_variables.clear()
        local_variables['t_spinup'] = [first_guess_t_spinup]
        # ATTENTION: it is assumed that the flowlines in gdir have the volume
        # we want to match during calibrate_inversion_from_consensus when we
        # set_local_variables
        fls_ref = gdir.read_pickle('model_flowlines')
        local_variables['vol_m3_ref'] = np.sum([f.volume_m3 for f in fls_ref])

        # we are done with preparing the local_variables for the upcoming iterations
        return None

    if target_yr is None:
        target_yr = gdir.rgi_date + 1  # + 1 converted to hydro years
    if min_spinup_period > target_yr - ys:
        log.info('The target year is closer to ys as the minimum spinup '
                 'period -> therefore the minimum spinup period is '
                 'adapted and it is the only period which is tried by the '
                 'dynamic spinup function!')
        min_spinup_period = target_yr - ys
        spinup_period = target_yr - ys

    if spinup_start_yr_max is None:
        spinup_start_yr_max = yr0_ref_mb

    if spinup_start_yr_max > yr0_ref_mb:
        log.info('The provided maximum start year is larger then the '
                 'start year of the geodetic period, therefore it will be '
                 'set to the start year of the geodetic period!')
        spinup_start_yr_max = yr0_ref_mb

    # check that inversion is only possible without providing own fls
    if do_inversion:
        if not np.all([np.all(getattr(fl_prov, 'surface_h') ==
                              getattr(fl_orig, 'surface_h')) and
                       np.all(getattr(fl_prov, 'bed_h') ==
                              getattr(fl_orig, 'bed_h'))
                       for fl_prov, fl_orig in
                       zip(fls_init, gdir.read_pickle('model_flowlines'))]):
            raise InvalidWorkflowError('If you want to perform a dynamic '
                                       'melt_f calibration including an '
                                       'inversion, it is not possible to '
                                       'provide your own flowlines! (fls_init '
                                       'should be None or '
                                       'the original model_flowlines)')

    # Here we start with the actual model run
    if melt_f == gdir.read_json('mb_calib')['melt_f']:
        # we do not need to define a new melt_f or do an inversion
        do_inversion = False
    else:
        define_new_melt_f_in_gdir(gdir, melt_f)

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
    model_flowline_filesuffix = '_dyn_melt_f_calib'
    init_present_time_glacier(gdir, filesuffix=model_flowline_filesuffix,
                              add_to_log_file=False)

    # Now do a dynamic spinup to match area
    # do not ignore errors in dynamic spinup, so all 'bad'/intermediate files
    # are deleted in run_dynamic_spinup function
    try:
        model, last_best_t_spinup = run_dynamic_spinup(
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
            spinup_start_yr_max=spinup_start_yr_max,
            min_spinup_period=min_spinup_period, target_yr=target_yr,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            t_spinup_max_step_length=t_spinup_max_step_length,
            maxiter=maxiter,
            minimise_for=minimise_for,
            first_guess_t_spinup=local_variables['t_spinup'][-1],
            output_filesuffix=output_filesuffix,
            store_model_evolution=True, ignore_errors=False,
            return_t_spinup_best=True, ye=ye,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            model_flowline_filesuffix=model_flowline_filesuffix,
            add_fixed_geometry_spinup=add_fixed_geometry_spinup,
            **kwargs)
        # save the temperature bias which was successful in the last iteration
        # as we expect we are not so far away in the next iteration (only
        # needed for optimisation, potentially need less iterations in
        # run_dynamic_spinup)
        local_variables['t_spinup'].append(last_best_t_spinup)
    except RuntimeError as e:
        raise RuntimeError(f'Dynamic spinup raised error! (Message: {e})')

    # calculate dmdtda from previous simulation here
    with utils.DisableLogger():
        ds = utils.compile_run_output(gdir, input_filesuffix=output_filesuffix,
                                      path=False)
    dmdtda_mdl = ((ds.volume.loc[yr1_ref_mb].values[0] -
                   ds.volume.loc[yr0_ref_mb].values[0]) /
                  gdir.rgi_area_m2 /
                  (yr1_ref_mb - yr0_ref_mb) *
                  cfg.PARAMS['ice_density'])

    return model, dmdtda_mdl


def dynamic_melt_f_run_with_dynamic_spinup_fallback(
        gdir, melt_f, fls_init, ys, ye, local_variables, output_filesuffix='',
        evolution_model=None, minimise_for='area',
        mb_model_historical=None, mb_model_spinup=None,
        climate_input_filesuffix='', spinup_period=20, min_spinup_period=10,
        target_yr=None, precision_percent=1,
        precision_absolute=1, min_ice_thickness=None,
        first_guess_t_spinup=-2, t_spinup_max_step_length=2, maxiter=30,
        store_model_geometry=True, store_fl_diagnostics=None,
        do_inversion=True, spinup_start_yr_max=None,
        add_fixed_geometry_spinup=True, **kwargs):
    """
    This is the fallback function corresponding to the function
    'dynamic_melt_f_run_with_dynamic_spinup', which are provided
    to 'run_dynamic_melt_f_calibration'. It is used if the run_function fails and
    if 'ignore_error == True' in 'run_dynamic_melt_f_calibration'. First it resets
    melt_f of gdir. Afterwards it tries to conduct a dynamic spinup. If this
    also fails the last thing is to just do a run without a dynamic spinup
    (only a fixed geometry spinup).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    melt_f : float
        the melt_f used for this run
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
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    mb_model_historical : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the historical run. Default
        is to use a MonthlyTIModel model  together with the provided
        parameter climate_input_filesuffix.
    mb_model_spinup : :py:class:`core.MassBalanceModel`
        User-povided MassBalanceModel instance for the spinup before the
        historical run. Default is to use a ConstantMassBalance model together
        with the provided parameter climate_input_filesuffix and during the
        period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
    minimise_for : str
        The variable we want to match at target_yr. Options are 'area' or
        'volume'.
        Default is 'area'.
    climate_input_filesuffix : str
        filesuffix for the input climate file
        Default is ''
    spinup_period : int
        The period how long the spinup should run. Start date of historical run
        is defined "target_yr - spinup_period". Minimum allowed value is
        defined with 'min_spinup_period'. If the provided climate data starts
        at year later than (target_yr - spinup_period) the spinup_period is set
        to (target_yr - yr_climate_start). Caution if spinup_start_yr is set
        the spinup_period is ignored.
        Default is 20
    min_spinup_period : int
        If the dynamic spinup function fails with the initial 'spinup_period'
        a shorter period is tried. Here you can define the minimum period to
        try.
        Default is 10
    target_yr : int or None
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
        Default is cfg.PARAMS['dynamic_spinup_min_ice_thick'].
    first_guess_t_spinup : float
        The initial guess for the temperature bias for the spinup
        MassBalanceModel in °C.
        Default is -2.
    t_spinup_max_step_length : float
        Defines the maximums allowed change of t_spinup between two iterations. Is
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
        If True a complete inversion is conducted using the provided melt_f
        before the actual fallback run.
        Default is False
    spinup_start_yr_max : int or None
        Possibility to provide a maximum year where the dynamic spinup must
        start from at least. If set, this overrides the min_spinup_period if
        target_yr - spinup_start_yr_max > min_spinup_period.
        Default is None
    add_fixed_geometry_spinup : bool
        If True and the original spinup_period of the dynamical spinup must be
        shortened (due to ice-free or out-of-boundary error) a
        fixed-geometry-spinup is added at the beginning so that the resulting
        model run always starts from ys.
        Default is True
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final model after the run.
    """
    from oggm.workflow import calibrate_inversion_from_consensus

    evolution_model = decide_evolution_model(evolution_model)

    if local_variables is None:
        raise RuntimeError('Need the volume to do'
                           'calibrate_inversion_from_consensus provided in '
                           'local_variables!')

    # revert gdir to original state if necessary
    if melt_f != gdir.read_json('mb_calib')['melt_f']:
        define_new_melt_f_in_gdir(gdir, melt_f)
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
                                   'model_flowlines_dyn_melt_f_calib.pkl')):
        os.remove(os.path.join(gdir.dir,
                               'model_flowlines_dyn_melt_f_calib.pkl'))

    if target_yr is None:
        target_yr = gdir.rgi_date + 1  # + 1 converted to hydro years
    if min_spinup_period > target_yr - ys:
        log.info('The RGI year is closer to ys as the minimum spinup '
                 'period -> therefore the minimum spinup period is '
                 'adapted and it is the only period which is tried by the '
                 'dynamic spinup function!')
        min_spinup_period = target_yr - ys
        spinup_period = target_yr - ys

    yr_clim_min = gdir.get_climate_info()['baseline_yr_0']
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
            spinup_start_yr_max=spinup_start_yr_max,
            target_yr=target_yr,
            minimise_for=minimise_for,
            precision_percent=precision_percent,
            precision_absolute=precision_absolute,
            min_ice_thickness=min_ice_thickness,
            first_guess_t_spinup=first_guess_t_spinup,
            t_spinup_max_step_length=t_spinup_max_step_length,
            maxiter=maxiter,
            output_filesuffix=output_filesuffix,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            ignore_errors=False,
            ye=ye,
            add_fixed_geometry_spinup=add_fixed_geometry_spinup,
            **kwargs)

        gdir.add_to_diagnostics('used_spinup_option', 'dynamic spinup only')

    except RuntimeError:
        log.warning('No dynamic spinup could be conducted by using the '
                    f'original melt factor ({melt_f}). Therefore the last '
                    'try is to conduct a run until ye without a dynamic '
                    'spinup.')

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

        # try to make a fixed geometry spinup
        model_end = run_from_climate_data(
            gdir,
            continue_on_error=False,  # force to raise an error in @entity_task
            add_to_log_file=False,
            min_ys=yr_clim_min, ye=ye,
            output_filesuffix=output_filesuffix,
            climate_input_filesuffix=climate_input_filesuffix,
            store_model_geometry=store_model_geometry,
            store_fl_diagnostics=store_fl_diagnostics,
            init_model_fls=fls_init, evolution_model=evolution_model,
            fixed_geometry_spinup_yr=ys)

        gdir.add_to_diagnostics('used_spinup_option', 'fixed geometry spinup')

    return model_end


def dynamic_melt_f_run(
        gdir, melt_f, yr0_ref_mb, yr1_ref_mb, fls_init, ys, ye,
        output_filesuffix='', evolution_model=None,
        local_variables=None, set_local_variables=False, target_yr=None,
        **kwargs):
    """
    This function is one option for a 'run_function' for the
    'run_dynamic_melt_f_calibration' function (the corresponding
    'fallback_function' is 'dynamic_melt_f_run_fallback'). It is meant to
    define a new melt_f in the given gdir and conduct a
    'run_from_climate_data' run between ys and ye and return the geodetic mass
    balance (units: kg m-2 yr-1) of the period yr0_ref_mb and yr1_ref_mb.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    melt_f : float
        the melt_f used for this run
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
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    local_variables : None
        Not needed in this function, just here to match with the function
        call in run_dynamic_melt_f_calibration.
    set_local_variables : bool
        Not needed in this function. Only here to be confirm with the use of
        this function in 'run_dynamic_melt_f_calibration'.
    target_yr : int or None
        The target year for a potential dynamic spinup (not needed here).
        Default is None
    kwargs : dict
        kwargs to pass to the evolution_model instance

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`, float
        The final model after the run and the calculated geodetic mass balance
    """

    evolution_model = decide_evolution_model(evolution_model)

    if set_local_variables:
        # No local variables needed in this function
        return None

    # Here we start with the actual model run
    define_new_melt_f_in_gdir(gdir, melt_f)

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


def dynamic_melt_f_run_fallback(
        gdir, melt_f, fls_init, ys, ye, local_variables, output_filesuffix='',
        evolution_model=None, target_yr=None, **kwargs):
    """
    This is the fallback function corresponding to the function
    'dynamic_melt_f_run', which are provided to
    'run_dynamic_melt_f_calibration'. It is used if the run_function fails and
    if 'ignore_error=True' in 'run_dynamic_melt_f_calibration'. It sets
    melt_f and conduct a run_from_climate_data run from ys to ye.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    melt_f : float
        the melt_f used for this run
    fls_init : []
        List of flowlines to use to initialise the model
    ys : int
        start year of the run
    ye : int
        end year of the run
    local_variables : dict
        Not needed in this function, just here to match with the function
        call in run_dynamic_melt_f_calibration.
    output_filesuffix : str
        For the output file.
        Default is ''
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: cfg.PARAMS['evolution_model']
        Not all models work in all circumstances!
    target_yr : int or None
        The target year for a potential dynamic spinup (not needed here).
        Default is None
    kwargs : dict
        kwargs to pass to the evolution_model instance

     Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final model after the run.
    """

    evolution_model = decide_evolution_model(evolution_model)

    define_new_melt_f_in_gdir(gdir, melt_f)

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
def run_dynamic_melt_f_calibration(
        gdir, ref_dmdtda=None, err_ref_dmdtda=None, err_dmdtda_scaling_factor=1,
        ref_period='', melt_f_min=None,
        melt_f_max=None, melt_f_max_step_length_minimum=0.1, maxiter=20,
        ignore_errors=False, output_filesuffix='_dynamic_melt_f',
        ys=None, ye=None, target_yr=None,
        run_function=dynamic_melt_f_run_with_dynamic_spinup,
        kwargs_run_function=None,
        fallback_function=dynamic_melt_f_run_with_dynamic_spinup_fallback,
        kwargs_fallback_function=None, init_model_filesuffix=None,
        init_model_yr=None, init_model_fls=None,
        first_guess_diagnostic_msg='dynamic spinup only'):
    """Calibrate melt_f to match a geodetic mass balance incorporating a
    dynamic model run.

    This task iteratively search for a melt_f to match a provided geodetic
    mass balance. How one model run looks like is defined in the 'run_function'.
    This function should take a new melt_f guess, conducts a dynamic run and
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
    melt_f_min : float or None
        Lower absolute limit for melt_f.
        Default is None (-> cfg.PARAMS['melt_f_min'])
    melt_f_max : float or None
        Upper absolute limit for melt_f.
        Default is None (-> cfg.PARAMS['melt_f_max'])
    melt_f_max_step_length_minimum : float
        Defines a minimum maximal change of melt_f between two iterations. The
        maximum step length is needed to avoid too large steps, which likely
        lead to an error.
        Default is 0.1
    maxiter : int
        Maximum number of minimisation iterations of minimising mismatch to
        dmdtda by changing melt_f. Each of this iterations conduct a complete
        run defined in the 'run_function'. If maxiter reached and
        'ignore_errors=False' an error is raised.
        Default is 20
    ignore_errors : bool
        If True and the 'run_function' with melt_f calibration is not working
        to match dmdtda inside the provided uncertainty fully, but their where
        some successful runs which improved the first guess, they are saved as
        part success, and if not a single run was successful the
        'fallback_function' is called.
        If False and the 'run_function' with melt_f calibration is not working
        fully an error is raised.
        Default is True
    output_filesuffix : str
        For the output file.
        Default is '_dynamic_melt_f'
    ys : int or None
        The start year of the conducted run. If None the first year of the
        provided climate file.
        Default is None
    ye : int or None
        The end year of the conducted run. If None the last year of the
        provided climate file.
        Default is None
    target_yr : int or None
        The target year for a potential dynamic spinup (see run_dynamic_spinup
        function for more info).
        If None, gdir.rgi_date + 1 is used (the default).
        Default is None
    run_function : function
        This function defines how a new defined melt_f is used to conduct the
        next model run. This function must contain the arguments 'gdir',
        'melt_f', 'yr0_ref_mb', 'yr1_ref_mb', 'fls_init', 'ys', 'ye' and
        'output_filesuffix'. Further this function must return the final model
        and the calculated geodetic mass balance dmdtda in kg m-2 yr-1.
    kwargs_run_function : None or dict
        Can provide additional keyword arguments to the run_function as a
        dictionary.
    fallback_function : function
        This is a fallback function if the calibration is not working using
        'run_function' it is called. This function must contain the arguments
        'gdir', 'melt_f', 'fls_init', 'ys', 'ye', 'local_variables' and
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
        default melt_f resulted in a successful 'run_function' run.
        Default is 'dynamic spinup only'

    Returns
    -------
    :py:class:`oggm.core.flowline.evolution_model`
        The final dynamically spined-up model. Type depends on the selected
        evolution_model.
    """
    # melt_f constraints
    if melt_f_min is None:
        melt_f_min = cfg.PARAMS['melt_f_min']
    if melt_f_max is None:
        melt_f_max = cfg.PARAMS['melt_f_max']

    if kwargs_run_function is None:
        kwargs_run_function = {}
    if kwargs_fallback_function is None:
        kwargs_fallback_function = {}

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
        sel = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period].iloc[0]
        # reference geodetic mass balance from Hugonnet 2021
        ref_dmdtda = float(sel['dmdtda'])
        # dmdtda: in meters water-equivalent per year -> we convert
        ref_dmdtda *= 1000  # kg m-2 yr-1
        # error of reference geodetic mass balance from Hugonnet 2021
        err_ref_dmdtda = float(sel['err_dmdtda'])
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

    clim_info = gdir.get_climate_info()

    if ye is None:
        # One adds 1 because the run ends at the end of the year
        ye = clim_info['baseline_yr_1'] + 1

    if ye < yr1_ref_mb:
        raise RuntimeError('The provided ye is smaller than the end year of '
                           'the given geodetic_mb_period!')

    if ys is None:
        ys = clim_info['baseline_yr_0']

    if ys > yr0_ref_mb:
        raise RuntimeError('The provided ys is larger than the start year of '
                           'the given geodetic_mb_period!')

    if target_yr is None:
        target_yr = gdir.rgi_date + 1  # + 1 converted to hydro years
    if target_yr < ys:
        if ignore_errors:
            log.info('The rgi year is smaller than the provided start year '
                     'ys -> setting the rgi year to ys to continue!')
            target_yr = ys
        else:
            raise RuntimeError('The rgi year is smaller than the provided '
                               'start year ys!')
    kwargs_run_function['target_yr'] = target_yr
    kwargs_fallback_function['target_yr'] = target_yr

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

    # save original melt_f for later to be able to recreate original gdir
    # (using the fallback function) if an error occurs
    melt_f_initial = gdir.read_json('mb_calib')['melt_f']

    # define maximum allowed change of melt_f between two iterations. Is needed
    # to avoid to large changes (=likely lead to an error). It is defined in a
    # way that in maxiter steps the further away limit can be reached
    melt_f_max_step_length = np.max(
        [np.max(np.abs(np.array([melt_f_min, melt_f_min]) - melt_f_initial)) /
         maxiter,
         melt_f_max_step_length_minimum])

    # only used to check performance of minimisation
    dynamic_melt_f_calibration_runs = [0]

    # this function is called if the actual dynamic melt_f calibration fails
    def fallback_run(melt_f, reset, best_mismatch=None, initial_mismatch=None,
                     only_first_guess=None):
        if reset:
            # unfortunately we could not conduct an error free run using the
            # provided run_function, so we us the fallback_function

            # this diagnostics should be overwritten inside the fallback_function
            gdir.add_to_diagnostics('used_spinup_option', 'fallback_function')

            model = fallback_function(gdir=gdir, melt_f=melt_f,
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
                                        'dynamic melt_f calibration (part '
                                        'success)')
            model, dmdtda_mdl = run_function(gdir=gdir, melt_f=melt_f,
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
            gdir.add_to_diagnostics('dmdtda_dynamic_calibration_error_scaling_factor',
                                    float(err_dmdtda_scaling_factor))
            gdir.add_to_diagnostics(
                'dmdtda_mismatch_dynamic_calibration',
                float(best_mismatch))
            gdir.add_to_diagnostics(
                'dmdtda_mismatch_with_initial_melt_f',
                float(initial_mismatch))
            gdir.add_to_diagnostics('melt_f_dynamic_calibration',
                                    float(melt_f))
            gdir.add_to_diagnostics('melt_f_before_dynamic_calibration',
                                    float(melt_f_initial))
            gdir.add_to_diagnostics('run_dynamic_melt_f_calibration_iterations',
                                    int(dynamic_melt_f_calibration_runs[-1]))

        return model

    # here we define the local variables which are used in the run_function,
    # for some run_functions this is useful to save parameters from a previous
    # run to be faster in the upcoming runs
    local_variables_run_function = {}
    run_function(gdir=gdir, melt_f=None, yr0_ref_mb=None, yr1_ref_mb=None,
                 fls_init=None, ys=None, ye=None,
                 local_variables=local_variables_run_function,
                 set_local_variables=True, **kwargs_run_function)

    # this is the actual model run which is executed each iteration in order to
    # minimise the mismatch of dmdtda of model and observation
    def model_run(melt_f):
        # to check performance of minimisation
        dynamic_melt_f_calibration_runs.append(
            dynamic_melt_f_calibration_runs[-1] + 1)

        model, dmdtda_mdl = run_function(gdir=gdir, melt_f=melt_f,
                                         yr0_ref_mb=yr0_ref_mb,
                                         yr1_ref_mb=yr1_ref_mb,
                                         fls_init=fls_init, ys=ys, ye=ye,
                                         output_filesuffix=output_filesuffix,
                                         local_variables=local_variables_run_function,
                                         **kwargs_run_function)
        return model, dmdtda_mdl

    def cost_fct(melt_f, model_dynamic_spinup_end):

        # actual model run
        model_dynamic_spinup, dmdtda_mdl = model_run(melt_f)

        # save final model for later
        model_dynamic_spinup_end.append(copy.deepcopy(model_dynamic_spinup))

        # calculate the mismatch of dmdtda
        try:
            cost = float(dmdtda_mdl - ref_dmdtda)
        except:
            t = 1
        return cost

    def init_cost_fun():
        model_dynamic_spinup_end = []

        def c_fun(melt_f):
            return cost_fct(melt_f, model_dynamic_spinup_end)

        return c_fun, model_dynamic_spinup_end

    # Here start with own spline minimisation algorithm
    def minimise_with_spline_fit(fct_to_minimise, melt_f_guess, mismatch):
        # defines limits of melt_f in accordance to maximal allowed change
        # between iterations
        melt_f_limits = [melt_f_initial - melt_f_max_step_length,
                         melt_f_initial + melt_f_max_step_length]

        # this two variables indicate that the limits were already adapted to
        # avoid an error
        was_min_error = False
        was_max_error = False
        was_errors = [was_min_error, was_max_error]

        def get_mismatch(melt_f):
            melt_f = copy.deepcopy(melt_f)
            # first check if the melt_f is inside limits
            if melt_f < melt_f_limits[0]:
                # was the smaller limit already executed, if not first do this
                if melt_f_limits[0] not in melt_f_guess:
                    melt_f = copy.deepcopy(melt_f_limits[0])
                else:
                    # smaller limit was already used, check if it was
                    # already newly defined with error
                    if was_errors[0]:
                        raise RuntimeError('Not able to minimise without '
                                           'raising an error at lower limit of '
                                           'melt_f!')
                    else:
                        # ok we set a new lower limit, consider also minimum
                        # limit
                        melt_f_limits[0] = max(melt_f_min,
                                               melt_f_limits[0] -
                                               melt_f_max_step_length)
            elif melt_f > melt_f_limits[1]:
                # was the larger limit already executed, if not first do this
                if melt_f_limits[1] not in melt_f_guess:
                    melt_f = copy.deepcopy(melt_f_limits[1])
                else:
                    # larger limit was already used, check if it was
                    # already newly defined with ice free glacier
                    if was_errors[1]:
                        raise RuntimeError('Not able to minimise without '
                                           'raising an error at upper limit of '
                                           'melt_f!')
                    else:
                        # ok we set a new upper limit, consider also maximum
                        # limit
                        melt_f_limits[1] = min(melt_f_max,
                                               melt_f_limits[1] +
                                               melt_f_max_step_length)

            # now clip melt_f with limits (to be sure)
            melt_f = np.clip(melt_f, melt_f_limits[0], melt_f_limits[1])
            if melt_f in melt_f_guess:
                raise RuntimeError('This melt_f was already tried. Probably '
                                   'we are at one of the max or min limit and '
                                   'still have no satisfactory mismatch '
                                   'found!')

            # if error during dynamic calibration this defines how much
            # melt_f is changed in the upcoming iterations to look for an
            # error free run
            melt_f_search_change = melt_f_max_step_length / 10
            # maximum number of changes to look for an error free run
            max_iterations = int(melt_f_max_step_length /
                                 melt_f_search_change)

            current_min_error = False
            current_max_error = False
            doing_first_guess = (len(mismatch) == 0)
            iteration = 0
            current_melt_f = copy.deepcopy(melt_f)

            # in this loop if an error at the limits is raised we go step by
            # step away from the limits until we are at the initial guess or we
            # found an error free run
            tmp_mismatch = None
            while ((current_min_error | current_max_error | iteration == 0) &
                   (iteration < max_iterations)):
                try:
                    tmp_mismatch = fct_to_minimise(melt_f)
                except RuntimeError as e:
                    # check if we are at the lower limit
                    if melt_f == melt_f_limits[0]:
                        # check if there was already an error at the lower limit
                        if was_errors[0]:
                            raise RuntimeError('Second time with error at '
                                               'lower limit of melt_f! '
                                               'Error message of model run: '
                                               f'{e}')
                        else:
                            was_errors[0] = True
                            current_min_error = True

                    # check if we are at the upperlimit
                    elif melt_f == melt_f_limits[1]:
                        # check if there was already an error at the lower limit
                        if was_errors[1]:
                            raise RuntimeError('Second time with error at '
                                               'upper limit of melt_f! '
                                               'Error message of model run: '
                                               f'{e}')
                        else:
                            was_errors[1] = True
                            current_max_error = True

                    if current_min_error:
                        # currently we searching for a new lower limit with no
                        # error
                        melt_f = np.round(melt_f + melt_f_search_change,
                                          decimals=1)
                    elif current_max_error:
                        # currently we searching for a new upper limit with no
                        # error
                        melt_f = np.round(melt_f - melt_f_search_change,
                                          decimals=1)

                    # if we end close to an already executed guess while
                    # searching for a new limit we quite
                    if np.isclose(melt_f, melt_f_guess).any():
                        raise RuntimeError('Not able to further minimise, '
                                           'return the best we have so far!'
                                           f'Error message: {e}')

                    if doing_first_guess:
                        # unfortunately first guess is not working
                        raise RuntimeError('Dynamic calibration is not working '
                                           'with first guess! Error '
                                           f'message: {e}')

                    if np.isclose(melt_f, current_melt_f):
                        # something unexpected happen so we end here
                        raise RuntimeError('Unexpected error not at the limits'
                                           f' of melt_f. Error Message: {e}')

                iteration += 1

            if iteration >= max_iterations:
                # ok we were not able to find an mismatch without error
                if current_min_error:
                    raise RuntimeError('Not able to find new lower limit for '
                                       'melt_f!')
                elif current_max_error:
                    raise RuntimeError('Not able to find new upper limit for '
                                       'melt_f!')
                else:
                    raise RuntimeError('Something unexpected happened during '
                                       'definition of new melt_f limits!')
            else:
                # if we found a new limit set it
                if current_min_error:
                    melt_f_limits[0] = copy.deepcopy(melt_f)
                elif current_max_error:
                    melt_f_limits[1] = copy.deepcopy(melt_f)

            if tmp_mismatch is None:
                raise RuntimeError('Not able to find a new mismatch for '
                                   'dmdtda!')

            return float(tmp_mismatch), float(melt_f)

        # first guess
        new_mismatch, new_melt_f = get_mismatch(melt_f_initial)
        melt_f_guess.append(new_melt_f)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < err_ref_dmdtda:
            return mismatch[-1], new_melt_f

        # second (arbitrary) guess is given depending on the outcome of first
        # guess, melt_f is changed for percent of mismatch relative to
        # err_ref_dmdtda times melt_f_max_step_length (if
        # mismatch = 2 * err_ref_dmdtda this corresponds to 100%; for 100% or
        # 150% the next step is (-1) * melt_f_max_step_length; if mismatch
        # -40%, next step is 0.4 * melt_f_max_step_length; but always at least
        # an absolute change of 0.02 is imposed to prevent too close guesses).
        # (-1) as if mismatch is negative we need a larger melt_f to get closer
        # to 0.
        step = (-1) * np.sign(mismatch[-1]) * \
            max((np.abs(mismatch[-1]) - err_ref_dmdtda) / err_ref_dmdtda *
                melt_f_max_step_length, 0.02)
        new_mismatch, new_melt_f = get_mismatch(melt_f_guess[0] + step)
        melt_f_guess.append(new_melt_f)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < err_ref_dmdtda:
            return mismatch[-1], new_melt_f

        # Now start with splin fit for guessing
        while len(melt_f_guess) < maxiter:
            # get next guess from splin (fit partial linear function to
            # previously calculated (mismatch, melt_f) pairs and get melt_f
            # value where mismatch=0 from this fitted curve)
            sort_index = np.argsort(np.array(mismatch))
            tck = interpolate.splrep(np.array(mismatch)[sort_index],
                                     np.array(melt_f_guess)[sort_index],
                                     k=1)
            # here we catch interpolation errors (two different melt_f with
            # same mismatch), could happen if one melt_f was close to a newly
            # defined limit
            if np.isnan(tck[1]).any():
                if was_errors[0]:
                    raise RuntimeError('Second time with error at lower '
                                       'limit of melt_f! (nan in splin fit)')
                elif was_errors[1]:
                    raise RuntimeError('Second time with error at upper '
                                       'limit of melt_f! (nan in splin fit)')
                else:
                    raise RuntimeError('Not able to minimise! Problem is '
                                       'unknown. (nan in splin fit)')
            new_mismatch, new_melt_f = get_mismatch(
                float(interpolate.splev(0, tck)))
            melt_f_guess.append(new_melt_f)
            mismatch.append(new_mismatch)

            if abs(mismatch[-1]) < err_ref_dmdtda:
                return mismatch[-1], new_melt_f

        # Ok when we end here the spinup could not find satisfying match after
        # maxiter(ations)
        raise RuntimeError(f'Could not find mismatch smaller '
                           f'{err_ref_dmdtda} kg m-2 yr-1 (only '
                           f'{np.min(np.abs(mismatch))} kg m-2 yr-1) in '
                           f'{maxiter} Iterations!')

    # wrapper to get values for intermediate (mismatch, melt_f) guesses if an
    # error is raised
    def init_minimiser():
        melt_f_guess = []
        mismatch = []

        def minimiser(fct_to_minimise):
            return minimise_with_spline_fit(fct_to_minimise, melt_f_guess,
                                            mismatch)

        return minimiser, melt_f_guess, mismatch

    # define function for the actual minimisation
    c_fun, models_dynamic_spinup_end = init_cost_fun()

    # define minimiser
    minimise_given_fct, melt_f_guesses, mismatch_dmdtda = init_minimiser()

    try:
        final_mismatch, final_melt_f = minimise_given_fct(c_fun)
    except RuntimeError as e:
        # something happened during minimisation, if there where some
        # successful runs we return the one with the best mismatch, otherwise
        # we conduct just a run with no dynamic spinup
        if len(mismatch_dmdtda) == 0:
            # we conducted no successful run, so run without dynamic spinup
            if ignore_errors:
                log.info('Dynamic melt_f calibration not successful. '
                         f'Error message: {e}')
                model_return = fallback_run(melt_f=melt_f_initial,
                                            reset=True)
                return model_return
            else:
                raise RuntimeError('Dynamic melt_f calibration was not '
                                   f'successful! Error Message: {e}')
        else:
            if ignore_errors:
                log.info('Dynamic melt_f calibration not successful. Error '
                         f'message: {e}')

                # there where some successful runs so we return the one with the
                # smallest mismatch of dmdtda
                min_mismatch_index = np.argmin(np.abs(mismatch_dmdtda))
                melt_f_best = np.array(melt_f_guesses)[min_mismatch_index]

                # check if the first guess was the best guess
                only_first_guess = False
                if min_mismatch_index == 1:
                    only_first_guess = True

                model_return = fallback_run(
                    melt_f=melt_f_best, reset=False,
                    best_mismatch=np.array(mismatch_dmdtda)[min_mismatch_index],
                    initial_mismatch=mismatch_dmdtda[0],
                    only_first_guess=only_first_guess)

                return model_return
            else:
                raise RuntimeError('Dynamic melt_f calibration not successful. '
                                   f'Error message: {e}')

    # check that new melt_f is correctly saved in gdir
    assert final_melt_f == gdir.read_json('mb_calib')['melt_f']

    # hurray, dynamic melt_f calibration successful
    gdir.add_to_diagnostics('used_spinup_option',
                            'dynamic melt_f calibration (full success)')
    gdir.add_to_diagnostics('dmdtda_mismatch_dynamic_calibration_reference',
                            float(ref_dmdtda))
    gdir.add_to_diagnostics('dmdtda_dynamic_calibration_given_error',
                            float(err_ref_dmdtda))
    gdir.add_to_diagnostics('dmdtda_dynamic_calibration_error_scaling_factor',
                            float(err_dmdtda_scaling_factor))
    gdir.add_to_diagnostics('dmdtda_mismatch_dynamic_calibration',
                            float(final_mismatch))
    gdir.add_to_diagnostics('dmdtda_mismatch_with_initial_melt_f',
                            float(mismatch_dmdtda[0]))
    gdir.add_to_diagnostics('melt_f_dynamic_calibration', float(final_melt_f))
    gdir.add_to_diagnostics('melt_f_before_dynamic_calibration',
                            float(melt_f_initial))
    gdir.add_to_diagnostics('run_dynamic_melt_f_calibration_iterations',
                            int(dynamic_melt_f_calibration_runs[-1]))

    log.info(f'Dynamic melt_f calibration worked for {gdir.rgi_id}!')

    return models_dynamic_spinup_end[-1]
