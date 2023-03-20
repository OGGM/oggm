import numpy as np

from oggm.core.flowline import (
    flowline_model_run,
    FileModel,
)
from oggm.core.massbalance import (
    ConstantMassBalance,
    MassBalanceModel,
)
from oggm import entity_task

# Module logger
import logging
log = logging.getLogger(__name__)


class BiasedConstantMassBalance(MassBalanceModel):
    """Time-dependant Temp and PRCP delta ConstantMassBalance model"""

    def __init__(
        self,
        gdir,
        temp_bias_ts=None,
        prcp_fac_ts=None,
        bias=0,
        y0=None,
        halfsize=15,
        filename="climate_historical",
        input_filesuffix="",
        **kwargs
    ):
        """Initialize

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        temp_bias_ts : pandas DataFrame
            the temperature bias timeseries (in °C) (index: time as years)
        prcp_fac_ts : pandas DataFrame
            the precipitaion bias timeseries (in % change, positive or negative)
            (index: time as years)
        bias : float, optional
            set to the alternative value of the annual bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
        y0 : int
            the year at the center of the period of interest. Has to be set!
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        super(BiasedConstantMassBalance, self).__init__()

        self.mbmod = ConstantMassBalance(
            gdir,
            bias=bias,
            y0=y0,
            halfsize=halfsize,
            filename=filename,
            input_filesuffix=input_filesuffix,
            **kwargs
        )

        self.valid_bounds = self.mbmod.valid_bounds
        self.hemisphere = gdir.hemisphere
        self.is_year_valid = self.mbmod.is_year_valid

        # Set ys and ye
        self.ys = int(temp_bias_ts.index[0])
        self.ye = int(temp_bias_ts.index[-1])

        if prcp_fac_ts is None:
            prcp_fac_ts = temp_bias_ts * 0

        self.prcp_fac_ts = self.mbmod.prcp_fac + prcp_fac_ts
        self.temp_bias_ts = self.mbmod.temp_bias + temp_bias_ts

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def _check_bias(self, year):
        t = np.asarray(self.temp_bias_ts.loc[int(year)])
        if np.any(t != self.temp_bias):
            self.temp_bias = t
        p = np.asarray(self.prcp_fac_ts.loc[int(year)])
        if np.any(p != self.prcp_fac):
            self.prcp_fac = p

    def get_monthly_mb(self, heights, year=None, **kwargs):
        self._check_bias(year)
        return self.mbmod.get_monthly_mb(heights, year=year, **kwargs)

    def get_annual_mb(self, heights, year=None, **kwargs):
        self._check_bias(year)
        return self.mbmod.get_annual_mb(heights, year=year, **kwargs)


@entity_task(log)
def run_constant_climate_with_bias(
    gdir,
    temp_bias_ts=None,
    prcp_fac_ts=None,
    ys=None,
    ye=None,
    y0=2014,
    halfsize=5,
    climate_filename="climate_historical",
    climate_input_filesuffix="",
    output_filesuffix="",
    init_model_fls=None,
    init_model_filesuffix=None,
    init_model_yr=None,
    bias=0,
    **kwargs
):
    """Runs a glacier with temperature and precipitation correction timeseries.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    temp_bias_ts : pandas DataFrame
        the temperature bias timeseries (in °C) (index: time as years)
    prcp_fac_ts : pandas DataFrame
        the precipitaion bias timeseries (in % change, positive or negative)
        (index: time as years)
    y0 : int
        central year of the constant climate period
    halfsize : int
        half-size of the constant climate period
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or 'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year available
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is zero usually anyways.
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath("model_geometry", filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        # Avoid issues here
        if init_model_yr > fmod.y0:
            fmod.run_until(init_model_yr)
        else:
            fmod.run_until(fmod.y0)

        init_model_fls = fmod.fls

    # Final crop
    mb = BiasedConstantMassBalance(
        gdir,
        temp_bias_ts=temp_bias_ts,
        prcp_fac_ts=prcp_fac_ts,
        y0=y0,
        bias=bias,
        halfsize=halfsize,
        filename=climate_filename,
        input_filesuffix=climate_input_filesuffix,
    )

    # Decide from climate
    if ye is None:
        ye = mb.ye
    if ys is None:
        ys = mb.ys

    return flowline_model_run(
        gdir,
        output_filesuffix=output_filesuffix,
        mb_model=mb,
        ys=ys,
        ye=ye,
        init_model_fls=init_model_fls,
        **kwargs
    )
