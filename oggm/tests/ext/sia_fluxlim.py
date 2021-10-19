""" THIS MODULE IS NOT PART OF OGGM.

It is shipped here for convenience, but the original code and it's license
(GPLv3) are found at https://github.com/alexjarosch/sia-fluxlim.

This module contains a 1D implementation of the numerical scheme
presented in Jarosch et al. 2013 (doi:10.5194/tc-7-229-2013).
"""

import warnings
import numpy as np
from oggm.core.flowline import FlowlineModel
from oggm.cfg import SEC_IN_DAY, G
from oggm.exceptions import InvalidParamsError
from oggm import utils


class MUSCLSuperBeeModel(FlowlineModel):
    """This model is based on Jarosch et al. 2013 (doi:10.5194/tc-7-229-2013)

       The equation references in the comments refer to the paper for clarity
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=None,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31*SEC_IN_DAY,
                 inplace=False, **kwargs):
        """ Instantiate.

        Parameters
        ----------
        """

        if len(flowlines) > 1:
            raise ValueError('MUSCL SuperBee model does not work with '
                             'tributaries.')

        super(MUSCLSuperBeeModel, self).__init__(flowlines, mb_model=mb_model,
                                                 y0=y0, glen_a=glen_a, fs=fs,
                                                 inplace=inplace, **kwargs)
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt

    def phi(self, r):
        """ a definition of the limiting scheme to use"""
        # minmod limiter Eq. 28
        # val_phi = numpy.maximum(0,numpy.minimum(1,r))

        # superbee limiter Eq. 29
        val_phi = np.maximum(0, np.minimum(2*r, 1), np.minimum(r, 2))

        return val_phi

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt
        dt = utils.clip_scalar(dt, min_dt, self.max_dt)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fl = self.fls[0]
            dx = fl.dx_meter

            # Switch to the notation from the MUSCL_1D example
            # This is useful to ensure that the MUSCL-SuperBee code
            # is working as it has been benchmarked many times

            # mass balance
            m_dot = self.get_mb(fl.surface_h, self.yr, fl_id=id(fl))
            # get in the surface elevation
            S = fl.surface_h
            # get the bed
            B = fl.bed_h
            # define Glen's law here
            N = self.glen_n
            # this is the correct Gamma !!
            Gamma = 2.*self.glen_a*(self.rho*G)**N / (N+2.)
            # time stepping
            c_stab = 0.165

            # define the finite difference indices
            k = np.arange(0, fl.nx)
            kp = np.hstack([np.arange(1, fl.nx), fl.nx-1])
            kpp = np.hstack([np.arange(2, fl.nx), fl.nx-1, fl.nx-1])
            km = np.hstack([0, np.arange(0, fl.nx-1)])
            kmm = np.hstack([0, 0, np.arange(0, fl.nx-2)])

            # I'm gonna introduce another level of adaptive time stepping here,
            # which is probably not necessary. However I keep it to be
            # consistent with my benchmarked and tested code.
            # If the OGGM time stepping is correctly working, this loop
            # should never run more than once
            # Fabi: actually no, it is your job to choose the right time step
            # but you can let OGGM decide whether a new time step is needed
            # or not -> to be meliorated one day
            stab_t = 0.
            while stab_t < dt:
                H = S - B

                # MUSCL scheme up. "up" denotes here the k+1/2 flux boundary
                r_up_m = (H[k]-H[km])/(H[kp]-H[k])  # Eq. 27
                H_up_m = H[k] + 0.5 * self.phi(r_up_m)*(H[kp]-H[k])  # Eq. 23
                # Eq. 27, k+1 is used instead of k
                r_up_p = (H[kp]-H[k])/(H[kpp]-H[kp])
                # Eq. 24
                H_up_p = H[kp] - 0.5 * self.phi(r_up_p)*(H[kpp]-H[kp])

                # surface slope gradient
                s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((N-1.)/2.)
                # like Eq. 30, now using Eq. 23 instead of Eq. 24
                D_up_m = Gamma * H_up_m**(N+2.) * s_grad_up
                D_up_p = Gamma * H_up_p**(N+2.) * s_grad_up  # Eq. 30

                # Eq. 31
                D_up_min = np.minimum(D_up_m, D_up_p)
                # Eq. 32
                D_up_max = np.maximum(D_up_m, D_up_p)
                D_up = np.zeros(fl.nx)

                # Eq. 33
                cond = (S[kp] <= S[k]) & (H_up_m <= H_up_p)
                D_up[cond] = D_up_min[cond]
                cond = (S[kp] <= S[k]) & (H_up_m > H_up_p)
                D_up[cond] = D_up_max[cond]
                cond = (S[kp] > S[k]) & (H_up_m <= H_up_p)
                D_up[cond] = D_up_max[cond]
                cond = (S[kp] > S[k]) & (H_up_m > H_up_p)
                D_up[cond] = D_up_min[cond]

                # MUSCL scheme down. "down" denotes the k-1/2 flux boundary
                r_dn_m = (H[km]-H[kmm])/(H[k]-H[km])
                H_dn_m = H[km] + 0.5 * self.phi(r_dn_m)*(H[k]-H[km])
                r_dn_p = (H[k]-H[km])/(H[kp]-H[k])
                H_dn_p = H[k] - 0.5 * self.phi(r_dn_p)*(H[kp]-H[k])

                # calculate the slope gradient
                s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((N-1.)/2.)
                D_dn_m = Gamma * H_dn_m**(N+2.) * s_grad_dn
                D_dn_p = Gamma * H_dn_p**(N+2.) * s_grad_dn

                D_dn_min = np.minimum(D_dn_m, D_dn_p)
                D_dn_max = np.maximum(D_dn_m, D_dn_p)
                D_dn = np.zeros(fl.nx)

                cond = (S[k] <= S[km]) & (H_dn_m <= H_dn_p)
                D_dn[cond] = D_dn_min[cond]
                cond = (S[k] <= S[km]) & (H_dn_m > H_dn_p)
                D_dn[cond] = D_dn_max[cond]
                cond = (S[k] > S[km]) & (H_dn_m <= H_dn_p)
                D_dn[cond] = D_dn_max[cond]
                cond = (S[k] > S[km]) & (H_dn_m > H_dn_p)
                D_dn[cond] = D_dn_min[cond]

                # Eq. 37
                dt_stab = c_stab * dx**2. / max(max(abs(D_up)), max(abs(D_dn)))
                dt_use = min(dt_stab, dt-stab_t)
                stab_t = stab_t + dt_use

                # explicit time stepping scheme, Eq. 36
                div_q = (D_up * (S[kp] - S[k])/dx - D_dn *
                         (S[k] - S[km])/dx)/dx
                # Eq. 35
                S = S[k] + (m_dot + div_q)*dt_use

                # Eq. 7
                S = np.maximum(S, B)

        # Done with the loop, prepare output
        fl.thick = S-B

        # Next step
        self.t += dt
