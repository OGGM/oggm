.. currentmodule:: oggm

Transitioning from OGGM 1.5 to 1.6
==================================

OGGM version 1.6 and above have considerably improved how the calibration is done
in OGGM. At the same time, we changed a few other aspects of the mass-balance
model. Here is an incomplete list of all the changes:

- the :math:`\mu^*` parameter (``mu_star`` in code) has been renamed to ``melt_f``
- its unit is now kg w.e. d:math:`^{-1}` day:math:`^{-1}` (instead on per month), regardless if the model is monthly or not (the conversion is done internally)
- we implemented a new, flexible calibration strategy. We apply it as explained in the tutorials, but allow for many more options for calibration
- we simplified the internal code a lot. Now all mass balance code can be found in the mass balance module (previously, mass balance calibration and application were separate).
- ... and much more.

As a result, code that was written with OGGM v1.5 probably wont work in 1.6 if you have some mass balance related code.
Furthermore, the results will probably change quite a lot (depending on your use case).

**We recommend all users to transition to OGGM 1.6 when they are ready to.** It's totally fine to continue using
OGGM 1.5.3 as long as you need to. For transitioning, we recommend to have a look at our brand new
`tutorials <https://oggm.org/tutorials>`_.

