"""Adaptive Particle Batch Smoother (AdaPBS).

Bayesian parameter estimation by adaptive multiple importance sampling
(AMIS, Cornuet et al., 2012), as implemented in MuSA
(https://github.com/ealonsogzl/MuSA, AdaPBS by K. Aalstad). This module keeps
MuSA's algorithm and replaces MuSA's model ensemble by a ``forward`` function.

This module has no OGGM dependency on purpose: it can be moved to a separate
package without changes.

Licence note: the AMIS function and the AdaPBS loop are adapted from MuSA,
which is licensed under the GPL-3.0. Do not release this module under OGGM's
BSD-3 licence before this is sorted out.

Differences with MuSA:

- log-determinants are computed with ``slogdet`` instead of
  ``det(...) ** -0.5`` (same result, no under/overflow with many parameters)
- statsmodels' ``cov_nearest(method='clipped')`` and
  ``DescrStatsW.quantile`` are replaced by numpy equivalents
- only bounded (logit-normal) parameters are supported
- failed forward runs (NaN predictions) get a zero weight
- ``amis_weights`` also returns the log of the model evidence
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import expit, logit

log = logging.getLogger(__name__)


def glogit(x, x_min, x_max):
    """Bounded parameter space (x_min, x_max) -> normal space (generalized logit).

    Values at the bounds are moved inside by a few float spacings, as in MuSA.
    """
    x = np.array(x, dtype=float)
    x_min = np.broadcast_to(x_min, x.shape)
    x_max = np.broadcast_to(x_max, x.shape)
    at_min = x == x_min
    x[at_min] += np.abs(np.spacing(x[at_min]) * 100)
    at_max = x == x_max
    x[at_max] -= np.abs(np.spacing(x[at_max]) * 100)
    return logit((x - x_min) / (x_max - x_min))


def gexpit(z, x_min, x_max):
    """Normal space -> bounded parameter space (x_min, x_max) (generalized expit)."""
    return (np.asarray(x_max) - np.asarray(x_min)) * expit(z) + np.asarray(x_min)


def _logdet(a):
    sign, ld = np.linalg.slogdet(a)
    if sign <= 0:
        raise np.linalg.LinAlgError('Matrix is not positive definite.')
    return ld


def _quad_form(a, cov):
    """0.5 * a^T cov^-1 a for each column of a (Np x N) -> (N,)."""
    return 0.5 * np.sum(a * np.linalg.solve(cov, a), axis=0)


def amis_weights(obs, pred, obs_var, prior_mean, prior_cov,
                 prop_means, prop_covs, prop_samples):
    """AMIS weights of all the particles so far (MuSA's ``AMIS``).

    The target is the posterior in normal space: Gaussian prior times a
    Gaussian likelihood with independent observation errors. The proposal is
    the deterministic mixture of all the Gaussian proposals so far.

    Parameters
    ----------
    obs : (No,) array
        the observations
    pred : (No, Ne, Nl) array
        predicted observations, for Ne particles at each of Nl iterations.
        Use +inf for failed runs (zero weight).
    obs_var : scalar or (No,) array
        observation error variances
    prior_mean : (Np,) array
        prior mean (normal space)
    prior_cov : (Np, Np) array
        prior covariance (normal space)
    prop_means : (Np, Nl) array
        means of the proposals
    prop_covs : (Np, Np, Nl) array
        covariances of the proposals
    prop_samples : (Np, Ne, Nl) array
        the particles (normal space)

    Returns
    -------
    (w, neff, log_z): normalised weights (Ne * Nl,), particles ordered
    iteration by iteration, the effective sample size 1 / sum(w**2), and the
    log of the model evidence.
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    No, Ne, Nl = pred.shape
    obs_var = np.broadcast_to(np.asarray(obs_var, dtype=float), (No,))

    # Normalising constants of the likelihood and prior (only needed for the
    # evidence, they cancel in the weights)
    log_b = (-0.5 * np.sum(np.log(2 * np.pi * obs_var))
             - 0.5 * _logdet(2 * np.pi * prior_cov))
    # Normalising constants of the proposals
    log_c = np.array([-0.5 * _logdet(2 * np.pi * prop_covs[:, :, j])
                      for j in range(Nl)])

    phi = np.zeros((Ne, Nl))  # negative log of the (unnormalised) target
    lse_psi = np.zeros((Ne, Nl))  # log of the mixture proposal density
    for ell in range(Nl):
        props = prop_samples[:, :, ell]  # Np x Ne
        phi0 = _quad_form((props.T - prior_mean).T, prior_cov)
        resid = (obs - pred[:, :, ell].T).T  # No x Ne
        phid = 0.5 * (1 / obs_var) @ (resid ** 2)
        phi[:, ell] = phi0 + phid

        psi = np.zeros((Ne, Nl))
        for j in range(Nl):
            a = (props.T - prop_means[:, j]).T
            psi[:, j] = log_c[j] - _quad_form(a, prop_covs[:, :, j])
        psi_max = np.max(psi, axis=1)
        lse_psi[:, ell] = psi_max + np.log(np.sum(np.exp((psi.T - psi_max).T),
                                                  axis=1))

    logwt = (log_b - phi - lse_psi).flatten('F')  # iteration by iteration
    lwt_max = np.max(logwt)
    lse_lwt = lwt_max + np.log(np.sum(np.exp(logwt - lwt_max)))
    log_z = lse_lwt - np.log(Nl * Ne)
    w = np.exp(logwt - lse_lwt)
    neff = 1 / np.sum(w ** 2)
    return w, neff, log_z


def nearest_pd(cov, threshold=1e-15):
    """Nearest positive (semi-)definite covariance, keeping the variances.

    Same as statsmodels' ``cov_nearest(cov, method='clipped')``: clip the
    eigenvalues of the correlation matrix at ``threshold``, rescale.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    evals, evecs = np.linalg.eigh(corr)
    if np.all(evals >= threshold):
        return cov
    corr = (evecs * np.maximum(evals, threshold)) @ evecs.T
    s = np.sqrt(np.diag(corr))
    corr = corr / np.outer(s, s)
    return corr * np.outer(std, std)


def _cholesky(cov, max_tries=10):
    """Cholesky factor, fixing a non positive definite matrix if needed."""
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(cov), cov
        except np.linalg.LinAlgError:
            cov = nearest_pd(cov)
    # MuSA loops until it works. We add a small jitter as last resort
    cov = cov + np.eye(len(cov)) * 1e-10 * np.mean(np.diag(cov))
    return np.linalg.cholesky(cov), cov


def weighted_quantile(x, weights, q):
    """Weighted quantiles (inverse of the weighted empirical CDF).

    Parameters
    ----------
    x : (N,) array
    weights : (N,) array
    q : scalar or array of probabilities in [0, 1]
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    x, weights = x[ok], weights[ok]
    order = np.argsort(x)
    x, weights = x[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = np.searchsorted(cdf, q, side='left')
    return x[np.clip(idx, 0, len(x) - 1)]


def run_adapbs(forward, obs, obs_err, x_min, x_max, *,
               prior_mean=None, prior_sd=1., n_particles=100,
               max_iterations=4, neff_threshold=0.3, rng=None):
    """Run the AdaPBS (MuSA's AdaPBS branch of ``implement_assimilation``).

    Each iteration draws ``n_particles`` parameter sets from a Gaussian
    proposal in normal space (the prior at the first iteration), runs
    ``forward`` on them and weights all the particles so far with AMIS. If
    the effective sample size is below ``neff_threshold * n_particles``, the
    largest weights are clipped, the particles resampled, and their mean and
    covariance become the next proposal. Otherwise (or at the last iteration)
    it stops.

    Parameters
    ----------
    forward : callable
        ``forward(theta, iteration)`` with theta a (Ne, Np) array of
        parameters (bounded space) and iteration the 0-based iteration. Returns
        the predicted observations as a (Ne, No) array (NaN for failed runs),
        or a tuple (predictions, extras) with extras a dict of arrays with
        one row per particle, stored with the results.
    obs : (No,) array
        the observations
    obs_err : scalar or (No,) array
        the observation errors (1 sigma)
    x_min, x_max : (Np,) arrays
        bounds of the parameters
    prior_mean : (Np,) array, optional
        prior mean in normal space (default: 0, the middle of the bounds).
        Use ``glogit(centre, x_min, x_max)`` to centre the prior on a value.
    prior_sd : scalar or (Np,) array
        prior standard deviation in normal space (default: 1). Above ~1.8
        the logit-normal becomes bimodal (mass at the bounds).
    n_particles : int
        particles per iteration (Ne)
    max_iterations : int
        maximum number of iterations
    neff_threshold : float
        adapt while Neff / Ne is below this (default 0.3; MuSA's default
        is 0.1)
    rng : np.random.Generator, optional

    Returns
    -------
    a dict with:

    - theta: (Nw, Np) all the particles (bounded space), iteration by iteration
    - weights: (Nw,) their posterior (AMIS) weights
    - pred: (Nw, No) their predictions (NaN for failed runs)
    - iteration: (Nw,) the 1-based iteration of each particle
    - weights_per_iteration: list of the AMIS weights after each iteration
    - posterior_theta: (Ne, Np) equally weighted resampled particles
    - log_evidence: log of the model evidence (AMIS estimate)
    - extras: dict of the forward extras, concatenated
    - diagnostics: pandas DataFrame, one row per iteration
    """
    if rng is None:
        rng = np.random.default_rng()

    x_min = np.atleast_1d(np.asarray(x_min, dtype=float))
    x_max = np.atleast_1d(np.asarray(x_max, dtype=float))
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    Np, No, Ne, Nl = len(x_min), len(obs), int(n_particles), int(max_iterations)
    obs_var = np.broadcast_to(np.asarray(obs_err, dtype=float) ** 2, (No,))
    if prior_mean is None:
        prior_mean = np.zeros(Np)
    prior_mean = np.broadcast_to(np.asarray(prior_mean, dtype=float), (Np,))
    prior_sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (Np,))
    prior_cov = np.diag(prior_sd ** 2)

    pred_all = np.full((No, Ne, Nl), np.nan)
    props_all = np.full((Np, Ne, Nl), np.nan)
    prop_means = np.full((Np, Nl), np.nan)
    prop_covs = np.full((Np, Np, Nl), np.nan)
    prop_means[:, 0] = prior_mean
    prop_covs[:, :, 0] = prior_cov

    proposal = (prior_mean +
                (np.linalg.cholesky(prior_cov) @ rng.standard_normal((Np, Ne))).T).T
    extras_hist, weights_hist, diagnostics = [], [], []

    for j in range(Nl):
        theta = gexpit(proposal.T, x_min, x_max)  # Ne x Np
        out = forward(theta, j)
        pred, extras = out if isinstance(out, tuple) else (out, {})
        pred = np.asarray(pred, dtype=float).reshape(Ne, No)
        extras_hist.append(extras)

        props_all[:, :, j] = proposal
        failed = ~np.all(np.isfinite(pred), axis=1)
        pred_all[:, :, j] = np.where(np.isfinite(pred), pred, np.inf).T
        ells = np.arange(j + 1)
        w, neff, log_z = amis_weights(obs, pred_all[:, :, ells], obs_var,
                                      prior_mean, prior_cov,
                                      prop_means[:, ells], prop_covs[:, :, ells],
                                      props_all[:, :, ells])
        if not np.all(np.isfinite(w)):
            raise RuntimeError('All particles so far have a zero weight (all '
                               'runs failed or are infinitely far from the '
                               'observations).')
        weights_hist.append(w.copy())

        diversity = neff / Ne
        doadapt = diversity < neff_threshold
        notlast = (j + 1) < Nl

        # Clip the largest weights before resampling (when adapting)
        doclip = doadapt and notlast
        wr = w.copy()
        if doclip:
            clip = int(np.round(neff_threshold * Ne))
            wc = -np.sort(-wr)[clip - 1]
            if wc > 0:
                wr[wr > wc] = wc
                wr = wr / np.sum(wr)
            else:
                doclip = False

        Nw = np.size(wr)
        reinds = rng.choice(Nw, Ne, p=wr)
        thetap = np.reshape(props_all[:, :, ells], (Np, Nw), order='F')[:, reinds]
        pm = np.mean(thetap, axis=1)
        if doclip:
            a = (thetap.T - pm).T
            pc = (a @ a.T) / Ne
        else:
            pc = prior_cov * (0.5 ** j)

        diagnostics.append(dict(iteration=j + 1, n_history=Nw,
                                n_failed_new=int(failed.sum()),
                                neff=neff, neff_frac=diversity,
                                clipped=doclip, log_evidence=log_z))
        log.info(f'AdaPBS iteration {j + 1}: history={Nw}, Neff={neff:.1f}, '
                 f'Neff/Ne={diversity:.3f}, failed runs={int(failed.sum())}')

        if not (doadapt and notlast):
            pred_hist = np.reshape(pred_all[:, :, ells], (No, Nw), order='F').T
            theta_hist = np.reshape(props_all[:, :, ells], (Np, Nw), order='F')
            extras = {}
            if extras_hist[0]:
                extras = {k: np.concatenate([e[k] for e in extras_hist])
                          for k in extras_hist[0]}
            return dict(
                theta=gexpit(theta_hist.T, x_min, x_max),
                weights=w,
                pred=np.where(np.isinf(pred_hist), np.nan, pred_hist),
                iteration=np.repeat(np.arange(1, j + 2), Ne),
                weights_per_iteration=weights_hist,
                posterior_theta=gexpit(thetap.T, x_min, x_max),
                log_evidence=log_z,
                extras=extras,
                diagnostics=pd.DataFrame(diagnostics),
            )

        L, pc = _cholesky(pc)
        proposal = (pm + (L @ rng.standard_normal((Np, Ne))).T).T
        prop_covs[:, :, j + 1] = pc
        prop_means[:, j + 1] = pm
