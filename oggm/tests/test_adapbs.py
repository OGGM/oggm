import numpy as np
import pytest
from numpy.testing import assert_allclose

from oggm.core.adapbs import (glogit, gexpit, amis_weights, nearest_pd,
                              weighted_quantile, run_adapbs)


def test_glogit_gexpit():
    x_min, x_max = np.array([1., -3]), np.array([5., 3])
    x = np.array([[1.5, 0.], [4.9, -2.9], [3, 2.5]])
    assert_allclose(gexpit(glogit(x, x_min, x_max), x_min, x_max), x)
    assert_allclose(glogit(0.5 * (x_min + x_max), x_min, x_max), 0, atol=1e-12)
    # Values at the bounds stay finite
    z = glogit(np.array([x_min, x_max]), x_min, x_max)
    assert np.all(np.isfinite(z))


def test_nearest_pd():
    cov = np.array([[1., 0.9, 0.7], [0.9, 1., 0.99], [0.7, 0.99, 1.]]) * 2
    assert np.any(np.linalg.eigvalsh(cov) < 0)
    out = nearest_pd(cov)
    assert_allclose(np.diag(out), np.diag(cov))
    np.linalg.cholesky(out + np.eye(3) * 1e-12)
    # PD matrices are untouched
    pd = np.eye(3) * 2
    assert nearest_pd(pd) is pd


def test_weighted_quantile():
    x = np.arange(10.)
    assert weighted_quantile(x, np.ones(10), 0.5) == 4
    w = np.zeros(10)
    w[7] = 1
    assert_allclose(weighted_quantile(x, w, [0.05, 0.5, 0.95]), 7)


def test_amis_single_proposal_is_importance_sampling():
    # With one proposal equal to the prior, AMIS weights are the likelihood
    rng = np.random.default_rng(0)
    Np, Ne, No = 2, 100, 3
    props = rng.standard_normal((Np, Ne, 1))
    pred = rng.standard_normal((No, Ne, 1))
    obs = np.array([0.1, -0.2, 0.3])
    r = np.array([0.5, 1., 2.])
    w, neff, _ = amis_weights(obs, pred, r, np.zeros(Np), np.eye(Np),
                              np.zeros((Np, 1)), np.eye(Np)[:, :, None], props)
    ref = np.exp(-0.5 * np.sum((pred[:, :, 0].T - obs) ** 2 / r, axis=1))
    assert_allclose(w, ref / ref.sum())
    assert_allclose(neff, 1 / np.sum(w ** 2))


def test_run_adapbs_linear_gaussian():
    # Model linear in normal space: the posterior is Gaussian and known
    x_min, x_max = np.array([-50., -50]), np.array([50., 50])
    A = np.array([[1., 0.5], [0., 1.], [1., -1.]])
    y = np.array([1.5, -0.8, 2.])
    r = np.array([0.3, 0.2, 0.4]) ** 2
    m0, sd0 = np.array([0.5, 0.]), 1.

    def forward(theta, it):
        return glogit(theta, x_min, x_max) @ A.T

    p0 = np.eye(2) / sd0 ** 2
    cov = np.linalg.inv(p0 + A.T @ np.diag(1 / r) @ A)
    mean = cov @ (p0 @ m0 + A.T @ (y / r))

    res = run_adapbs(forward, y, np.sqrt(r), x_min, x_max, prior_mean=m0,
                     prior_sd=sd0, n_particles=500, max_iterations=8,
                     neff_threshold=0.5, rng=np.random.default_rng(1))
    assert len(res['diagnostics']) > 1  # it had to adapt
    z = glogit(res['theta'], x_min, x_max)
    w = res['weights']
    zm = np.average(z, axis=0, weights=w)
    zc = np.cov(z.T, aweights=w)
    sd = np.sqrt(np.diag(cov))
    assert_allclose(zm, mean, atol=0.1 * sd.max())
    assert_allclose(np.sqrt(np.diag(zc)), sd, rtol=0.15)
    assert len(res['theta']) == len(w) == len(res['pred']) == len(res['iteration'])
    assert res['posterior_theta'].shape == (500, 2)
    assert_allclose(np.sum(w), 1)


def test_run_adapbs_failed_runs():
    x_min, x_max = np.array([0.]), np.array([10.])

    def forward(theta, it):
        out = theta.copy()
        out[theta > 7] = np.nan  # failed runs
        return out, {'x': theta[:, 0]}

    res = run_adapbs(forward, [5.], [1.], x_min, x_max, n_particles=200,
                     max_iterations=3, rng=np.random.default_rng(0))
    failed = np.isnan(res['pred'][:, 0])
    assert failed.any()
    assert np.all(res['weights'][failed] == 0)
    assert_allclose(res['extras']['x'], res['theta'][:, 0])


def test_amis_matches_musa():
    # Local check against MuSA's own AMIS (needs a MuSA clone, see the
    # adapbs project notes). Skipped otherwise.
    import ast
    import os
    from types import SimpleNamespace
    from scipy.special import expit, logit
    musa = os.environ.get('MUSA_DIR')
    if musa is None or not os.path.exists(musa):
        pytest.skip('Set MUSA_DIR to a MuSA clone to run this test.')
    path = os.path.join(musa, 'modules', 'filters.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == 'AMIS']

    class _NoLimits:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    ns = {'np': np, 'logit': logit, 'expit': expit,
          'threadpool_limits': _NoLimits, 'threadpool_limits_safe': _NoLimits,
          'cfg': SimpleNamespace(numpy_threads=1)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)

    rng = np.random.default_rng(0)
    No, Ne, Nl, Np = 2, 50, 3, 4
    obs = rng.normal(size=No)
    pred = rng.normal(size=(No, Ne, Nl))
    r = rng.uniform(0.5, 2, size=No)
    a = rng.normal(size=(Np, Np))
    pric = a @ a.T + np.eye(Np)
    propm = rng.normal(size=(Np, Nl))
    propc = np.stack([b @ b.T + np.eye(Np) for b in
                      rng.normal(size=(Nl, Np, Np))], -1)
    props = rng.normal(size=(Np, Ne, Nl))
    prim = rng.normal(size=Np)
    w1, n1 = ns['AMIS'](obs, pred, r, prim, pric, propm, propc, props)
    w2, n2, _ = amis_weights(obs, pred, r, prim, pric, propm, propc, props)
    assert_allclose(w1, w2, atol=1e-14)
    assert_allclose(n1, n2)


class TestMBCalibration:

    @pytest.mark.slow
    def test_mb_calibration_adapbs(self, hef_gdir):
        from oggm import tasks
        from oggm.core import massbalance

        gdir = hef_gdir
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        ref_mb = mbdf.mean()
        period = f'{mbdf.index[0]}-01-01_{mbdf.index[-1] + 1}-01-01'

        # Deterministic: melt_f only, prcp_fac=2.5 (cfg) and temp_bias=0
        det = massbalance.mb_calibration_from_scalar_mb(
            gdir, ref_mb=ref_mb, ref_mb_period=period, write_to_gdir=False,
            overwrite_observations=True)

        # One parameter, small error: the posterior is around det
        ds = tasks.mb_calibration_from_geodetic_mb_adapbs(
            gdir, ref_mb=ref_mb, ref_mb_err=20, ref_mb_period=period,
            calibrate_params=['melt_f'], n_particles=200, seed=0)
        assert ds.attrs['calibrate_params'] == ['melt_f']
        assert_allclose(ds.attrs['prior_centre'], [5])
        w = ds.weight.values
        med = weighted_quantile(ds.melt_f.values, w, 0.5)
        assert_allclose(med, det['melt_f'], atol=0.3)
        assert_allclose(np.average(ds.mb[:, 0], weights=w), ref_mb, atol=20)
        assert np.median(ds.posterior_melt_f) == pytest.approx(med, abs=0.3)

        # Three parameters: runs, weights sum to 1, within the bounds
        ds = tasks.mb_calibration_from_geodetic_mb_adapbs(
            gdir, ref_mb=ref_mb, ref_mb_err=100, ref_mb_period=period,
            n_particles=100, max_iterations=3, seed=0)
        assert_allclose(ds.weight.sum(), 1)
        for k, p in enumerate(ds.attrs['calibrate_params']):
            assert ds[p].min() >= ds.attrs['prior_min'][k]
            assert ds[p].max() <= ds.attrs['prior_max'][k]
        assert_allclose(ds.attrs['prior_centre'][1], 2.5)
        assert len(ds.adapbs_iteration) >= 1
