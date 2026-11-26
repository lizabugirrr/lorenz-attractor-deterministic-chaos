"""Перевірка коректності власної реалізації через відомі аналітичні факти."""
import numpy as np

from lorenzlab.core import (rhs, jacobian, rk4_step, integrate, fixed_points, eigenvalues_at,
                            hopf_rho, SIGMA, RHO, BETA)
from lorenzlab.lyapunov import lyapunov_spectrum, kaplan_yorke, gram_schmidt


def test_fixed_points_are_equilibria():
    for r in (0.5, 5.0, 28.0, 100.0):
        for p in fixed_points(SIGMA, r, BETA):
            assert np.allclose(rhs(p, SIGMA, r, BETA), 0.0, atol=1e-12)


def test_divergence_is_constant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.normal(scale=20, size=3)
        assert np.isclose(np.trace(jacobian(s, SIGMA, RHO, BETA)), -(SIGMA + 1 + BETA))


def test_jacobian_matches_finite_differences():
    s = np.array([3.0, -2.0, 25.0]); h = 1e-6
    J = jacobian(s, SIGMA, RHO, BETA)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        fd = (rhs(s + e, SIGMA, RHO, BETA) - rhs(s - e, SIGMA, RHO, BETA)) / (2 * h)
        assert np.allclose(J[:, k], fd, atol=1e-6)


def test_symmetry():
    s0 = np.array([1.3, -0.4, 12.0])
    a = integrate(s0, 0.01, 300)
    b = integrate(s0 * np.array([-1, -1, 1.0]), 0.01, 300)
    assert np.allclose(a * np.array([-1, -1, 1.0]), b, atol=1e-9)


def test_rk4_order_four():
    s0 = np.array([1.0, 1.0, 20.0]); T = 1.0
    ref = integrate(s0, T / 40000, 40000)[-1]
    errs = []
    for n in (400, 800, 1600):
        errs.append(np.linalg.norm(integrate(s0, T / n, n)[-1] - ref))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 4) < 0.4)


def test_hopf_value():
    assert np.isclose(hopf_rho(10, 8 / 3), 470 / 19)
    lam_below = eigenvalues_at(fixed_points(10, 24.0, 8 / 3)[1], 10, 24.0, 8 / 3)
    lam_above = eigenvalues_at(fixed_points(10, 25.0, 8 / 3)[1], 10, 25.0, 8 / 3)
    assert np.max(lam_below.real) < 0 < np.max(lam_above.real)


def test_lyapunov_sum_equals_divergence():
    lam, _ = lyapunov_spectrum(np.array([1.0, 1.0, 1.0]), 0.01, 2000, 50000)
    assert np.isclose(lam.sum(), -(SIGMA + 1 + BETA), atol=1e-3)
    assert 0.8 < lam[0] < 1.0 and abs(lam[1]) < 0.05
    assert 2.0 < kaplan_yorke(lam) < 2.1


def test_gram_schmidt_orthonormal():
    Q = np.random.default_rng(1).normal(size=(3, 3))
    V, r = gram_schmidt(Q)
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-12)
    assert np.isclose(np.prod(r), abs(np.linalg.det(Q)))


# --- застосування (lorenzlab/applications.py) -------------------------------
from lorenzlab.core import on_attractor  # noqa: E402
from lorenzlab.applications import (chaotic_channel, waterwheel, wheel_params_for,  # noqa: E402
                                    wheel_to_lorenz, poly_library, stlsq, true_coefficients,
                                    derivative)


def test_receiver_synchronizes_and_recovers_message():
    dt = 0.001; n = 20000; t = np.arange(n) * dt
    msg = 0.3 * np.sin(2 * np.pi * 10 * t)
    u0 = np.concatenate([on_attractor(), [0.0, 0.0, 0.0]])
    _, rec = chaotic_channel(u0, dt, msg, np.array([SIGMA, RHO, BETA]))
    assert np.std((rec - msg)[t > 8]) < 0.05 * 0.3
    _, rec0 = chaotic_channel(u0, dt, np.zeros(n), np.array([SIGMA, RHO, BETA]))
    assert np.max(np.abs(rec0[t > 15])) < 1e-6      # без повідомлення — точна синхронізація


def test_waterwheel_is_lorenz_with_beta_one():
    """Моди маси колеса задовольняють Лоренца з beta=1: SINDy це підтверджує."""
    p = wheel_params_for(N=24)
    t, th, om, ms = waterwheel(p["N"], 60.0, 0.002, p["Q"], p["K"], p["nu"], p["I"], p["gR"], 0.1, 5)
    L = wheel_to_lorenz(th, om, ms, p)[t > 20]
    Xi = stlsq(poly_library(L[50:-50]), derivative(L, t[1] - t[0])[50:-50], 0.5)
    assert np.allclose(Xi, true_coefficients(10.0, 28.0, 1.0), atol=1e-2)


def test_sindy_recovers_lorenz():
    X = integrate(on_attractor(), 0.002, 5000)
    Xi = stlsq(poly_library(X[10:-10]), derivative(X, 0.002)[10:-10], 0.5)
    assert np.allclose(Xi, true_coefficients(), atol=1e-3)
