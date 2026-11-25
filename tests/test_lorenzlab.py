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
