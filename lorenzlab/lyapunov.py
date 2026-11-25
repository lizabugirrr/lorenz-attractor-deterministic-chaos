"""
Ляпуновські показники: алгоритм Бенеттіна (QR / Грама–Шмідта),
скінченночасові показники (FTLE), розмірність Каплана–Йорке.
"""
import numpy as np
from numba import njit, prange

from .core import rk4_step, rk4_tangent_step, SIGMA, RHO, BETA


@njit(cache=True)
def gram_schmidt(Q):
    """Модифікований Грам–Шмідт: Q = Q_orth * R; повертає (Q_orth, diag(R))."""
    n, k = Q.shape
    V = Q.copy()
    r = np.empty(k)
    for j in range(k):
        for i in range(j):
            proj = 0.0
            for m in range(n):
                proj += V[m, i] * V[m, j]
            for m in range(n):
                V[m, j] -= proj * V[m, i]
        nrm = 0.0
        for m in range(n):
            nrm += V[m, j] * V[m, j]
        nrm = np.sqrt(nrm)
        r[j] = nrm
        for m in range(n):
            V[m, j] /= nrm
    return V, r


@njit(cache=True)
def lyapunov_spectrum(s0, dt, n_transient, n_steps, sigma=SIGMA, rho=RHO, beta=BETA,
                      renorm=1, record_every=100):
    """Повний спектр показників Ляпунова методом Бенеттіна.

    Повертає (lambdas, history) де history[k] — поточні оцінки після
    (k+1)*record_every кроків (для графіка збіжності)."""
    s = s0.copy()
    for _ in range(n_transient):
        s = rk4_step(s, dt, sigma, rho, beta)
    Q = np.eye(3)
    sums = np.zeros(3)
    n_rec = n_steps // record_every
    hist = np.empty((n_rec, 3))
    rec = 0
    for i in range(1, n_steps + 1):
        s, Q = rk4_tangent_step(s, Q, dt, sigma, rho, beta)
        if i % renorm == 0:
            Q, r = gram_schmidt(Q)
            for j in range(3):
                sums[j] += np.log(r[j])
        if i % record_every == 0 and rec < n_rec:
            hist[rec] = sums / (i * dt)
            rec += 1
    return sums / (n_steps * dt), hist


@njit(parallel=True, cache=True)
def spectrum_vs_rho(rhos, dt, n_transient, n_steps, sigma=SIGMA, beta=BETA):
    out = np.empty((rhos.size, 3))
    for k in prange(rhos.size):
        s0 = np.array([1.0, 1.0, 1.0]) + 1e-3 * k
        lam, _ = lyapunov_spectrum(s0, dt, n_transient, n_steps, sigma, rhos[k], beta, 1, n_steps)
        out[k] = lam
    return out


@njit(cache=True)
def ftle_along_trajectory(s0, dt, n_transient, n_steps, window, sigma=SIGMA, rho=RHO, beta=BETA):
    """Скінченночасовий найбільший показник Ляпунова λ_T(x(t)) уздовж
    траєкторії: дотичний вектор, вже вирівняний у найбільш нестійкому
    напрямі, переноситься вздовж орбіти; λ_T(t) = (1/T) Σ_{[t,t+T]} ln‖·‖.

    Повертає (points[n_steps-window], ftle[n_steps-window])."""
    s = s0.copy()
    Q = np.ones((3, 1)) / np.sqrt(3.0)
    for _ in range(n_transient):           # вирівнювання вектора
        s, Q = rk4_tangent_step(s, Q, dt, sigma, rho, beta)
        Q /= np.sqrt(np.sum(Q * Q))
    pts = np.empty((n_steps, 3))
    logg = np.empty(n_steps)
    for i in range(n_steps):
        pts[i] = s
        s, Q = rk4_tangent_step(s, Q, dt, sigma, rho, beta)
        nrm = np.sqrt(np.sum(Q * Q))
        logg[i] = np.log(nrm)
        Q /= nrm
    m = n_steps - window
    out = np.empty(m)
    acc = 0.0
    for i in range(window):
        acc += logg[i]
    for i in range(m):
        out[i] = acc / (window * dt)
        acc += logg[i + window] - logg[i]
    return pts[:m], out


def kaplan_yorke(lams):
    lams = np.sort(np.asarray(lams))[::-1]
    c = np.cumsum(lams)
    j = np.max(np.where(c >= 0)[0]) if np.any(c >= 0) else -1
    if j == -1:
        return 0.0
    if j == len(lams) - 1:
        return float(len(lams))
    return (j + 1) + c[j] / abs(lams[j + 1])
