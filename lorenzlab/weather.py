"""
Хаос і погода: горизонт передбачуваності, ансамблеві прогнози,
модель Лоренца-96, асиміляція даних ансамблевим фільтром Калмана (EnKF).
"""
import numpy as np
from numba import njit, prange

from .core import rk4_step, advance, SIGMA, RHO, BETA


# ---------------------------------------------------------------------------
# Горизонт передбачуваності для Лоренца-63
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def predictability_times(starts, delta0s, tol, dt, t_max, sigma=SIGMA, rho=RHO, beta=BETA):
    """T[i, j] — час, за який похибка δ0=delta0s[j], внесена у стан starts[i]
    (у випадковому напрямі), зростає до tol."""
    n, m = starts.shape[0], delta0s.size
    out = np.empty((n, m))
    n_max = int(t_max / dt)
    for i in prange(n):
        # фіксований псевдовипадковий напрям
        d = np.array([np.sin(1.3 * i + 0.1), np.cos(2.1 * i + 0.7), np.sin(0.7 * i + 1.9)])
        d /= np.sqrt(np.sum(d * d))
        for j in range(m):
            a = starts[i].copy()
            b = a + delta0s[j] * d
            t = np.inf
            for k in range(1, n_max + 1):
                a = rk4_step(a, dt, sigma, rho, beta)
                b = rk4_step(b, dt, sigma, rho, beta)
                e = np.sqrt(np.sum((a - b) ** 2))
                if e > tol:
                    t = k * dt
                    break
            out[i, j] = t
    return out


@njit(parallel=True, cache=True)
def ensemble_run(members, dt, n_steps, stride, sigma=SIGMA, rho=RHO, beta=BETA):
    """Інтегрує ансамбль (N,3) -> (N, n_steps//stride+1, 3)."""
    N = members.shape[0]
    n_out = n_steps // stride + 1
    out = np.empty((N, n_out, 3))
    for i in prange(N):
        s = members[i].copy()
        out[i, 0] = s
        j = 1
        for k in range(1, n_steps + 1):
            s = rk4_step(s, dt, sigma, rho, beta)
            if k % stride == 0:
                out[i, j] = s
                j += 1
    return out


# ---------------------------------------------------------------------------
# Модель Лоренца-96:  dX_k/dt = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F
# ---------------------------------------------------------------------------
@njit(cache=True)
def l96_rhs(x, F):
    n = x.size
    d = np.empty(n)
    for k in range(n):
        d[k] = (x[(k + 1) % n] - x[(k - 2) % n]) * x[(k - 1) % n] - x[k] + F
    return d


@njit(cache=True)
def l96_step(x, dt, F):
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def l96_integrate(x0, dt, n_steps, F, stride=1):
    n_out = n_steps // stride + 1
    out = np.empty((n_out, x0.size))
    x = x0.copy()
    out[0] = x
    j = 1
    for i in range(1, n_steps + 1):
        x = l96_step(x, dt, F)
        if i % stride == 0:
            out[j] = x
            j += 1
    return out


@njit(cache=True)
def l96_jac(x, F):
    n = x.size
    J = np.zeros((n, n))
    for k in range(n):
        J[k, (k + 1) % n] += x[(k - 1) % n]
        J[k, (k - 2) % n] -= x[(k - 1) % n]
        J[k, (k - 1) % n] += x[(k + 1) % n] - x[(k - 2) % n]
        J[k, k] -= 1.0
    return J


@njit(cache=True)
def l96_lyapunov(x0, dt, n_transient, n_steps, F, k_exp, renorm=10):
    """Перші k_exp показників Ляпунова Лоренца-96 (Бенеттін, QR)."""
    x = x0.copy()
    for _ in range(n_transient):
        x = l96_step(x, dt, F)
    n = x.size
    Q = np.eye(n)[:, :k_exp].copy()
    sums = np.zeros(k_exp)
    for i in range(1, n_steps + 1):
        # RK4 для (x, Q)
        a1 = l96_rhs(x, F); A1 = l96_jac(x, F) @ Q
        xa = x + 0.5 * dt * a1; Qa = Q + 0.5 * dt * A1
        a2 = l96_rhs(xa, F); A2 = l96_jac(xa, F) @ Qa
        xb = x + 0.5 * dt * a2; Qb = Q + 0.5 * dt * A2
        a3 = l96_rhs(xb, F); A3 = l96_jac(xb, F) @ Qb
        xc = x + dt * a3; Qc = Q + dt * A3
        a4 = l96_rhs(xc, F); A4 = l96_jac(xc, F) @ Qc
        x = x + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        Q = Q + dt / 6.0 * (A1 + 2 * A2 + 2 * A3 + A4)
        if i % renorm == 0:
            Qn, R = np.linalg.qr(Q)
            for j in range(k_exp):
                sums[j] += np.log(abs(R[j, j]))
            Q = np.ascontiguousarray(Qn)
    return sums / (n_steps * dt)


# ---------------------------------------------------------------------------
# Ансамблевий фільтр Калмана (стохастичний EnKF) для Лоренца-63
# ---------------------------------------------------------------------------
def enkf_lorenz63(truth0, n_cycles, dt_obs, dt, obs_idx, obs_std, n_ens=20,
                  inflation=1.05, init_spread=3.0, seed=1,
                  sigma=SIGMA, rho=RHO, beta=BETA):
    """Twin-експеримент: 'істина' -> зашумлені спостереження компонент obs_idx
    кожні dt_obs -> аналіз EnKF. Повертає словник з істиною, спостереженнями,
    середнім аналізу, розкидом та вільним прогнозом (без асиміляції)."""
    rng = np.random.default_rng(seed)
    nsub = int(round(dt_obs / dt))
    H = np.zeros((len(obs_idx), 3))
    for i, k in enumerate(obs_idx):
        H[i, k] = 1.0
    Rm = obs_std ** 2 * np.eye(len(obs_idx))

    truth = np.asarray(truth0, float).copy()
    ens = truth + init_spread * rng.standard_normal((n_ens, 3))
    free = ens.mean(axis=0).copy()          # прогноз без асиміляції

    T, O, A, S, Fr, Bk = [], [], [], [], [], []
    for c in range(n_cycles):
        truth = advance(truth, dt, nsub, sigma, rho, beta)
        free = advance(free, dt, nsub, sigma, rho, beta)
        for m in range(n_ens):
            ens[m] = advance(ens[m], dt, nsub, sigma, rho, beta)
        bkg_mean = ens.mean(axis=0)
        y = H @ truth + obs_std * rng.standard_normal(len(obs_idx))

        # інфляція коваріації
        Xp = ens - ens.mean(axis=0)
        ens = ens.mean(axis=0) + inflation * Xp
        Xp = ens - ens.mean(axis=0)
        P = Xp.T @ Xp / (n_ens - 1)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rm)
        for m in range(n_ens):
            y_pert = y + obs_std * rng.standard_normal(len(obs_idx))
            ens[m] = ens[m] + K @ (y_pert - H @ ens[m])

        T.append(truth.copy()); O.append(y.copy()); A.append(ens.mean(axis=0))
        S.append(np.sqrt(np.trace(np.cov(ens.T)) / 3)); Fr.append(free.copy()); Bk.append(bkg_mean)
    return dict(truth=np.array(T), obs=np.array(O), analysis=np.array(A),
                spread=np.array(S), free=np.array(Fr), background=np.array(Bk),
                t=dt_obs * np.arange(1, n_cycles + 1))
