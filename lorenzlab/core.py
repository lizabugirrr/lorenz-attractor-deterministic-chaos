"""
Ядро: система Лоренца, якобіан, власні інтегратори (Euler, RK4, RK4 для
варіаційних рівнянь), рівноваги та їхня лінійна стійкість.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z
"""
import numpy as np
from numba import njit

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


# ---------------------------------------------------------------------------
# Векторне поле та якобіан
# ---------------------------------------------------------------------------
@njit(cache=True)
def rhs(s, sigma, rho, beta):
    x, y, z = s[0], s[1], s[2]
    out = np.empty(3)
    out[0] = sigma * (y - x)
    out[1] = x * (rho - z) - y
    out[2] = x * y - beta * z
    return out


@njit(cache=True)
def jacobian(s, sigma, rho, beta):
    x, y, z = s[0], s[1], s[2]
    J = np.empty((3, 3))
    J[0, 0] = -sigma; J[0, 1] = sigma; J[0, 2] = 0.0
    J[1, 0] = rho - z; J[1, 1] = -1.0; J[1, 2] = -x
    J[2, 0] = y; J[2, 1] = x; J[2, 2] = -beta
    return J


# ---------------------------------------------------------------------------
# Інтегратори
# ---------------------------------------------------------------------------
@njit(cache=True)
def euler_step(s, dt, sigma, rho, beta):
    return s + dt * rhs(s, sigma, rho, beta)


@njit(cache=True)
def rk4_step(s, dt, sigma, rho, beta):
    k1 = rhs(s, sigma, rho, beta)
    k2 = rhs(s + 0.5 * dt * k1, sigma, rho, beta)
    k3 = rhs(s + 0.5 * dt * k2, sigma, rho, beta)
    k4 = rhs(s + dt * k3, sigma, rho, beta)
    return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def integrate(s0, dt, n_steps, sigma=SIGMA, rho=RHO, beta=BETA, stride=1, method=4):
    """Інтегрує n_steps кроків; повертає траєкторію кожні `stride` кроків.
    method=4 -> RK4, method=1 -> явний Ейлер."""
    n_out = n_steps // stride + 1
    traj = np.empty((n_out, 3))
    s = s0.copy()
    traj[0] = s
    j = 1
    for i in range(1, n_steps + 1):
        if method == 4:
            s = rk4_step(s, dt, sigma, rho, beta)
        else:
            s = euler_step(s, dt, sigma, rho, beta)
        if i % stride == 0:
            traj[j] = s
            j += 1
    return traj


@njit(cache=True)
def advance(s0, dt, n_steps, sigma=SIGMA, rho=RHO, beta=BETA):
    """Лише кінцевий стан (без збереження траєкторії)."""
    s = s0.copy()
    for _ in range(n_steps):
        s = rk4_step(s, dt, sigma, rho, beta)
    return s


@njit(cache=True)
def integrate_f32(s0, dt, n_steps, sigma, rho, beta, stride=1):
    """RK4 в одинарній точності (float32) — для експерименту з округленням."""
    n_out = n_steps // stride + 1
    traj = np.empty((n_out, 3), dtype=np.float32)
    s = s0.astype(np.float32)
    sg = np.float32(sigma); r = np.float32(rho); b = np.float32(beta)
    h = np.float32(dt); half = np.float32(0.5); six = np.float32(6.0); two = np.float32(2.0)
    traj[0] = s
    j = 1
    for i in range(1, n_steps + 1):
        k1 = np.empty(3, dtype=np.float32)
        k1[0] = sg * (s[1] - s[0]); k1[1] = s[0] * (r - s[2]) - s[1]; k1[2] = s[0] * s[1] - b * s[2]
        u = s + half * h * k1
        k2 = np.empty(3, dtype=np.float32)
        k2[0] = sg * (u[1] - u[0]); k2[1] = u[0] * (r - u[2]) - u[1]; k2[2] = u[0] * u[1] - b * u[2]
        u = s + half * h * k2
        k3 = np.empty(3, dtype=np.float32)
        k3[0] = sg * (u[1] - u[0]); k3[1] = u[0] * (r - u[2]) - u[1]; k3[2] = u[0] * u[1] - b * u[2]
        u = s + h * k3
        k4 = np.empty(3, dtype=np.float32)
        k4[0] = sg * (u[1] - u[0]); k4[1] = u[0] * (r - u[2]) - u[1]; k4[2] = u[0] * u[1] - b * u[2]
        s = s + h / six * (k1 + two * k2 + two * k3 + k4)
        if i % stride == 0:
            traj[j] = s
            j += 1
    return traj


# ---------------------------------------------------------------------------
# Варіаційні рівняння:  d(delta)/dt = J(x(t)) delta
# ---------------------------------------------------------------------------
@njit(cache=True)
def _aug_rhs(s, Q, sigma, rho, beta):
    return rhs(s, sigma, rho, beta), jacobian(s, sigma, rho, beta) @ Q


@njit(cache=True)
def rk4_tangent_step(s, Q, dt, sigma, rho, beta):
    """Один крок RK4 для розширеної системи (x, Q), Q — 3xk матриця
    дотичних векторів."""
    a1, A1 = _aug_rhs(s, Q, sigma, rho, beta)
    a2, A2 = _aug_rhs(s + 0.5 * dt * a1, Q + 0.5 * dt * A1, sigma, rho, beta)
    a3, A3 = _aug_rhs(s + 0.5 * dt * a2, Q + 0.5 * dt * A2, sigma, rho, beta)
    a4, A4 = _aug_rhs(s + dt * a3, Q + dt * A3, sigma, rho, beta)
    s_new = s + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
    Q_new = Q + dt / 6.0 * (A1 + 2 * A2 + 2 * A3 + A4)
    return s_new, Q_new


# ---------------------------------------------------------------------------
# Рівноваги та їхня стійкість
# ---------------------------------------------------------------------------
def fixed_points(sigma=SIGMA, rho=RHO, beta=BETA):
    pts = [np.zeros(3)]
    if rho > 1:
        c = np.sqrt(beta * (rho - 1))
        pts.append(np.array([c, c, rho - 1]))
        pts.append(np.array([-c, -c, rho - 1]))
    return pts


def eigenvalues_at(p, sigma=SIGMA, rho=RHO, beta=BETA):
    return np.linalg.eigvals(jacobian(np.asarray(p, float), sigma, rho, beta))


def hopf_rho(sigma=SIGMA, beta=BETA):
    """Критичне r субкритичної біфуркації Хопфа для C±."""
    return sigma * (sigma + beta + 3) / (sigma - beta - 1)


def on_attractor(n_transient=5000, dt=0.01, s0=None, sigma=SIGMA, rho=RHO, beta=BETA):
    """Точка, що (практично) лежить на атракторі."""
    if s0 is None:
        s0 = np.array([1.0, 1.0, 1.0])
    return advance(np.asarray(s0, float), dt, n_transient, sigma, rho, beta)
