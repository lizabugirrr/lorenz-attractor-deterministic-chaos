"""
Геометричний аналіз: відображення Лоренца (максимуми z), перетин Пуанкаре,
біфуркаційна діаграма, кореляційна розмірність, пошук глобальних біфуркацій
(гомоклінічний вибух r≈13.926, поява дивного атрактора r≈24.06).
"""
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from .core import rk4_step, rhs, jacobian, SIGMA, RHO, BETA


@njit(cache=True)
def z_maxima(s0, dt, n_transient, n_steps, sigma=SIGMA, rho=RHO, beta=BETA, max_out=200000):
    """Послідовні локальні максимуми z(t) (параболічне уточнення)."""
    s = s0.copy()
    for _ in range(n_transient):
        s = rk4_step(s, dt, sigma, rho, beta)
    out = np.empty(max_out)
    k = 0
    z0 = s[2]
    s = rk4_step(s, dt, sigma, rho, beta)
    z1 = s[2]
    for _ in range(n_steps):
        s = rk4_step(s, dt, sigma, rho, beta)
        z2 = s[2]
        if z1 > z0 and z1 >= z2 and k < max_out:
            den = z0 - 2 * z1 + z2
            if den != 0.0:
                p = 0.5 * (z0 - z2) / den
                out[k] = z1 - 0.25 * (z0 - z2) * p
            else:
                out[k] = z1
            k += 1
        z0, z1 = z1, z2
    return out[:k]


@njit(parallel=True, cache=True)
def bifurcation_zmax(rhos, dt, n_transient, n_steps, keep, sigma=SIGMA, beta=BETA):
    """Для кожного r — останні `keep` максимумів z (NaN, якщо їх менше)."""
    out = np.full((rhos.size, keep), np.nan)
    for i in prange(rhos.size):
        s0 = np.array([0.1, 0.2, 0.3])
        m = z_maxima(s0, dt, n_transient, n_steps, sigma, rhos[i], beta, 100000)
        n = min(keep, m.size)
        for j in range(n):
            out[i, j] = m[m.size - n + j]
    return out


@njit(cache=True)
def poincare_section(s0, dt, n_transient, n_steps, z_plane, sigma=SIGMA, rho=RHO, beta=BETA,
                     max_out=200000):
    """Перетини площини z = z_plane знизу вгору (dz/dt>0); лінійна інтерполяція."""
    s = s0.copy()
    for _ in range(n_transient):
        s = rk4_step(s, dt, sigma, rho, beta)
    out = np.empty((max_out, 2))
    k = 0
    for _ in range(n_steps):
        s_new = rk4_step(s, dt, sigma, rho, beta)
        if s[2] < z_plane <= s_new[2] and k < max_out:
            a = (z_plane - s[2]) / (s_new[2] - s[2])
            out[k, 0] = s[0] + a * (s_new[0] - s[0])
            out[k, 1] = s[1] + a * (s_new[1] - s[1])
            k += 1
        s = s_new
    return out[:k]


def correlation_sum(points, radii, n_ref=4000, seed=0):
    """Кореляційний інтеграл Грассбергера–Прокачча C(ε)."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(points)
    idx = rng.choice(len(points), size=min(n_ref, len(points)), replace=False)
    ref = cKDTree(points[idx])
    counts = ref.count_neighbors(tree, radii).astype(float)
    counts -= len(idx)  # прибираємо пари точки з самою собою
    return counts / (len(idx) * (len(points) - 1))


# ---------------------------------------------------------------------------
# Глобальні біфуркації через нестійкий многовид початку координат
# ---------------------------------------------------------------------------
def _unstable_dir(rho, sigma=SIGMA, beta=BETA):
    J = jacobian(np.zeros(3), sigma, rho, beta)
    w, V = np.linalg.eig(J)
    v = np.real(V[:, np.argmax(np.real(w))])
    return v if v[0] > 0 else -v


@njit(cache=True)
def _ever_crosses(s, dt, n_max, sigma, rho, beta):
    """Чи стає x<0 для гілки W^u(0), що стартує в бік x>0."""
    for i in range(n_max):
        s = rk4_step(s, dt, sigma, rho, beta)
        if s[0] < 0:
            return True
    return False


def homoclinic_rho(lo=10.0, hi=20.0, tol=1e-6, dt=1e-3, sigma=SIGMA, beta=BETA):
    """Бісекція по r: при r < r_hom гілка W^u(0) з першого ж оберту
    закручується до C+ (не перетинає x=0), при r > r_hom — перескакує на
    інший бік. Межа — гомоклінічна петля (гомоклінічний вибух)."""
    def crosses(rho):
        v = _unstable_dir(rho, sigma, beta)
        s = 1e-9 * v
        return _ever_crosses(s, dt, int(30.0 / dt), sigma, rho, beta)
    assert not crosses(lo) and crosses(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if crosses(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _settles(s, dt, n_max, sigma, rho, beta):
    """Чи збігається траєкторія до C± за n_max кроків (повертає крок або -1)."""
    c = np.sqrt(beta * (rho - 1))
    for i in range(n_max):
        s = rk4_step(s, dt, sigma, rho, beta)
        if i % 100 == 0:
            d1 = (s[0] - c) ** 2 + (s[1] - c) ** 2 + (s[2] - rho + 1) ** 2
            d2 = (s[0] + c) ** 2 + (s[1] + c) ** 2 + (s[2] - rho + 1) ** 2
            if d1 < 1e-2 or d2 < 1e-2:
                return i
    return -1


@njit(parallel=True, cache=True)
def transient_lifetimes(rho, starts, dt, n_max, sigma=SIGMA, beta=BETA):
    """Час життя хаотичного транзієнта до захоплення C± для набору старт. точок."""
    out = np.empty(starts.shape[0])
    for k in prange(starts.shape[0]):
        i = _settles(starts[k].copy(), dt, n_max, sigma, rho, beta)
        out[k] = i * dt if i >= 0 else np.inf
    return out


def strange_attractor_onset(lo=23.0, hi=24.6, tol=2e-3, dt=5e-3, T=4000.0, sigma=SIGMA, beta=BETA):
    """Бісекція для r_A≈24.06: гілка W^u(0) при r<r_A врешті падає на C±,
    при r>r_A — залишається на дивному атракторі (вже не захоплюється).
    Оскільки поблизу r_A тривалість транзієнта необмежено зростає, оцінка
    залежить від T; повертаємо інтервал, що містить межу."""
    def settles(rho):
        v = _unstable_dir(rho, sigma, beta)
        s = 1e-6 * v
        return _settles(s, dt, int(T / dt), sigma, rho, beta) >= 0
    assert settles(lo) and not settles(hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if settles(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi
