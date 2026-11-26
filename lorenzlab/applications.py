"""
Застосування системи Лоренца поза метеорологією:

  1. Секретний зв'язок на хаосі (Куомо–Оппенгейм 1993): повідомлення
     ховається в хаотичному сигналі, приймач-«близнюк» його відновлює.
  2. Водяне колесо Малкуса: дискретне колесо з N відер, що протікають;
     у континуальній границі — рівно система Лоренца з beta = 1.
  3. Reservoir computing (мережа луни, ESN): машинне навчання без рівнянь
     прогнозує хаотичну траєкторію (Pathak та ін. 2017/2018).
  4. SINDy (Brunton, Proctor, Kutz 2016): відновлення самих рівнянь
     із часових рядів розрідженою регресією.
"""
import numpy as np
from numba import njit

from .core import SIGMA, RHO, BETA


# ---------------------------------------------------------------------------
# 1. Секретний зв'язок: хаотичне маскування
# ---------------------------------------------------------------------------
@njit(cache=True)
def _txrx_rhs(u, s, sigma, rho, beta, sigma_r, rho_r, beta_r):
    """u = (x, y, z, xr, yr, zr). Передавач — система Лоренца; приймач — її копія,
    у якій x у нелінійних членах замінено переданим сигналом s (Куомо–Оппенгейм)."""
    x, y, z, xr, yr, zr = u[0], u[1], u[2], u[3], u[4], u[5]
    d = np.empty(6)
    d[0] = sigma * (y - x)
    d[1] = x * (rho - z) - y
    d[2] = x * y - beta * z
    d[3] = sigma_r * (yr - xr)
    d[4] = rho_r * s - yr - s * zr
    d[5] = s * yr - beta_r * zr
    return d


@njit(cache=True)
def chaotic_channel(u0, dt, message, key, sigma=SIGMA, rho=RHO, beta=BETA):
    """Передається s(t) = x(t) + m(t); приймач з параметрами key = (sigma, rho, beta)
    синхронізується з передавачем і відновлює m^ = s - x_r.
    Повертає (s, m^). Між вузлами RK4 повідомлення інтерполюється лінійно."""
    n = message.size
    sr, rr, br = key[0], key[1], key[2]
    u = u0.copy()
    s_out = np.empty(n)
    rec = np.empty(n)
    for i in range(n):
        m0 = message[i]
        m1 = message[min(i + 1, n - 1)]
        mh = 0.5 * (m0 + m1)
        s_out[i] = u[0] + m0
        rec[i] = u[0] + m0 - u[3]
        k1 = _txrx_rhs(u, u[0] + m0, sigma, rho, beta, sr, rr, br)
        v = u + 0.5 * dt * k1
        k2 = _txrx_rhs(v, v[0] + mh, sigma, rho, beta, sr, rr, br)
        v = u + 0.5 * dt * k2
        k3 = _txrx_rhs(v, v[0] + mh, sigma, rho, beta, sr, rr, br)
        v = u + dt * k3
        k4 = _txrx_rhs(v, v[0] + m1, sigma, rho, beta, sr, rr, br)
        u = u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s_out, rec


MORSE = {"Х": "....", "А": ".-", "О": "---", "С": "...", "H": "....", "A": ".-", "O": "---", "S": "..."}


def morse_envelope(text, unit, dt):
    """0/1-обвідна азбуки Морзе: крапка = 1 одиниця, тире = 3, пауза між
    знаками = 1, між літерами = 3. unit — тривалість одиниці (модельний час)."""
    seq = []
    for li, ch in enumerate(text):
        if li:
            seq += [0] * 3
        for si, sym in enumerate(MORSE[ch]):
            if si:
                seq += [0]
            seq += [1] * (1 if sym == "." else 3)
    k = int(round(unit / dt))
    return np.repeat(np.array(seq, float), k)


def envelope(signal, dt, window):
    """Амплітудна обвідна: ковзне RMS * sqrt(2)."""
    k = max(1, int(round(window / dt)))
    w = np.ones(k) / k
    return np.sqrt(2 * np.convolve(signal ** 2, w, mode="same"))


# ---------------------------------------------------------------------------
# 2. Водяне колесо Малкуса
# ---------------------------------------------------------------------------
@njit(cache=True)
def _wheel_rhs(state, N, Q, K, nu, I, gR):
    """state = (theta, omega, m_0..m_{N-1}).  Кут відра i відносно верхньої
    точки: phi_i = theta + 2 pi i / N. Вода ллється у відра біля верху з
    швидкістю Q cos^2(phi) при |phi| < pi/2, витікає зі
    швидкістю K m_i. Момент сили тяжіння: gR sum m_i sin phi_i."""
    th, om = state[0], state[1]
    d = np.empty(N + 2)
    torque = 0.0
    for i in range(N):
        phi = th + 2.0 * np.pi * i / N
        c = np.cos(phi)
        inflow = Q * c * c if c > 0.0 else 0.0
        m = state[2 + i]
        d[2 + i] = inflow - K * m
        torque += m * np.sin(phi)
    d[0] = om
    d[1] = (gR * torque - nu * om) / I
    return d


@njit(cache=True)
def waterwheel(N, T, dt, Q, K, nu, I, gR, omega0=0.1, stride=10):
    """Інтегрує дискретне колесо RK4; повертає (t, theta, omega, masses)."""
    n = int(T / dt)
    n_out = n // stride + 1
    st = np.zeros(N + 2)
    st[1] = omega0
    out_th = np.empty(n_out); out_om = np.empty(n_out); out_m = np.empty((n_out, N))
    out_th[0] = st[0]; out_om[0] = st[1]; out_m[0] = st[2:]
    j = 1
    for k in range(1, n + 1):
        k1 = _wheel_rhs(st, N, Q, K, nu, I, gR)
        k2 = _wheel_rhs(st + 0.5 * dt * k1, N, Q, K, nu, I, gR)
        k3 = _wheel_rhs(st + 0.5 * dt * k2, N, Q, K, nu, I, gR)
        k4 = _wheel_rhs(st + dt * k3, N, Q, K, nu, I, gR)
        st = st + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % stride == 0:
            out_th[j] = st[0]; out_om[j] = st[1]; out_m[j] = st[2:]
            j += 1
    t = np.arange(n_out) * dt * stride
    return t, out_th, out_om, out_m


def _q1_per_unit():
    """Перший косинус-коефіцієнт профілю притоку cos^2(phi)_+ (на одиницю Q)."""
    c = np.cos(2 * np.pi * np.arange(4096) / 4096)
    return 2 * np.mean(np.where(c > 0, c * c, 0.0) * c)


def wheel_params_for(sigma=10.0, rho=28.0, N=24, K=1.0, I=1.0, gR=1.0):
    """Параметри колеса, для яких воно (у границі N -> inf) збігається з
    системою Лоренца (sigma, rho, beta = 1).

    Для перших фур'є-мод маси  a1 = (2/N) sum m_i cos phi_i,  b1 = (2/N) sum m_i sin phi_i:
        a1' = q1 - K a1 - omega b1,     b1' = -K b1 + omega a1,
        omega' = (gR (N/2) b1 - nu omega) / I,
    де q1 — перший косинус-коефіцієнт притоку. Заміна  tau = K t,
        x = omega/K,  y = S b1,  z = r - S a1,   S = gR N / (2 nu K),
    дає рівно Лоренца з  sigma = nu/(I K),  r = S q1 / K  і  beta = 1
    (бо всі моди маси витікають з однаковою швидкістю K)."""
    nu = sigma * I * K
    Q = rho * nu * K ** 2 / (gR * N / 2 * _q1_per_unit())
    return dict(N=N, Q=Q, K=K, nu=nu, I=I, gR=gR)


def wheel_to_lorenz(theta, omega, masses, p):
    """Перерахунок стану колеса в координати Лоренца (x, y, z)."""
    N, K, nu, gR = p["N"], p["K"], p["nu"], p["gR"]
    phis = theta[:, None] + 2 * np.pi * np.arange(N)[None, :] / N
    a1 = (2.0 / N) * np.sum(masses * np.cos(phis), axis=1)
    b1 = (2.0 / N) * np.sum(masses * np.sin(phis), axis=1)
    scale = gR * N / 2 / (nu * K)
    r = scale * p["Q"] * _q1_per_unit() / K
    return np.stack([omega / K, scale * b1, r - scale * a1], axis=1)


# ---------------------------------------------------------------------------
# 3. Reservoir computing (мережа луни)
# ---------------------------------------------------------------------------
class EchoStateNetwork:
    """Резервуар r(t+1) = tanh(A r(t) + W_in u(t)); вихід u = W_out r~,
    де r~ — r з квадратами в парних вузлах (трюк Pathak et al.).
    Тренується лише W_out (ridge-регресія) — мережа «не знає» рівнянь."""

    def __init__(self, n_res=600, degree=3, radius=1.1, input_scale=0.1, ridge=1e-6, seed=0, dim=3):
        rng = np.random.default_rng(seed)
        A = np.zeros((n_res, n_res))
        p = degree / n_res
        mask = rng.random((n_res, n_res)) < p
        A[mask] = rng.uniform(-1, 1, mask.sum())
        A *= radius / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.Win = np.zeros((n_res, dim))
        rows = np.arange(n_res)
        self.Win[rows, rng.integers(0, dim, n_res)] = rng.uniform(-input_scale, input_scale, n_res)
        self.ridge = ridge
        self.n = n_res

    def _feat(self, R):
        F = R.copy()
        F[..., ::2] = F[..., ::2] ** 2
        return F

    def train(self, U, washout=100):
        """U — (T, dim) навчальний ряд (нормований)."""
        r = np.zeros(self.n)
        R = np.empty((len(U), self.n))
        for t in range(len(U)):
            r = np.tanh(self.A @ r + self.Win @ U[t])
            R[t] = r
        X = self._feat(R[washout:-1])
        Y = U[washout + 1:]
        self.Wout = np.linalg.solve(X.T @ X + self.ridge * np.eye(self.n), X.T @ Y).T
        self.r = R[-1]
        fit = X @ self.Wout.T
        return float(np.sqrt(np.mean((fit - Y) ** 2)))

    def listen(self, U):
        """Прогріває резервуар справжніми даними (без навчання) —
        так мережа «синхронізується» з поточним станом системи."""
        r = np.zeros(self.n)
        for u in U:
            r = np.tanh(self.A @ r + self.Win @ u)
        self.r = r

    def predict(self, n_steps):
        """Автономний прогноз: вихід мережі подається їй же на вхід."""
        r = self.r.copy()
        out = np.empty((n_steps, self.Win.shape[1]))
        u = self.Wout @ self._feat(r)
        for t in range(n_steps):
            out[t] = u
            r = np.tanh(self.A @ r + self.Win @ u)
            u = self.Wout @ self._feat(r)
        return out


def valid_time(truth, pred, tol=0.4):
    """Перший індекс, де нормована похибка ||u - u^|| / <||u||^2>^{1/2} > tol."""
    err = np.linalg.norm(truth - pred, axis=1) / np.sqrt(np.mean(np.sum(truth ** 2, axis=1)))
    idx = np.argmax(err > tol)
    return (idx if err[idx] > tol else len(err)), err


# ---------------------------------------------------------------------------
# 4. SINDy
# ---------------------------------------------------------------------------
POLY_NAMES = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]


def poly_library(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    one = np.ones_like(x)
    return np.stack([one, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z], axis=1)


def stlsq(Theta, dX, threshold=0.1, n_iter=10):
    """Послідовна порогова регресія найменших квадратів (Brunton et al. 2016).
    Стовпці нормуються, щоб поріг був однаковим для всіх членів."""
    norms = np.linalg.norm(Theta, axis=0)
    Th = Theta / norms
    Xi = np.linalg.lstsq(Th, dX, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(Xi / norms[:, None]) < threshold
        Xi[small] = 0.0
        for k in range(dX.shape[1]):
            big = ~small[:, k]
            if big.any():
                Xi[big, k] = np.linalg.lstsq(Th[:, big], dX[:, k], rcond=None)[0]
    return Xi / norms[:, None]


def true_coefficients(sigma=SIGMA, rho=RHO, beta=BETA):
    Xi = np.zeros((10, 3))
    Xi[1, 0], Xi[2, 0] = -sigma, sigma
    Xi[1, 1], Xi[2, 1], Xi[6, 1] = rho, -1.0, -1.0
    Xi[3, 2], Xi[5, 2] = -beta, 1.0
    return Xi


def derivative(X, dt, smooth=0):
    """Похідна: центральні різниці 4-го порядку; при smooth>0 —
    фільтр Савицького–Голея (вікно smooth, поліном 3-го ступеня)."""
    if smooth:
        from scipy.signal import savgol_filter
        return savgol_filter(X, smooth, 3, deriv=1, delta=dt, axis=0)
    d = np.empty_like(X)
    d[2:-2] = (-X[4:] + 8 * X[3:-1] - 8 * X[1:-3] + X[:-4]) / (12 * dt)
    d[:2] = (X[1:3] - X[:2]) / dt
    d[-2:] = (X[-2:] - X[-3:-1]) / dt
    return d
