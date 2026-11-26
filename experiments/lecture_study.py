"""
Експерименти для розгорнутого дослідження (report/study.tex): одновимірна редукція
біля r = 1, коректність за Адамаром, (не)оборотність, статистичні тотожності.

-> results/numbers_lecture.tex, figures/lec_*.png
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lorenzlab import core, style  # noqa: E402
from lorenzlab.weather import predictability_times  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

S, R, B = core.SIGMA, core.RHO, core.BETA
A = S / (S + 1)
numbers = {}


def f(t, s, r=R):
    return [S * (s[1] - s[0]), s[0] * (r - s[2]) - s[1], s[0] * s[1] - B * s[2]]


def solve(s0, t_eval, r=R, rtol=1e-12):
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), s0, args=(r,), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=1e-14)
    assert sol.success, sol.message
    return sol.y.T


def v_red(u, mu):
    """Права частина одновимірної редукції u' = v(u), mu = r - 1."""
    return A * u * (mu - u**2 / B)


def u_red(t, u0, mu):
    """Точний розв'язок редукції (логістичне рівняння для w = u^2)."""
    if mu == 0:
        return u0 / np.sqrt(1 + 2 * A * u0**2 * t / B)
    w = mu * B / (1 + (mu * B / u0**2 - 1) * np.exp(-2 * A * mu * t))
    return np.sign(u0) * np.sqrt(w)


def project(s):
    """Координата вздовж критичного напрямку: u = (x + sigma y) / (1 + sigma)."""
    return (s[..., 0] + S * s[..., 1]) / (1 + S)


def fig_phase_lines():
    cases = [(-0.5, "r < 1"), (0.0, "r = 1"), (0.5, "r > 1")]
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.4), gridspec_kw={"width_ratios": [1, 2.6]})
    u = np.linspace(-1.9, 1.9, 400)
    t = np.linspace(0, 12, 400)
    for (mu, name), (axv, axt) in zip(cases, axes):
        axv.plot(v_red(u, mu), u, color=style.TEAL, lw=2)
        axv.axvline(0, color="#444444", lw=0.8)
        axv.invert_xaxis()
        axv.set_xlabel("v(u)")
        axv.set_ylabel("u")
        axv.set_title(name, loc="left")
        eq = [0.0] + ([np.sqrt(B * mu), -np.sqrt(B * mu)] if mu > 0 else [])
        for e in eq:
            stable = not (mu > 0 and e == 0)
            col = style.TEAL if stable else style.ORANGE
            axv.plot(0, e, "o", color=col, ms=7, zorder=3)
            axt.axhline(e, color=col, lw=1.6)
        if mu > 0:
            for e in (np.sqrt(B * mu / 3), -np.sqrt(B * mu / 3)):
                axt.axhline(e, color=style.GOLD, lw=1, ls="--")
        starts = [-1.7, -1.1, 1.1, 1.7] + ([-0.12, -0.02, 0.02, 0.12] if mu > 0 else [-0.5, 0.5])
        for u0 in starts:
            axt.plot(t, u_red(t, u0, mu), color=style.PURPLE, lw=1.2)
        axt.set_ylim(-1.9, 1.9)
        axv.set_ylim(-1.9, 1.9)
        axt.set_xlabel("t")
    fig.tight_layout()
    style.save(fig, "lec_phase_lines", png=True)


def reduction_accuracy():
    """Порівняння редукції з повною тривимірною системою."""
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for mu, col in zip([0.4, 0.2, 0.1, 0.05], [style.NAVY, style.TEAL, style.ORANGE, style.PURPLE]):
        r = 1 + mu
        # старт на критичному напрямку, далеко від кінцевої рівноваги
        u0 = 0.1 * np.sqrt(B * mu)
        t = np.linspace(0, 12 / (A * mu), 1200)
        tr = solve([u0, u0 * (1 + mu / (S + 1)), u0**2 / B], t, r=r)
        full = project(tr)
        # перші 2 од. часу відкидаємо: швидкі моди осідають на центральний многовид
        i0 = np.searchsorted(t, 2.0)
        red = u_red(t[i0:] - t[i0], full[i0], mu)
        err = np.max(np.abs(full[i0:] - red)) / np.sqrt(B * mu)
        lam = (-(S + 1) + np.sqrt((S + 1)**2 + 4 * S * mu)) / 2
        rows.append((mu, err * 100, lam, A * mu))
        ax.plot(t * mu, full / np.sqrt(B * mu), color=col, lw=2, label=f"r = {r:g}")
        ax.plot(t[i0:] * mu, red / np.sqrt(B * mu), color="black", lw=0.9, ls="--")
    ax.set_xlabel("(r − 1) t")
    ax.set_ylabel("u / u*")
    ax.legend(title="суцільна: 3D, пунктир: 1D")
    ax.grid(True)
    style.save(fig, "lec_reduction", png=True)
    table = " \\\\\n".join(f"{m:g} & {e:.2f}\\% & {l:.4f} & {a:.4f}" for m, e, l, a in rows)
    numbers["redTable"] = table + " \\\\"
    numbers["redErrSmall"] = f"{rows[-1][1]:.2f}"


def well_posedness():
    """Неперервна залежність: |delta(t)| <= |delta0| exp(int mu(sym J) dt)."""
    s0 = core.on_attractor()
    d0 = 1e-8
    t = np.linspace(0, 30, 6001)
    # обидві траєкторії інтегруються разом, щоб похибка кроку була спільною
    def pair(_, y):
        return f(0, y[:3]) + f(0, y[3:])
    sol = solve_ivp(pair, (0, 30), np.r_[s0, s0 + [d0, 0, 0]], method="DOP853",
                    t_eval=t, rtol=1e-13, atol=1e-14)
    a, b = sol.y[:3].T, sol.y[3:].T
    dist = np.linalg.norm(a - b, axis=1)
    mu = np.empty(len(t))
    for i, m in enumerate((a + b) / 2):  # для квадратичного поля f(a)-f(b) = J(середина)(a-b)
        J = core.jacobian(m, S, R, B)
        mu[i] = np.linalg.eigvalsh((J + J.T) / 2)[-1]
    integ = np.r_[0, np.cumsum((mu[1:] + mu[:-1]) / 2 * np.diff(t))]
    log_bound = np.log10(dist[0]) + integ / np.log(10)
    assert np.all(np.log10(dist) <= log_bound + 1e-6)
    L = mu.max()
    numbers["Lmax"] = f"{L:.1f}"
    numbers["tBoundUseless"] = f"{t[np.argmax(log_bound > np.log10(40))]:.2f}"
    numbers["tActual"] = f"{t[np.argmax(dist > 1)]:.1f}"
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.semilogy(t, dist, color=style.NAVY, label="фактична відстань")
    keep = log_bound < 4
    ax.semilogy(t[keep], 10.0**log_bound[keep], color=style.ORANGE, label="гарантована оцінка (Гронуолл)")
    ax.semilogy(t, d0 * np.exp(0.9058 * t), color=style.TEAL, ls="--", label=r"$\delta_0 e^{\lambda_1 t}$")
    ax.set_ylim(1e-9, 1e4)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\delta(t)\|$")
    ax.legend()
    ax.grid(True)
    style.save(fig, "lec_gronwall", png=True)


def reversibility():
    """Вперед на T, потім назад на T: чи повертаємось у початковий стан?"""
    s0 = core.on_attractor()
    Ts = np.arange(0.25, 3.01, 0.25)
    errs = []
    for T in Ts:
        end = solve(s0, np.array([0.0, T]), rtol=1e-13)[-1]
        sol = solve_ivp(f, (T, 0.0), end, method="DOP853", rtol=1e-13, atol=1e-14)
        errs.append(np.linalg.norm(sol.y[:, -1] - s0) if sol.success else np.inf)
    errs = np.array(errs)
    ok = np.isfinite(errs) & (errs < 1e-2) & (errs > 0)
    slope = np.polyfit(Ts[ok][2:], np.log(errs[ok][2:]), 1)[0]
    lost = Ts[np.argmax(~(errs < 1.0))] if np.any(~(errs < 1.0)) else None
    numbers["revSlope"] = f"{slope:.1f}"
    numbers["revLost"] = f"{lost:g}" if lost is not None else r"> 3"
    print("  reversibility errors:", dict(zip(Ts.tolist(), [f"{e:.1e}" for e in errs])))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fin = np.isfinite(errs)
    ax.semilogy(Ts[fin], errs[fin], "o-", color=style.NAVY, label="похибка повернення")
    ax.semilogy(Ts[ok], errs[ok][-1] * np.exp(14.572 * (Ts[ok] - Ts[ok][-1])),
                color=style.ORANGE, ls="--", label=r"$\propto e^{|\lambda_3| T}$")
    ax.set_xlabel("T")
    ax.set_ylabel("похибка")
    ax.legend()
    ax.grid(True)
    style.save(fig, "lec_reversibility", png=True)


def statistics():
    """Часові середні: три тотожності та незалежність від початкової умови."""
    dt, n = 0.005, 4_000_000
    res = []
    for s0 in ([1.0, 1.0, 1.0], [-7.0, 3.0, 30.0]):
        s = core.advance(np.array(s0), dt, 10_000)
        tr = core.integrate(s, dt, n, stride=4)
        x, y, z = tr.T
        res.append((np.mean(x * x), np.mean(x * y), B * np.mean(z),
                    np.mean(y * y), B * (R * np.mean(z) - np.mean(z * z)), np.mean(z)))
    a, b = res
    for i, k in enumerate(["avgXX", "avgXY", "avgBZ", "avgYY", "avgRhs", "avgZ"]):
        numbers[k + "A"] = f"{a[i]:.2f}"
        numbers[k + "B"] = f"{b[i]:.2f}"


def horizon():
    """Горизонт передбачуваності для 200 випадкових стартів (світлий варіант графіка)."""
    rng = np.random.default_rng(7)
    starts = np.array([core.advance(core.on_attractor(), 0.01, int(rng.integers(100, 20000)))
                       for _ in range(200)])
    d0 = np.logspace(-14, -1, 14)
    T = predictability_times(starts, d0, 1.0, 0.005, 200.0)
    T = np.where(np.isfinite(T), T, np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.fill_between(d0, np.nanpercentile(T, 10, axis=0), np.nanpercentile(T, 90, axis=0),
                    color=style.TEAL, alpha=0.2, lw=0, label="10–90 % стартів")
    ax.semilogx(d0, np.nanmean(T, axis=0), "o", color=style.NAVY, label="середнє")
    ax.semilogx(d0, np.log(1 / d0) / 0.9058, color=style.ORANGE, ls="--",
                label=r"$\ln(1/\delta_0)/\lambda_1$")
    ax.invert_xaxis()
    ax.set_xlabel(r"початкова похибка $\delta_0$")
    ax.set_ylabel(r"$T_p$")
    ax.legend()
    ax.grid(True)
    style.save(fig, "lec_horizon", png=True)


def main():
    style.setup()
    fig_phase_lines()
    reduction_accuracy()
    well_posedness()
    reversibility()
    statistics()
    horizon()
    with open(os.path.join(ROOT, "results", "numbers_lecture.tex"), "w") as fh:
        fh.write("% автоматично згенеровано experiments/lecture_study.py\n")
        for k, v in numbers.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in numbers.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
