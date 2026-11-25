"""
Повний набір чисельних експериментів. Кожна функція exp_* будує графік(и)
у figures/ і записує числові результати в results/results.json та
results/numbers.tex (макроси \\newcommand для презентації).

Запуск:  python experiments/run_all.py            # усі експерименти
         python experiments/run_all.py lyap bif   # лише вибрані (за підрядком)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lorenzlab import style  # noqa: E402
from lorenzlab.style import NAVY, ORANGE, TEAL, GOLD, GREY, PURPLE, SAND, BUTTERFLY, save  # noqa: E402
from lorenzlab.core import (SIGMA, RHO, BETA, integrate, integrate_f32, advance, rhs, fixed_points,  # noqa: E402
                            eigenvalues_at, hopf_rho, on_attractor)
from lorenzlab.lyapunov import (lyapunov_spectrum, spectrum_vs_rho, ftle_along_trajectory,  # noqa: E402
                                kaplan_yorke)
from lorenzlab.analysis import (z_maxima, bifurcation_zmax, poincare_section, correlation_sum,  # noqa: E402
                                homoclinic_rho, strange_attractor_onset, transient_lifetimes)
from lorenzlab.weather import (predictability_times, ensemble_run, l96_integrate, l96_lyapunov,  # noqa: E402
                               enkf_lorenz63)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_JSON = os.path.join(ROOT, "results", "results.json")
os.makedirs(os.path.dirname(RES_JSON), exist_ok=True)
R = json.load(open(RES_JSON)) if os.path.exists(RES_JSON) else {}

EXPERIMENTS = []


def experiment(fn):
    EXPERIMENTS.append(fn)
    return fn


def colored_line(ax, x, y, c, cmap=BUTTERFLY, lw=0.6, alpha=1.0):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=lw, alpha=alpha)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    return lc


S0 = np.array([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
@experiment
def exp_attractor():
    """3D-атрактор, проєкції, часові ряди, фон для титулки."""
    dt = 0.005
    s = on_attractor()
    tr = integrate(s, dt, 40000)
    t = np.arange(len(tr)) * dt

    # 3D
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(projection="3d")
    n = len(tr)
    for i in range(0, n - 1, 200):
        seg = tr[i:i + 201]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=BUTTERFLY(i / n), lw=0.45)
    for p in fixed_points():
        ax.scatter(*p, color=ORANGE, s=22, depthshade=False, zorder=10)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.view_init(elev=18, azim=-58)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    save(fig, "attractor_3d", png=True)

    # фон титульного слайда (темний)
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#0B132B")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="#0B132B")
    tr2 = integrate(s, 0.004, 120000)
    n2 = len(tr2)
    for i in range(0, n2 - 1, 300):
        seg = tr2[i:i + 301]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=plt.cm.magma(0.25 + 0.7 * i / n2), lw=0.35, alpha=0.75)
    ax.view_init(elev=12, azim=-62)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.9), zoom=1.25)
    fig.savefig(os.path.join(style.FIG_DIR, "title_bg.png"), facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)

    # проєкції
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (i, j, lab) in zip(axs, [(0, 1, ("x", "y")), (0, 2, ("x", "z")), (1, 2, ("y", "z"))]):
        colored_line(ax, tr[:, i], tr[:, j], t, lw=0.35)
        ax.set_xlabel(f"${lab[0]}$"); ax.set_ylabel(f"${lab[1]}$")
        for p in fixed_points():
            ax.plot(p[i], p[j], "o", color=ORANGE, ms=4)
    fig.tight_layout()
    save(fig, "projections", png=True)

    # часові ряди
    fig, axs = plt.subplots(2, 1, figsize=(10, 3.8), sharex=True)
    m = t <= 50
    axs[0].plot(t[m], tr[m, 0], color=NAVY, lw=0.9); axs[0].set_ylabel("$x(t)$")
    axs[0].axhline(0, color=GREY, lw=0.6, ls="--")
    axs[1].plot(t[m], tr[m, 2], color=ORANGE, lw=0.9); axs[1].set_ylabel("$z(t)$")
    axs[1].set_xlabel("$t$")
    fig.tight_layout()
    save(fig, "timeseries")


# ---------------------------------------------------------------------------
@experiment
def exp_regimes():
    """Режими при різних r: від стійкого спокою до хаосу."""
    cases = [(0.5, "r = 0.5: глоб. стійкий 0"), (10, "r = 10: стійкі $C_\\pm$"),
             (16, "r = 16: перехідний хаос"), (24.5, "r = 24.5: бістабільність"),
             (28, "r = 28: дивний атрактор"), (350, "r = 350: граничний цикл")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (r, title) in zip(axs.flat, cases):
        dt = 0.002 if r > 100 else 0.005
        if r == 24.5:
            a = integrate(fixed_points(SIGMA, r, BETA)[1] + np.array([1.0, 1.0, 1.0]), dt, 30000, SIGMA, r, BETA)
            b = integrate(np.array([-0.1, 0.1, 0.0]) + on_attractor(rho=r), dt, 30000, SIGMA, r, BETA)
            ax.plot(b[:, 0], b[:, 2], color=NAVY, lw=0.25, alpha=0.6, label="хаос")
            ax.plot(a[:, 0], a[:, 2], color=ORANGE, lw=1.4, label="до $C_+$")
            ax.legend(fontsize=8, loc="lower right")
        else:
            s0 = np.array([0.5, 2.0, 1.0]) if r < 1 else np.array([0.1, 0.0, 0.0])
            n = 60000 if r > 100 else 12000
            tr = integrate(s0, dt, n, SIGMA, r, BETA)
            if r > 100:
                tr = tr[len(tr) // 2:]
            colored_line(ax, tr[:, 0], tr[:, 2], np.arange(len(tr)), lw=0.5)
        for p in fixed_points(SIGMA, r, BETA):
            ax.plot(p[0], p[2], "o", color=ORANGE, ms=4, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$z$")
    fig.tight_layout()
    save(fig, "regimes", png=True)


# ---------------------------------------------------------------------------
@experiment
def exp_eigen():
    """Власні значення рівноваг як функції r."""
    rs = np.linspace(0.01, 40, 4000)
    ev0 = np.array([np.sort(eigenvalues_at(np.zeros(3), SIGMA, r, BETA).real) for r in rs])
    evc = []
    for r in rs:
        if r > 1:
            e = eigenvalues_at(fixed_points(SIGMA, r, BETA)[1], SIGMA, r, BETA)
            evc.append(e[np.argsort(-e.real)])
        else:
            evc.append(np.full(3, np.nan, complex))
    evc = np.array(evc)
    rH = hopf_rho()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    for k, c in zip(range(3), [NAVY, TEAL, GOLD]):
        ax.plot(rs, ev0[:, k], color=c, lw=1.6)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(1, color=ORANGE, ls="--", lw=1)
    ax.text(1.4, 8, "вилкова\nбіфуркація\n$r=1$", color=ORANGE, fontsize=9)
    ax.set_ylim(-25, 15); ax.set_xlabel("$r$"); ax.set_ylabel("Re $\\lambda$")
    ax.set_title("Початок координат $O$")
    ax = axs[1]
    ax.plot(rs, evc[:, 0].real, color=ORANGE, lw=1.8, label="Re $\\lambda_{1,2}$")
    ax.plot(rs, evc[:, 2].real, color=NAVY, lw=1.6, label="$\\lambda_3$")
    ax.plot(rs, np.abs(evc[:, 0].imag), color=TEAL, lw=1.2, ls=":", label="|Im $\\lambda_{1,2}$|")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(rH, color=ORANGE, ls="--", lw=1)
    ax.text(rH + 0.5, -10, f"Хопф\n$r_H={rH:.3f}$", color=ORANGE, fontsize=9)
    ax.axvline(1.3456, color=GREY, ls=":", lw=1)
    ax.set_ylim(-16, 12); ax.set_xlabel("$r$"); ax.set_title("Рівноваги $C_\\pm$")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "eigenvalues")
    e28 = eigenvalues_at(fixed_points()[1])
    e0 = eigenvalues_at(np.zeros(3))
    R["eig_C_28"] = [str(np.round(v, 4)) for v in e28]
    R["eig_O_28"] = sorted([float(v.real) for v in e0])
    R["r_hopf"] = rH


# ---------------------------------------------------------------------------
@experiment
def exp_global_bifurcations():
    """Власні обчислення r_hom ≈ 13.926 і r_A ≈ 24.06 та часу життя транзієнтів."""
    t0 = time.time()
    r_hom = homoclinic_rho(tol=1e-7)
    lo, hi = strange_attractor_onset(tol=5e-4, T=6000.0)
    R["r_hom"] = r_hom
    R["r_A"] = [lo, hi]
    print(f"  r_hom = {r_hom:.5f}; r_A in [{lo:.4f}, {hi:.4f}]  ({time.time() - t0:.1f}s)")

    # гомоклінічна петля: нестійкий многовид O при r < r_hom, = r_hom, > r_hom
    from lorenzlab.analysis import _unstable_dir
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, r, ttl in zip(axs, [12.0, r_hom, 16.0], ["$r<r_{hom}$", "$r\\approx r_{hom}$", "$r>r_{hom}$"]):
        for sgn, c in [(1, ORANGE), (-1, NAVY)]:
            v = sgn * _unstable_dir(r)
            tr = integrate(1e-7 * v, 0.001, 20000, SIGMA, r, BETA)
            if r == r_hom:   # обрізаємо на першому поверненні до O
                nrm = np.linalg.norm(tr, axis=1)
                far = np.argmax(nrm > 5)
                back = far + np.argmax(nrm[far:] < 0.05)
                tr = tr[:back + 1]
            ax.plot(tr[:, 0], tr[:, 2], color=c, lw=1.0)
        for p in fixed_points(SIGMA, r, BETA):
            ax.plot(p[0], p[2], "o", color=TEAL, ms=4)
        ax.set_title(ttl + f"  (r = {r:.3f})", fontsize=10); ax.set_xlabel("$x$")
    axs[0].set_ylabel("$z$")
    fig.tight_layout()
    save(fig, "homoclinic")

    # час життя хаотичного транзієнта.
    # Час життя розподілений експоненційно; траєкторії, що не вийшли за T_max,
    # цензуровані -> оцінка максимальної правдоподібності
    #   <tau> = (сума спостережених часів + n_cens * T_max) / n_вийшли
    rng = np.random.default_rng(3)
    rs = np.array([15, 16, 17, 18, 19, 20, 21, 21.5, 22, 22.5, 23, 23.3, 23.6, 23.8, 23.9])
    Tmax = 20000.0
    taus, cens = [], []
    for r in rs:
        starts = rng.uniform([-15, -20, 5], [15, 20, 40], size=(400, 3))
        L = transient_lifetimes(r, starts, 0.005, int(Tmax / 0.005))
        fin = np.isfinite(L)
        taus.append((L[fin].sum() + (~fin).sum() * Tmax) / max(fin.sum(), 1))
        cens.append(float((~fin).mean()))
    taus = np.array(taus)
    R["transient_censored"] = dict(zip(map(str, rs), cens))
    rA = 0.5 * (lo + hi)
    m = (rs >= 21) & (np.array(cens) < 0.2)
    fit = linregress(np.log(rA - rs[m]), np.log(taus[m]))
    R["transient_gamma"] = -fit.slope
    R["transient_gamma_err"] = fit.stderr
    print("  tau:", dict(zip(rs, np.round(taus, 1))), "censored:", cens)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].semilogy(rs, taus, "o-", color=NAVY)
    axs[0].axvline(rA, color=ORANGE, ls="--"); axs[0].text(rA - 3.5, taus.max() * 0.6, "$r_A\\approx 24.06$", color=ORANGE)
    axs[0].set_xlabel("$r$"); axs[0].set_ylabel("середній час життя $\\langle\\tau\\rangle$")
    axs[1].loglog(rA - rs, taus, "o", color=NAVY)
    xx = np.logspace(np.log10((rA - rs[m]).min()), np.log10((rA - rs[m]).max()), 50)
    axs[1].loglog(xx, np.exp(fit.intercept) * xx ** fit.slope, color=ORANGE,
                  label=f"$\\langle\\tau\\rangle \\propto (r_A-r)^{{-{-fit.slope:.2f}}}$")
    axs[1].set_xlabel("$r_A - r$"); axs[1].legend()
    fig.tight_layout()
    save(fig, "transient_chaos")


# ---------------------------------------------------------------------------
@experiment
def exp_lyapunov():
    """Спектр Ляпунова при r=28, збіжність, розмірність Каплана–Йорке."""
    dt = 0.005
    n = 2_000_000          # T = 10^4
    lam, hist = lyapunov_spectrum(S0, dt, 20000, n, SIGMA, RHO, BETA, 1, 200)
    # оцінка похибки: розкид по 8 незалежних стартах (T=2000 кожен)
    rng = np.random.default_rng(0)
    reps = np.array([lyapunov_spectrum(rng.normal(size=3) * 5 + np.array([0, 0, 20.0]), dt, 20000, 400000,
                                       SIGMA, RHO, BETA, 1, 400000)[0] for _ in range(8)])
    err = reps.std(axis=0, ddof=1) / np.sqrt(len(reps))
    dky = kaplan_yorke(lam)
    R["lyap"] = lam.tolist(); R["lyap_err"] = err.tolist(); R["lyap_sum"] = float(lam.sum())
    R["D_KY"] = dky
    print("  lambda =", lam, "+-", err, " sum =", lam.sum(), " D_KY =", dky)

    t = (np.arange(len(hist)) + 1) * 200 * dt
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, (ax, c, ref) in enumerate(zip(axs, [ORANGE, TEAL, NAVY], lam)):
        ax.semilogx(t, hist[:, k], color=c, lw=1.2)
        ax.axhline(ref, color="k", lw=0.6, ls="--")
        ax.set_title(f"$\\lambda_{k + 1} \\to {ref:.4f}$")
        ax.set_xlabel("$t$")
    axs[0].set_ylim(0.6, 1.2); axs[1].set_ylim(-0.1, 0.1); axs[2].set_ylim(-15, -14)
    fig.tight_layout()
    save(fig, "lyapunov_convergence")


# ---------------------------------------------------------------------------
@experiment
def exp_lyap_vs_r():
    """Біфуркаційна діаграма (максимуми z) + λ1(r)."""
    t0 = time.time()
    rs = np.linspace(20, 320, 2400)
    Z = bifurcation_zmax(rs, 0.005, 60000, 120000, 120)
    rs2 = np.linspace(20, 320, 900)
    L = spectrum_vs_rho(rs2, 0.005, 40000, 200000)
    print(f"  bifurcation+LE: {time.time() - t0:.1f}s")
    np.savez_compressed(os.path.join(ROOT, "results", "bifurcation.npz"), rs=rs, Z=Z, rs2=rs2, L=L)

    fig, axs = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    rr = np.repeat(rs, Z.shape[1]); zz = Z.ravel()
    ok = np.isfinite(zz)
    axs[0].scatter(rr[ok], (zz[ok] / rr[ok]), s=0.05, color=NAVY, alpha=0.5, lw=0, rasterized=True)
    axs[0].set_ylabel("$z_{\\max}/r$")
    axs[1].plot(rs2, L[:, 0], color=ORANGE, lw=0.8, label="$\\lambda_1$")
    axs[1].plot(rs2, L[:, 1], color=TEAL, lw=0.6, label="$\\lambda_2$")
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel("$\\lambda$"); axs[1].set_xlabel("$r$"); axs[1].legend(loc="upper left", ncol=2)
    for w, lbl in [(99.6, "99.6"), (145.9, "146"), (166.1, "166"), (214.4, "214"), (313, "313")]:
        for ax in axs:
            ax.axvline(w, color=GREY, lw=0.5, ls=":")
        axs[0].text(w, axs[0].get_ylim()[1] if False else 1.02, lbl, fontsize=8, ha="center", color=GREY,
                    transform=axs[0].get_xaxis_transform())
    fig.tight_layout()
    save(fig, "bifurcation", png=True)

    # вікна періодичності: λ1 ≤ 0.01 при r > 30
    per = rs2[(L[:, 0] < 0.01) & (rs2 > 30)]
    R["periodic_r_samples"] = per.tolist()


# ---------------------------------------------------------------------------
@experiment
def exp_lorenz_map():
    """Відображення Лоренца z_{n+1}=f(z_n) і перетин Пуанкаре."""
    zm = z_maxima(S0, 0.002, 50000, 5_000_000)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot(zm[:-1], zm[1:], ".", ms=1.2, color=NAVY)
    lo, hi = zm.min() - 1, zm.max() + 1
    axs[0].plot([lo, hi], [lo, hi], color=ORANGE, lw=1, ls="--", label="$z_{n+1}=z_n$")
    axs[0].set_xlabel("$z_n$"); axs[0].set_ylabel("$z_{n+1}$"); axs[0].legend(loc="upper left")
    axs[0].set_title("Відображення Лоренца")
    # оцінка нахилу: локальні похідні вздовж гілок
    order = np.argsort(zm[:-1])
    xs, ys = zm[:-1][order], zm[1:][order]
    bins = np.linspace(xs.min(), xs.max(), 60)
    slopes, centers = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        mm = (xs >= a) & (xs < b)
        if mm.sum() > 30:
            slopes.append(linregress(xs[mm], ys[mm]).slope); centers.append(0.5 * (a + b))
    slopes = np.array(slopes)
    axs[1].plot(centers, np.abs(slopes), "o-", color=TEAL, ms=3)
    axs[1].axhline(1, color=ORANGE, ls="--"); axs[1].set_yscale("log")
    axs[1].set_xlabel("$z_n$"); axs[1].set_ylabel("$|f'(z_n)|$")
    axs[1].set_title("Розтяг: $|f'|>1$ всюди")
    fig.tight_layout()
    save(fig, "lorenz_map", png=True)
    R["lorenz_map_min_slope"] = float(np.abs(slopes).min())
    # перетин Пуанкаре z = r-1
    P = poincare_section(S0, 0.002, 50000, 5_000_000, RHO - 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(P[:, 0], P[:, 1], ".", ms=0.8, color=NAVY)
    c = np.sqrt(BETA * (RHO - 1))
    ax.plot([c, -c], [c, -c], "o", color=ORANGE, ms=5)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("Перетин Пуанкаре $z=r-1$, $\\dot z>0$")
    save(fig, "poincare", png=True)

    # символьна динаміка: кількість обертів навколо C+ / C- (погодні "режими")
    tr = integrate(on_attractor(), 0.005, 4_000_000, stride=2)
    sgn = np.sign(tr[:, 0])
    runs = []
    cur, length = sgn[0], 0
    # кількість обертів у "крилі" = кількість максимумів |x|... рахуємо максимуми z на одному крилі
    z = tr[:, 2]
    is_max = (z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])
    side = sgn[1:-1][is_max]
    for s_ in side:
        if s_ == cur:
            length += 1
        else:
            if length > 0:
                runs.append(length)
            cur, length = s_, 1
    runs = np.array(runs[1:])
    R["regime_mean_loops"] = float(runs.mean())
    R["regime_max_loops"] = int(runs.max())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    k = np.arange(1, runs.max() + 1)
    freq = np.array([(runs == kk).mean() for kk in k])
    ax.bar(k, freq, color=NAVY, width=0.7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_yscale("log"); ax.set_xlabel("кількість обертів у крилі до перемикання")
    ax.set_ylabel("частота")
    ax.set_title(f"Тривалість «режиму»: середнє {runs.mean():.2f} обертів")
    save(fig, "regime_residence")


# ---------------------------------------------------------------------------
@experiment
def exp_dimension():
    """Кореляційна розмірність Грассбергера–Прокачча."""
    tr = integrate(on_attractor(), 0.01, 1_000_000, stride=20)[1:]
    eps = np.logspace(-1.3, 1.3, 30)
    C = correlation_sum(tr, eps, n_ref=6000)
    le, lc = np.log(eps), np.log(C)
    slope_local = np.gradient(lc, le)
    m = (eps > 0.3) & (eps < 3.0)
    fit = linregress(le[m], lc[m])
    R["D2"] = fit.slope; R["D2_err"] = fit.stderr
    print(f"  D2 = {fit.slope:.3f} ± {fit.stderr:.3f}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].loglog(eps, C, "o", color=NAVY, ms=4)
    axs[0].loglog(eps[m], np.exp(fit.intercept) * eps[m] ** fit.slope, color=ORANGE, lw=2,
                  label=f"нахил $D_2={fit.slope:.3f}$")
    axs[0].set_xlabel("$\\varepsilon$"); axs[0].set_ylabel("$C(\\varepsilon)$"); axs[0].legend()
    axs[1].semilogx(eps, slope_local, "o-", color=TEAL, ms=4)
    axs[1].axhline(fit.slope, color=ORANGE, ls="--")
    axs[1].axhline(R.get("D_KY", 2.06), color=PURPLE, ls=":", label="$D_{KY}$")
    axs[1].axvspan(0.3, 3.0, color=SAND, alpha=0.25)
    axs[1].set_ylim(1, 3.2); axs[1].set_xlabel("$\\varepsilon$")
    axs[1].set_ylabel("$d\\ln C/d\\ln\\varepsilon$"); axs[1].legend()
    fig.tight_layout()
    save(fig, "correlation_dimension")


# ---------------------------------------------------------------------------
@experiment
def exp_spectrum():
    """Спектр потужності x(t), z(t)."""
    dt = 0.01
    tr = integrate(on_attractor(), dt, 2_000_000)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, c, lab in [(0, NAVY, "$x$"), (2, ORANGE, "$z$")]:
        f, p = welch(tr[:, k] - tr[:, k].mean(), fs=1 / dt, nperseg=2 ** 14)
        ax.loglog(f[1:], p[1:], color=c, lw=0.9, label=lab)
    ax.set_xlabel("частота $f$"); ax.set_ylabel("спектральна густина"); ax.legend()
    ax.set_title("Неперервний (широкосмуговий) спектр — ознака неперіодичності")
    save(fig, "power_spectrum")


# ---------------------------------------------------------------------------
@experiment
def exp_butterfly():
    """Чутлива залежність: дві близькі траєкторії; ріст відстані."""
    lam1 = R.get("lyap", [0.9056])[0]
    dt = 0.002
    a0 = on_attractor()
    b0 = a0 + np.array([1e-9, 0, 0])
    n = int(40 / dt)
    A = integrate(a0, dt, n); B = integrate(b0, dt, n)
    t = np.arange(n + 1) * dt
    d = np.linalg.norm(A - B, axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(t, A[:, 0], color=NAVY, lw=1, label="$x(0)=x_0$")
    axs[0].plot(t, B[:, 0], color=ORANGE, lw=1, ls="--", label="$x(0)=x_0+10^{-9}$")
    axs[0].set_ylabel("$x(t)$"); axs[0].legend(ncol=2, loc="upper left")
    axs[1].semilogy(t, d, color=TEAL, lw=0.9, label="$\\|\\delta(t)\\|$")
    axs[1].semilogy(t, 1e-9 * np.exp(lam1 * t), color=ORANGE, ls="--", lw=1.2,
                    label=f"$10^{{-9}} e^{{\\lambda_1 t}}$, $\\lambda_1={lam1:.3f}$")
    axs[1].set_ylim(1e-11, 1e3); axs[1].set_xlabel("$t$"); axs[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "butterfly")
    R["butterfly_div_time"] = float(t[np.argmax(d > 1)])

    # еволюція "краплі" початкових умов: розтяг і згин
    rng = np.random.default_rng(1)
    blob = a0 + 1e-3 * rng.normal(size=(3000, 3))
    E = ensemble_run(blob, 0.002, int(16 / 0.002), int(0.5 / 0.002))
    times = [0, 4, 8, 14, 20, 32]
    bg = integrate(a0, 0.005, 6000)
    fig, axs = plt.subplots(1, 6, figsize=(13, 2.7), sharey=True)
    for ax, k in zip(axs, times):
        ax.plot(bg[:, 0], bg[:, 2], color="#cccccc", lw=0.3)
        ax.plot(E[:, k, 0], E[:, k, 2], ".", ms=0.8, color=ORANGE)
        ax.set_title(f"t = {k * 0.5:g}", fontsize=10); ax.set_xticks([])
    axs[0].set_ylabel("$z$")
    fig.tight_layout()
    save(fig, "blob", png=True)


# ---------------------------------------------------------------------------
@experiment
def exp_lorenz_1961():
    """Реконструкція спостереження Лоренца: перезапуск з округлених до 3 знаків даних."""
    dt = 0.001
    a0 = on_attractor()
    n1 = int(10 / dt)
    A = integrate(a0, dt, n1 + int(25 / dt))
    restart = np.round(A[n1], 3)
    B = integrate(restart, dt, int(25 / dt))
    t = np.arange(len(A)) * dt
    tb = 10 + np.arange(len(B)) * dt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, A[:, 0], color=NAVY, lw=1, label="«оригінальний» розрахунок")
    ax.plot(tb, B[:, 0], color=ORANGE, lw=1, ls="--", label="перезапуск з округлених до $10^{-3}$ даних")
    ax.axvline(10, color=GREY, ls=":")
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$"); ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax.set_ylim(-25, 25)
    save(fig, "lorenz1961")
    R["round_err"] = float(np.linalg.norm(A[n1] - restart))


# ---------------------------------------------------------------------------
@experiment
def exp_ftle():
    """Локальна передбачуваність: скінченночасовий показник λ_T на атракторі."""
    dt = 0.005
    pts, ft = ftle_along_trajectory(on_attractor(), dt, 4000, 600000, int(1.0 / dt))
    pts, ft = pts[::10], ft[::10]
    order = np.argsort(ft)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
    sc = axs[0].scatter(pts[order, 0], pts[order, 2], c=ft[order], s=0.6, cmap="coolwarm",
                        vmin=-3, vmax=5, rasterized=True)
    fig.colorbar(sc, ax=axs[0], label="$\\lambda_{T=1}$")
    axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$z$")
    axs[0].set_title("Де атмосфера-іграшка «непередбачувана»")
    axs[1].hist(ft, bins=120, color=NAVY, density=True)
    axs[1].axvline(ft.mean(), color=ORANGE, ls="--", label=f"середнє {ft.mean():.3f}")
    axs[1].set_xlabel("$\\lambda_{T=1}$"); axs[1].legend()
    axs[1].set_title(f"частка $\\lambda_T<0$: {np.mean(ft < 0) * 100:.0f}%")
    fig.tight_layout()
    save(fig, "ftle", png=True)
    R["ftle_negative_frac"] = float(np.mean(ft < 0))
    R["ftle_mean"] = float(ft.mean())


# ---------------------------------------------------------------------------
@experiment
def exp_numerics():
    """Збіжність RK4, порівняння Ейлер/RK4/DOP853 та float32 vs float64."""
    s0 = np.array([1.0, 1.0, 20.0]); T = 1.0
    ref = solve_ivp(lambda t, u: rhs(u, SIGMA, RHO, BETA), (0, T), s0, method="DOP853",
                    rtol=1e-13, atol=1e-13).y[:, -1]
    hs, e4, e1 = [], [], []
    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
        hs.append(T / n)
        e4.append(np.linalg.norm(integrate(s0, T / n, n)[-1] - ref))
        e1.append(np.linalg.norm(integrate(s0, T / n, n, method=1)[-1] - ref))
    hs, e4, e1 = map(np.array, (hs, e4, e1))
    m = (e4 > 1e-12)
    p4 = linregress(np.log(hs[m][2:]), np.log(e4[m][2:])).slope
    p1 = linregress(np.log(hs[2:]), np.log(e1[2:])).slope
    R["order_rk4"] = p4; R["order_euler"] = p1

    Tlong = 70
    lam1 = R.get("lyap", [0.9056])[0]
    hs_list = [0.02, 0.01, 0.005, 0.0025, 0.001]
    rng = np.random.default_rng(5)
    Tmat = []          # [start, h] + float32
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    for kstart in range(20):
        a0 = advance(on_attractor(), 0.01, int(rng.integers(100, 30000)))
        ref = solve_ivp(lambda t_, u: rhs(u, SIGMA, RHO, BETA), (0, Tlong), a0, method="DOP853",
                        rtol=1e-13, atol=1e-13, dense_output=True)
        row = []
        for h in hs_list:
            n = int(round(Tlong / h))
            tr = integrate(a0, h, n)
            tt = np.arange(n + 1) * h
            err = np.linalg.norm(tr - ref.sol(tt).T, axis=1)
            row.append(tt[np.argmax(err > 1)] if np.any(err > 1) else np.nan)
            if kstart == 0 and h in (0.01, 0.0025, 0.001):
                axs[1].semilogy(tt, err, lw=0.8, label=f"RK4, h={h:g}",
                                color={0.01: NAVY, 0.0025: TEAL, 0.001: ORANGE}[h])
        n = int(round(Tlong / 0.001))
        tr32 = integrate_f32(a0, 0.001, n, SIGMA, RHO, BETA).astype(float)
        tt = np.arange(n + 1) * 0.001
        err = np.linalg.norm(tr32 - ref.sol(tt).T, axis=1)
        row.append(tt[np.argmax(err > 1)] if np.any(err > 1) else np.nan)
        if kstart == 0:
            axs[1].semilogy(tt, err, color=PURPLE, lw=0.8, label="float32, h=0.001")
        Tmat.append(row)
    Tmat = np.array(Tmat)
    Tmean = np.nanmean(Tmat, axis=0)
    fitT = linregress(np.log(1 / np.array(hs_list)), Tmean[:-1])
    R["numerics_T"] = {str(h): float(v) for h, v in zip(hs_list, Tmean[:-1])}
    R["numerics_T"]["f32"] = float(Tmean[-1])
    R["numerics_slope"] = fitT.slope
    R["numerics_slope_theory"] = 4 / lam1
    axs[1].set_xlabel("$t$"); axs[1].set_ylabel("відхилення від еталону DOP853")
    axs[1].set_ylim(1e-14, 1e2); axs[1].legend(fontsize=8, loc="lower right")
    axs[1].set_title("один старт", fontsize=10)
    x_ = np.log(1 / np.array(hs_list))
    axs[2].errorbar(x_, Tmean[:-1], yerr=np.nanstd(Tmat, axis=0)[:-1] / np.sqrt(len(Tmat)), fmt="o",
                    color=NAVY, capsize=3, label="середнє по 20 стартах")
    axs[2].plot(x_, fitT.intercept + fitT.slope * x_, color=ORANGE,
                label=f"нахил {fitT.slope:.2f} (теорія $4/\\lambda_1$={4 / lam1:.2f})")
    axs[2].axhline(Tmean[-1], color=PURPLE, ls=":", label="float32, h=0.001")
    axs[2].set_xlabel("$\\ln(1/h)$"); axs[2].set_ylabel("$T_{num}$: похибка $<1$")
    axs[2].legend(fontsize=8)
    axs[0].loglog(hs, e1, "s-", color=GREY, label=f"Ейлер, нахил {p1:.2f}")
    axs[0].loglog(hs, e4, "o-", color=NAVY, label=f"RK4, нахил {p4:.2f}")
    axs[0].set_xlabel("крок $h$"); axs[0].set_ylabel("глобальна похибка при $t=1$")
    axs[0].legend(fontsize=8)
    axs[0].set_title("порядок збіжності", fontsize=10)
    fig.tight_layout()
    save(fig, "numerics")


# ---------------------------------------------------------------------------
@experiment
def exp_predictability():
    """Горизонт передбачуваності T(δ0) та крива росту похибки."""
    rng = np.random.default_rng(7)
    starts = np.array([advance(on_attractor(), 0.01, int(rng.integers(100, 20000))) for _ in range(300)])
    d0 = np.logspace(-14, -1, 14)
    tol = 1.0
    T = predictability_times(starts, d0, tol, 0.005, 200.0)
    R["pred_censored"] = float(np.mean(~np.isfinite(T)))
    T = np.where(np.isfinite(T), T, np.nan)
    Tm = np.nanmean(T, axis=0)
    Ts = np.nanstd(T, axis=0)
    fit = linregress(np.log(1 / d0), Tm)
    R["pred_slope"] = fit.slope; R["pred_lambda_eff"] = 1 / fit.slope
    R["pred_gain_per_decade"] = fit.slope * np.log(10)
    print(f"  dT/dln(1/d0) = {fit.slope:.3f} -> lambda_eff={1 / fit.slope:.3f}")
    lam1 = R.get("lyap", [0.9056])[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].errorbar(d0, Tm, yerr=Ts, fmt="o", color=NAVY, ms=4, capsize=2, label="середнє ± ст. відх. (300 станів)")
    axs[0].semilogx(d0, fit.intercept + fit.slope * np.log(1 / d0), color=ORANGE,
                    label=f"$T \\approx {fit.intercept:.2f} + {fit.slope:.3f}\\,\\ln(1/\\delta_0)$")
    axs[0].semilogx(d0, np.log(tol / d0) / lam1, color=TEAL, ls="--", label="$\\frac{1}{\\lambda_1}\\ln(\\Delta/\\delta_0)$")
    axs[0].set_xlabel("початкова похибка $\\delta_0$"); axs[0].set_ylabel("горизонт $T_p$ (до $\\|\\delta\\|=1$)")
    axs[0].legend(fontsize=8)

    # середня крива росту похибки (як у чисельних прогнозах погоди)
    N = 400
    st = starts[:N] if len(starts) >= N else np.vstack([starts, starts])[:N]
    dt, stride, n = 0.005, 10, int(30 / 0.005)
    rngd = rng.normal(size=(len(st), 3)); rngd /= np.linalg.norm(rngd, axis=1)[:, None]
    A = ensemble_run(st, dt, n, stride)
    B = ensemble_run(st + 1e-8 * rngd, dt, n, stride)
    t = np.arange(A.shape[1]) * dt * stride
    E = np.sqrt(np.mean(np.sum((A - B) ** 2, axis=2), axis=0))
    lnE = np.mean(np.log(np.linalg.norm(A - B, axis=2) + 1e-300), axis=0)
    Einf = np.sqrt(2 * np.mean(np.sum((A[:, -1] - A[:, -1].mean(0)) ** 2, axis=1)))
    R["E_inf"] = float(Einf)
    axs[1].semilogy(t, E, color=NAVY, lw=1.5, label="СКВ похибки (ансамбль 300)")
    axs[1].semilogy(t, np.exp(lnE), color=TEAL, lw=1, label="$\\exp\\langle\\ln\\|\\delta\\|\\rangle$")
    axs[1].semilogy(t, 1e-8 * np.exp(lam1 * t), color=ORANGE, ls="--", label="$\\delta_0 e^{\\lambda_1 t}$")
    axs[1].axhline(Einf, color=GREY, ls=":", label="насичення $\\sqrt{2}\\,\\sigma_{клім}$")
    axs[1].set_ylim(1e-9, 1e2); axs[1].set_xlabel("$t$"); axs[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "predictability")
    R["doubling_time"] = float(np.log(2) / lam1)


# ---------------------------------------------------------------------------
@experiment
def exp_ensemble():
    """Ансамблевий прогноз: «спагеті», розкид, імовірність режиму."""
    rng = np.random.default_rng(11)
    dt = 0.005
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col", gridspec_kw=dict(width_ratios=[2.2, 1]))
    for row, (seed_shift, title) in enumerate([(300, "Стан «легкої» передбачуваності"),
                                              (1730, "Стан «важкої» передбачуваності")]):
        truth0 = advance(on_attractor(), dt, seed_shift * 7)
        ens0 = truth0 + 0.05 * rng.normal(size=(60, 3))
        n = int(12 / dt)
        E = ensemble_run(ens0, dt, n, 4)
        Tr = integrate(truth0, dt, n, stride=4)
        t = np.arange(E.shape[1]) * dt * 4
        ax = axs[row, 0]
        for m in range(E.shape[0]):
            ax.plot(t, E[m, :, 0], color=GREY, lw=0.4, alpha=0.6)
        ax.plot(t, E[:, :, 0].mean(0), color=ORANGE, lw=1.8, label="середнє ансамблю")
        ax.plot(t, Tr[:, 0], color=NAVY, lw=1.4, ls="--", label="«істина»")
        ax.set_ylabel("$x$"); ax.set_title(title, fontsize=10)
        if row == 0:
            ax.legend(ncol=2, fontsize=8, loc="lower left")
        ax2 = axs[row, 1]
        p = (E[:, :, 0] > 0).mean(0)
        ax2.plot(t, p, color=TEAL, lw=1.5)
        ax2.fill_between(t, 0, p, color=TEAL, alpha=0.2)
        ax2.set_ylim(0, 1); ax2.set_ylabel("$P(x>0)$")
    axs[1, 0].set_xlabel("$t$ (час прогнозу)"); axs[1, 1].set_xlabel("$t$")
    fig.tight_layout()
    save(fig, "ensemble")

    # розкид проти похибки середнього (калібрування ансамблю)
    spreads, errs = [], []
    n = int(8 / dt)
    for k in range(150):
        truth0 = advance(on_attractor(), dt, int(rng.integers(100, 20000)))
        an = truth0 + 0.1 * rng.normal(size=3)                   # «аналіз» з похибкою
        ens0 = an + 0.1 * rng.normal(size=(40, 3))
        E = ensemble_run(ens0, dt, n, 20)
        Tr = integrate(truth0, dt, n, stride=20)
        spreads.append(np.sqrt(np.mean(np.sum((E - E.mean(0)) ** 2, axis=2), axis=0)))
        errs.append(np.sqrt(np.sum((E.mean(0) - Tr) ** 2, axis=1)))
    t = np.arange(len(spreads[0])) * dt * 20
    sp_, er_ = np.sqrt(np.mean(np.array(spreads) ** 2, 0)), np.sqrt(np.mean(np.array(errs) ** 2, 0))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.semilogy(t, er_, color=NAVY, lw=1.5, label="СКВ похибки середнього")
    ax.semilogy(t, sp_, color=ORANGE, lw=1.5, ls="--", label="розкид ансамблю")
    ax.set_xlabel("час прогнозу $t$"); ax.legend()
    ax.set_title("Розкид ≈ похибка: ансамбль «знає», наскільки він невпевнений", fontsize=10)
    save(fig, "spread_skill")


# ---------------------------------------------------------------------------
@experiment
def exp_l96():
    """Модель Лоренца-96 як «іграшкова атмосфера»."""
    N, F = 40, 8.0
    rng = np.random.default_rng(0)
    x0 = F * np.ones(N); x0[19] += 0.01
    dt = 0.01
    tr = l96_integrate(x0, dt, 100000, F, stride=5)[2000:]
    tr = tr[:400]
    t = np.arange(len(tr)) * dt * 5
    t0 = time.time()
    lam = l96_lyapunov(l96_integrate(x0, dt, 2000, F)[-1], dt, 5000, 200000, F, N, 5)
    print(f"  L96 Lyapunov: {time.time() - t0:.1f}s; lambda1 = {lam[0]:.3f}, n+ = {(lam > 0).sum()}")
    R["l96_lam1"] = float(lam[0]); R["l96_npos"] = int((lam > 0).sum())
    R["l96_DKY"] = float(kaplan_yorke(lam)); R["l96_doubling"] = float(np.log(2) / lam[0])
    R["l96_ks_entropy"] = float(lam[lam > 0].sum())

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    im = axs[0].imshow(tr.T, aspect="auto", origin="lower", cmap="RdBu_r", extent=[t[0], t[-1], 0, N],
                       vmin=-10, vmax=14)
    fig.colorbar(im, ax=axs[0], label="$X_k$")
    axs[0].set_xlabel("$t$ (1 од. ≈ 5 діб)"); axs[0].set_ylabel("довгота $k$")
    axs[0].set_title("Діаграма Говмьоллера: хвилі, що біжать на захід")
    axs[1].plot(np.arange(1, N + 1), lam, "o-", color=NAVY, ms=3)
    axs[1].axhline(0, color=ORANGE, ls="--")
    axs[1].set_xlabel("номер $i$"); axs[1].set_ylabel("$\\lambda_i$")
    axs[1].set_title(f"Спектр Ляпунова: {(lam > 0).sum()} додатних, $D_{{KY}}\\approx{kaplan_yorke(lam):.1f}$")
    fig.tight_layout()
    save(fig, "lorenz96", png=True)

    # ріст похибки в L96
    xa = tr[-1].copy()
    xb = xa + 1e-6 * rng.normal(size=N)
    A = l96_integrate(xa, dt, 2000, F); B = l96_integrate(xb, dt, 2000, F)
    tt = np.arange(len(A)) * dt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.semilogy(tt, np.sqrt(np.mean((A - B) ** 2, axis=1)), color=NAVY, label="СКВ похибки")
    ax.semilogy(tt, 1e-6 * np.exp(lam[0] * tt) / np.sqrt(1), color=ORANGE, ls="--",
                label=f"$\\propto e^{{\\lambda_1 t}}$, $\\lambda_1={lam[0]:.2f}$")
    ax.set_xlabel("$t$"); ax.legend(); ax.set_title("Лоренц-96: похибка зростає за експонентою")
    save(fig, "l96_error")


# ---------------------------------------------------------------------------
@experiment
def exp_enkf():
    """Асиміляція даних EnKF: як «приборкати» хаос спостереженнями."""
    res = enkf_lorenz63(on_attractor(), 400, 0.10, 0.005, [0, 1, 2], 2.0, n_ens=20)
    res_x = enkf_lorenz63(on_attractor(), 400, 0.10, 0.005, [0], 2.0, n_ens=20, seed=2)
    rm = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=1))
    burn = 50
    R["enkf_rmse"] = float(rm(res["analysis"], res["truth"])[burn:].mean())
    R["enkf_rmse_x"] = float(rm(res_x["analysis"], res_x["truth"])[burn:].mean())
    R["enkf_obs_rmse"] = 2.0
    R["enkf_free_rmse"] = float(rm(res["free"], res["truth"])[burn:].mean())
    print(f"  EnKF RMSE: {R['enkf_rmse']:.3f} (all), {R['enkf_rmse_x']:.3f} (x only), free {R['enkf_free_rmse']:.2f}")
    t = res["t"]
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    m = t <= 20
    axs[0].plot(t[m], res["truth"][m, 0], color=NAVY, lw=1.4, label="істина")
    axs[0].plot(t[m], res["obs"][m, 0], ".", color=GREY, ms=3, label="спостереження ($\\sigma_o=2$)")
    axs[0].plot(t[m], res["analysis"][m, 0], color=ORANGE, lw=1.1, label="аналіз EnKF")
    axs[0].plot(t[m], res["free"][m, 0], color=TEAL, lw=0.9, ls=":", label="прогноз без асиміляції")
    axs[0].set_ylabel("$x$"); axs[0].legend(ncol=4, fontsize=8, loc="lower left")
    axs[1].plot(t, rm(res["analysis"], res["truth"]), color=ORANGE, lw=1, label="RMSE аналізу (спост. x,y,z)")
    axs[1].plot(t, rm(res_x["analysis"], res_x["truth"]), color=PURPLE, lw=0.8, label="RMSE аналізу (спост. лише x)")
    axs[1].plot(t, rm(res["free"], res["truth"]), color=TEAL, lw=0.8, ls=":", label="без асиміляції")
    axs[1].axhline(2.0, color=GREY, ls="--", lw=0.8, label="похибка спостережень")
    axs[1].set_yscale("log"); axs[1].set_xlabel("$t$"); axs[1].set_ylabel("RMSE")
    axs[1].legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "enkf")


# ---------------------------------------------------------------------------
@experiment
def exp_sync():
    """Синхронізація Пекори–Керролла (x-керування)."""
    dt = 0.002
    n = int(15 / dt)

    def F(u):   # u = (x, y, z, y', z')
        x, y, z, yr, zr = u
        return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z,
                         x * (RHO - zr) - yr, x * yr - BETA * zr])
    u = np.concatenate([on_attractor(), [-10.0, 40.0]])
    U = np.empty((n + 1, 5)); U[0] = u
    for i in range(n):        # спільний RK4 для передавача і приймача
        k1 = F(u); k2 = F(u + 0.5 * dt * k1); k3 = F(u + 0.5 * dt * k2); k4 = F(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        U[i + 1] = u
    drive = U[:, :3]; ys = U[:, 3]
    err = np.hypot(U[:, 3] - U[:, 1], U[:, 4] - U[:, 2])
    R["sync_final_err"] = float(err[-1])
    t = np.arange(n + 1) * dt
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3))
    axs[0].plot(t, drive[:, 1], color=NAVY, lw=1, label="$y$ (передавач)")
    axs[0].plot(t, ys, color=ORANGE, lw=1, ls="--", label="$y'$ (приймач)")
    axs[0].set_xlabel("$t$"); axs[0].legend(); axs[0].set_xlim(0, 8)
    axs[1].semilogy(t, err, color=TEAL, lw=1)
    axs[1].semilogy(t, err[0] * np.exp(-t), color=ORANGE, ls="--", label="$e^{-t}$ (оцінка $\\min(1,\\beta)$)")
    axs[1].set_xlabel("$t$"); axs[1].set_ylabel("$\\|(e_y,e_z)\\|$"); axs[1].legend()
    fig.tight_layout()
    save(fig, "sync")


# ---------------------------------------------------------------------------
@experiment
def exp_convection():
    """Поле температури та лінії течії в конвективній комірці для станів атрактора."""
    a = 1 / np.sqrt(2)
    xs = np.linspace(0, 2 / a, 300); zs = np.linspace(0, 1, 120)
    Xg, Zg = np.meshgrid(xs, zs)
    tr = integrate(on_attractor(), 0.01, 3000)
    i1 = np.argmax(tr[:, 0]); i2 = np.argmin(tr[:, 0])
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.6))
    for ax, i in zip(axs, [i1, i2]):
        X, Y, Z = tr[i]
        psi = X * np.sin(np.pi * a * Xg) * np.sin(np.pi * Zg)
        theta = Y * np.cos(np.pi * a * Xg) * np.sin(np.pi * Zg) - Z * np.sin(2 * np.pi * Zg)
        Ttot = (1 - Zg) * RHO * 1.0 + theta     # лінійний профіль + відхилення (умовні одиниці)
        ax.contourf(Xg, Zg, theta, 30, cmap="RdBu_r")
        u = -np.pi * X * np.sin(np.pi * a * Xg) * np.cos(np.pi * Zg)
        w = np.pi * a * X * np.cos(np.pi * a * Xg) * np.sin(np.pi * Zg)
        ax.streamplot(xs, zs, u, w, color="k", density=0.8, linewidth=0.6, arrowsize=0.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$X={X:.1f},\\ Y={Y:.1f},\\ Z={Z:.1f}$", fontsize=10)
    fig.tight_layout()
    save(fig, "convection", png=True)


# ---------------------------------------------------------------------------
def write_numbers_tex():
    """Числові результати -> LaTeX-макроси."""
    f = lambda v, d: f"{v:.{d}f}"
    lines = ["% автоматично згенеровано experiments/run_all.py"]
    def nc(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    if "lyap" in R:
        l = R["lyap"]; e = R["lyap_err"]
        nc("lamOne", f(l[0], 4)); nc("lamTwo", f(l[1], 4)); nc("lamThree", f(l[2], 3))
        nc("lamOneErr", f(e[0], 4)); nc("lamSum", f(R["lyap_sum"], 4)); nc("DKY", f(R["D_KY"], 4))
    if "D2" in R:
        nc("DTwo", f(R["D2"], 3)); nc("DTwoErr", f(R["D2_err"], 3))
    if "r_hom" in R:
        nc("rHom", f(R["r_hom"], 4)); nc("rAlo", f(R["r_A"][0], 4)); nc("rAhi", f(R["r_A"][1], 4))
    if "transient_gamma" in R:
        nc("transGamma", f(R["transient_gamma"], 2)); nc("transGammaErr", f(R.get("transient_gamma_err", 0), 2))
    if "pred_slope" in R:
        nc("predSlope", f(R["pred_slope"], 3)); nc("predLam", f(R["pred_lambda_eff"], 3))
        nc("predGain", f(R["pred_gain_per_decade"], 2)); nc("doublingTime", f(R["doubling_time"], 3))
    if "l96_lam1" in R:
        nc("lnsixLam", f(R["l96_lam1"], 3)); nc("lnsixNpos", str(R["l96_npos"]))
        nc("lnsixDKY", f(R["l96_DKY"], 1)); nc("lnsixDoubling", f(R["l96_doubling"], 2))
        nc("lnsixKS", f(R["l96_ks_entropy"], 2))
    if "enkf_rmse" in R:
        nc("enkfRmse", f(R["enkf_rmse"], 2)); nc("enkfRmseX", f(R["enkf_rmse_x"], 2))
        nc("enkfFree", f(R["enkf_free_rmse"], 1))
    if "order_rk4" in R:
        nc("orderRK", f(R["order_rk4"], 2)); nc("orderEuler", f(R["order_euler"], 2))
        T = R["numerics_T"]
        nc("TnumA", f(T["0.01"], 1)); nc("TnumB", f(T["0.005"], 1)); nc("TnumC", f(T["0.001"], 1))
        nc("Tnumf", f(T["f32"], 1)); nc("TnumSlope", f(R["numerics_slope"], 2))
        nc("TnumSlopeTh", f(R["numerics_slope_theory"], 2))
    if "ftle_negative_frac" in R:
        nc("ftleNeg", f(100 * R["ftle_negative_frac"], 0))
    if "regime_mean_loops" in R:
        nc("regimeMean", f(R["regime_mean_loops"], 2)); nc("regimeMax", str(R["regime_max_loops"]))
    if "lorenz_map_min_slope" in R:
        nc("mapMinSlope", f(R["lorenz_map_min_slope"], 2))
    if "butterfly_div_time" in R:
        nc("butterflyT", f(R["butterfly_div_time"], 1))
    if "round_err" in R:
        nc("roundErr", f"{R['round_err']:.1e}".replace("e-0", "\\cdot 10^{-").replace("e-", "\\cdot 10^{-") + "}")
    with open(os.path.join(ROOT, "results", "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    style.setup()
    sel = sys.argv[1:]
    for fn in EXPERIMENTS:
        if sel and not any(s in fn.__name__ for s in sel):
            continue
        t0 = time.time()
        print(f"[{fn.__name__}] {(fn.__doc__ or '').strip()}", flush=True)
        fn()
        print(f"  done in {time.time() - t0:.1f}s")
        json.dump(R, open(RES_JSON, "w"), indent=2, ensure_ascii=False)
    write_numbers_tex()
    json.dump(R, open(RES_JSON, "w"), indent=2, ensure_ascii=False)
    print("results ->", os.path.relpath(RES_JSON))
