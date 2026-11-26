"""
Графіки для науково-популярної версії презентації: темна палітра карти,
великі шрифти, мінімум підписів. Результат: figures/pop_*.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lorenzlab import style  # noqa: E402
from lorenzlab.core import SIGMA, RHO, BETA, integrate, advance, fixed_points, on_attractor  # noqa: E402
from lorenzlab.weather import ensemble_run, predictability_times, enkf_lorenz63  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
LAM1 = R["lyap"][0]

INK = "#1F2A37"
DIM = "#4F4B42"
GOLD = "#B7791F"
ORANGE = "#B83227"
TEAL = "#2F7A5B"
SAND = "#C8901F"
SKY = "#1F5C99"


# від холодного фронту до теплого
FRONTS = LinearSegmentedColormap.from_list("fronts", [SKY, "#4B3F72", ORANGE])


def setup():
    style.setup()
    plt.rcParams.update({
        "savefig.bbox": "tight", "axes.linewidth": 1.2,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": DIM, "ytick.color": DIM,
        "axes.edgecolor": "#9C937C", "axes.facecolor": "none", "figure.facecolor": "none",
        "savefig.transparent": True, "font.size": 15, "axes.labelsize": 16,
        "legend.fontsize": 13, "axes.titlesize": 15, "axes.titlecolor": INK,
        "savefig.dpi": 250,
    })


def save(fig, name):
    p = os.path.join(style.FIG_DIR, f"pop_{name}.png")
    fig.savefig(p)
    plt.close(fig)
    print("  ->", os.path.relpath(p, ROOT))


def thick(lw):
    """Мінімальна читабельна товщина на проєкторі."""
    return 0.35 + 1.35 * lw


def glow_line(ax, x, y, color, lw=1.6, alpha=1.0, **kw):
    return ax.plot(x, y, color=color, lw=thick(lw), alpha=alpha, **kw)


def gradient_line(ax, x, y, cmap, lw=0.6, alpha=0.9):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=lw, alpha=alpha)
    lc.set_array(np.linspace(0, 1, len(seg)))
    ax.add_collection(lc)
    ax.autoscale()


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
def fig_rounding():
    """Історія 1961: перезапуск з округлених даних."""
    dt = 0.001
    a0 = on_attractor()
    n1 = int(10 / dt)
    A = integrate(a0, dt, n1 + int(25 / dt))
    B = integrate(np.round(A[n1], 3), dt, int(25 / dt))
    t = np.arange(len(A)) * dt
    tb = 10 + np.arange(len(B)) * dt
    fig, ax = plt.subplots(figsize=(11, 3.6))
    glow_line(ax, t, A[:, 0], SKY, lw=2.24)
    glow_line(ax, tb, B[:, 0], ORANGE, lw=2.24, ls=(0, (4, 2)))
    ax.axvline(10, color=DIM, lw=1.43, ls=":")
    ax.text(10.3, 21, "перезапуск з даних,\nокруглених до 0.001", color=DIM, fontsize=12, va="top")
    ax.axvspan(19, 35, color=ORANGE, alpha=0.1)
    ax.text(27, 21, "зовсім інша «погода»", color=ORANGE, fontsize=13, ha="center", va="top")
    ax.set_xlim(0, 35); ax.set_ylim(-24, 24)
    ax.set_xlabel("час"); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "rounding")


def fig_convection():
    """Конвективна комірка."""
    a = 1 / np.sqrt(2)
    xs = np.linspace(0, 2 / a, 400); zs = np.linspace(0, 1, 150)
    Xg, Zg = np.meshgrid(xs, zs)
    X, Y, Z = 9.0, 9.0, 27.0
    theta = Y * np.cos(np.pi * a * Xg) * np.sin(np.pi * Zg) - Z * np.sin(2 * np.pi * Zg)
    T = (1 - Zg) * 28 + 0.5 * theta
    u = -np.pi * X * np.sin(np.pi * a * Xg) * np.cos(np.pi * Zg)
    w = np.pi * a * X * np.cos(np.pi * a * Xg) * np.sin(np.pi * Zg)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.contourf(Xg, Zg, T, 60, cmap="RdBu_r", alpha=0.8)
    ax.streamplot(xs, zs, u, w, color=INK, density=0.9, linewidth=1.0, arrowsize=1.1)
    ax.set_aspect("equal"); clean(ax)
    ax.text(xs[-1] / 2, -0.09, "гаряче дно", ha="center", va="top", color=GOLD, fontsize=13)
    ax.text(xs[-1] / 2, 1.04, "холодна кришка", ha="center", va="bottom", color=SKY, fontsize=13)
    save(fig, "convection")


def fig_attractor():
    """Великий атрактор."""
    s = on_attractor()
    tr = integrate(s, 0.004, 90000)
    fig = plt.figure(figsize=(7, 6.2))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor("none")
    n = len(tr)
    for i in range(0, n - 1, 250):
        seg = tr[i:i + 251]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=FRONTS(i / n), lw=0.82, alpha=0.85)
    for p in fixed_points()[1:]:
        ax.scatter(*p, color=SKY, s=40, depthshade=False)
    ax.view_init(elev=16, azim=-50)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.9), zoom=1.25)
    save(fig, "attractor")


def fig_blob():
    """Розтяг і згин краплі початкових умов."""
    rng = np.random.default_rng(1)
    a0 = on_attractor()
    blob = a0 + 1e-3 * rng.normal(size=(4000, 3))
    E = ensemble_run(blob, 0.002, int(16 / 0.002), int(0.5 / 0.002))
    bg = integrate(a0, 0.005, 8000)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, k in zip(axs, [0, 8, 14, 32]):
        ax.plot(bg[:, 0], bg[:, 2], color="#9C937C", lw=0.89)
        ax.plot(E[:, k, 0], E[:, k, 2], ".", ms=2.4, color=GOLD, alpha=0.9)
        ax.set_title(f"t = {k * 0.5:g}", color=INK, fontsize=15)
        clean(ax)
    save(fig, "blob")


def fig_gallery():
    """Шлях до хаосу: 4 режими."""
    cases = [(10, "r = 10\nспокійна конвекція"), (16, "r = 16\nхаос, що згасає"),
             (24.5, "r = 24.5\nдва світи поруч"), (28, "r = 28\nдивний атрактор")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (r, title) in zip(axs, cases):
        dt = 0.005
        if r == 24.5:
            b = integrate(on_attractor(rho=r), dt, 30000, SIGMA, r, BETA)
            ax.plot(b[:, 0], b[:, 2], color=SKY, lw=0.69, alpha=0.6)
            a = integrate(fixed_points(SIGMA, r, BETA)[1] + 1.0, dt, 30000, SIGMA, r, BETA)
            ax.plot(a[:, 0], a[:, 2], color=ORANGE, lw=1.97)
        elif r == 28:
            tr = integrate(on_attractor(), dt, 20000)
            gradient_line(ax, tr[:, 0], tr[:, 2], FRONTS, lw=0.89)
        else:
            tr = integrate(np.array([0.1, 0.0, 0.0]), dt, 14000, SIGMA, r, BETA)
            gradient_line(ax, tr[:, 0], tr[:, 2], FRONTS, lw=1.56)
        for p in fixed_points(SIGMA, r, BETA)[1:]:
            ax.plot(p[0], p[2], "o", color=SAND, ms=5)
        ax.set_title(title, fontsize=21, color=INK)
        clean(ax)
    fig.tight_layout()
    save(fig, "gallery")


def fig_butterfly():
    """Дві траєкторії, що відрізняються на 1e-9."""
    dt = 0.002
    a0 = on_attractor()
    n = int(40 / dt)
    A = integrate(a0, dt, n); B = integrate(a0 + np.array([1e-9, 0, 0]), dt, n)
    t = np.arange(n + 1) * dt
    d = np.linalg.norm(A - B, axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.1]))
    glow_line(axs[0], t, A[:, 0], SKY, lw=1.84)
    glow_line(axs[0], t, B[:, 0], ORANGE, lw=1.84, ls=(0, (3, 2)))
    axs[0].set_yticks([]); axs[0].set_ylabel("x(t)")
    axs[1].semilogy(t, d, color=GOLD, lw=1.70)
    axs[1].semilogy(t, 1e-9 * np.exp(LAM1 * t), color=INK, lw=2.24, ls="--", alpha=0.8)
    axs[1].text(9, 1e-4, r"$\delta_0\, e^{\lambda_1 t}$", color=INK, fontsize=18)
    axs[1].set_ylim(1e-10, 1e3); axs[1].set_ylabel("відстань")
    axs[1].set_xlabel("час")
    for ax in axs:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "butterfly")


def fig_horizon():
    """Горизонт передбачуваності vs початкова похибка."""
    rng = np.random.default_rng(7)
    starts = np.array([advance(on_attractor(), 0.01, int(rng.integers(100, 20000))) for _ in range(200)])
    d0 = np.logspace(-14, -1, 14)
    T = predictability_times(starts, d0, 1.0, 0.005, 200.0)
    T = np.where(np.isfinite(T), T, np.nan)
    Tm = np.nanmean(T, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lo, hi = np.nanpercentile(T, 10, axis=0), np.nanpercentile(T, 90, axis=0)
    ax.fill_between(d0, lo, hi, color=TEAL, alpha=0.2, lw=0)
    ax.semilogx(d0, np.log(1 / d0) / LAM1, color=INK, ls="--", lw=1.97, alpha=0.8)
    ax.semilogx(d0, Tm, "o", color=GOLD, ms=7)
    ax.invert_xaxis()
    ax.set_xlabel("початкова похибка  →  точніше")
    ax.set_ylabel("скільки часу прогноз корисний")
    for i in (3, 9):
        ax.annotate("", xy=(d0[i], Tm[i]), xytext=(d0[i + 1], Tm[i + 1]),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.51))
    ax.text(d0[5], Tm[2] + 1, "×10 точніше\n= лише +2.5", color=ORANGE, fontsize=14)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "horizon")


def fig_ensemble():
    """Ансамблевий прогноз."""
    rng = np.random.default_rng(11)
    dt = 0.005
    truth0 = advance(on_attractor(), dt, 1730 * 7)
    ens0 = truth0 + 0.05 * rng.normal(size=(50, 3))
    n = int(12 / dt)
    E = ensemble_run(ens0, dt, n, 4)
    t = np.arange(E.shape[1]) * dt * 4
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for m in range(E.shape[0]):
        axs[0].plot(t, E[m, :, 0], color=SKY, lw=1.02, alpha=0.6)
    glow_line(axs[0], t, E[:, :, 0].mean(0), ORANGE, lw=2.78)
    axs[0].text(0.3, 17, "50 прогнозів з майже однакових стартів", color=SKY, fontsize=13)
    axs[0].text(8.2, -3, "середнє", color=ORANGE, fontsize=13)
    axs[0].set_yticks([])
    p = (E[:, :, 0] > 0).mean(0)
    axs[1].fill_between(t, 0, p, color=TEAL, alpha=0.35, lw=0)
    axs[1].plot(t, p, color=TEAL, lw=2.51)
    axs[1].set_ylim(0, 1); axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_ylabel("P(праве крило)", fontsize=13)
    axs[1].set_xlabel("час прогнозу")
    for ax in axs:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "ensemble")


def fig_enkf():
    """Асиміляція даних."""
    res = enkf_lorenz63(on_attractor(), 200, 0.10, 0.005, [0, 1, 2], 2.0, n_ens=20)
    t = res["t"]
    m = t <= 14
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t[m], res["free"][m, 0], color=DIM, lw=1.70, ls=":", label="модель без даних")
    ax.plot(t[m], res["obs"][m, 0], "o", color=SAND, ms=3.2, alpha=0.8, label="зашумлені вимірювання")
    glow_line(ax, t[m], res["truth"][m, 0], SKY, lw=2.51, label="реальність")
    ax.plot(t[m], res["analysis"][m, 0], color=ORANGE, lw=2.24, ls=(0, (4, 2)), label="модель + дані (EnKF)")
    ax.set_yticks([]); ax.set_xlabel("час")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=15, handlelength=1.4, columnspacing=1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    save(fig, "enkf")


def fig_isobars():
    """Тло слайдів: ізобари вигаданої синоптичної карти (В — високий тиск, Н — низький)."""
    x, y = np.meshgrid(np.linspace(0, 16, 640), np.linspace(0, 9, 360))
    centres = [(1.2, 7.8, 1.0, 2.6, "В"), (14.6, 8.2, -1.2, 2.4, "Н"), (13.8, 0.6, 0.9, 2.8, "В"),
               (2.6, -0.4, -0.8, 2.5, "Н"), (8.0, 11.0, 0.5, 4.0, "")]
    pressure = sum(a * np.exp(-((x - cx)**2 + (y - cy)**2) / w**2) for cx, cy, a, w, _ in centres)
    pressure += 0.05 * np.sin(1.1 * x + 0.4 * y) * np.cos(0.9 * y - 0.3 * x)
    for name, with_labels in [("isobars", False), ("isobars_title", True)]:
        _draw_isobars(name, with_labels, x, y, pressure, centres)


def _draw_isobars(name, with_labels, x, y, pressure, centres):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contour(x, y, pressure, levels=np.arange(-1.15, 1.2, 0.11), colors="#A99B76",
               linewidths=1.1, linestyles="solid")
    for cx, cy, a, _, label in centres:
        if with_labels and label and 0 < cy < 9:
            ax.text(cx, cy, label, color=ORANGE if a < 0 else SKY, fontsize=30,
                    fontweight="bold", ha="center", va="center")
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.set_axis_off()
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(os.path.join(style.FIG_DIR, f"pop_{name}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    setup()
    for f in [fig_isobars, fig_rounding, fig_convection, fig_attractor, fig_blob, fig_gallery, fig_butterfly,
              fig_horizon, fig_ensemble, fig_enkf]:
        print(f.__name__)
        f()
