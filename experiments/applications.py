"""
Застосування системи Лоренца (додатковий розділ презентації).

  secret     — секретний зв'язок: Морзе «ХАОС», захований у хаосі; ключ-атака
  wheel      — водяне колесо Малкуса (24 відра) = Лоренц з beta = 1
  reservoir  — мережа луни (ESN) прогнозує хаос без рівнянь
  sindy      — SINDy відновлює рівняння з даних (і з даних колеса)

Результати: results/applications.json, results/numbers_apps.tex,
графіки figures/pop_{secret,wheel,reservoir,sindy}.png

Запуск:  python experiments/applications.py [secret wheel ...]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from popfigs import setup, save, glow_line, clean, INK, DIM, GOLD, ORANGE, TEAL, SAND, SKY  # noqa: E402
from lorenzlab.core import SIGMA, RHO, BETA, integrate, on_attractor  # noqa: E402
from lorenzlab.lyapunov import lyapunov_spectrum  # noqa: E402
from lorenzlab.analysis import z_maxima  # noqa: E402
from lorenzlab.applications import (chaotic_channel, morse_envelope, envelope, MORSE,  # noqa: E402
                                    waterwheel, wheel_params_for, wheel_to_lorenz,
                                    EchoStateNetwork, valid_time,
                                    poly_library, stlsq, true_coefficients, derivative, POLY_NAMES)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Circle  # noqa: E402

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "applications.json")
R = json.load(open(RES)) if os.path.exists(RES) else {}
LAM1 = json.load(open(os.path.join(ROOT, "results", "results.json")))["lyap"][0]


def decode_morse(env01, dt, unit):
    """0/1-сигнал -> текст (кирилиця Х, А, О, С)."""
    inv = {v: k for k, v in MORSE.items() if k in "ХАОС"}
    runs, cur, n = [], env01[0], 0
    for v in env01:
        if v == cur:
            n += 1
        else:
            runs.append((cur, n * dt / unit)); cur, n = v, 1
    runs.append((cur, n * dt / unit))
    while runs and runs[0][0] == 0:
        runs.pop(0)
    out, sym = "", ""
    for v, L in runs:
        if v:
            sym += "." if L < 2 else "-"
        elif L > 2:
            out += inv.get(sym, "?"); sym = ""
    if sym:
        out += inv.get(sym, "?")
    return out


# ---------------------------------------------------------------------------
def exp_secret():
    """Хаотичне маскування: Морзе «ХАОС» ховається в x(t)."""
    dt, unit, f, A, lead = 0.001, 0.4, 10.0, 0.4, 5.0
    env = morse_envelope("ХАОС", unit, dt)
    env = np.concatenate([np.zeros(int(lead / dt)), env, np.zeros(int(2 / dt))])
    n = env.size
    t = np.arange(n) * dt
    msg = A * env * np.sin(2 * np.pi * f * t)
    u0 = np.concatenate([on_attractor(), [0.0, 0.0, 0.0]])

    s, rec = chaotic_channel(u0, dt, msg, np.array([SIGMA, RHO, BETA]))
    m = t > lead - 1
    R["secret_ratio"] = float(np.std(msg[env > 0]) / np.std(s))        # повідомлення / сигнал
    R["secret_err"] = float(np.std((rec - msg)[m]) / A)
    e_ok = envelope(rec, dt, 0.1)
    dec = decode_morse((e_ok[t > lead - 0.5] > A / 2).astype(int), dt, unit)
    R["secret_decoded"] = dec

    # атака: приймач з неточним ключем r' = r + dr
    drs = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    out = []
    for dr in drs:
        _, rc = chaotic_channel(u0, dt, msg, np.array([SIGMA, RHO + dr, BETA]))
        e = envelope(rc, dt, 0.1)[t > lead - 0.5]
        noise = float(np.std((rc - msg)[m]) / A)
        out.append(dict(dr=float(dr), noise=noise, decoded=decode_morse((e > A / 2).astype(int), dt, unit)))
    R["secret_attack"] = out
    print("  decoded:", dec, "| message/signal:", round(R["secret_ratio"], 3))
    for o in out:
        print(f"    dr={o['dr']:<5} noise/A={o['noise']:.3f}  -> {o['decoded']}")
    bad = next(o for o in out if o["noise"] > 0.5)
    _, rc_bad = chaotic_channel(u0, dt, msg, np.array([SIGMA, RHO + bad["dr"], BETA]))

    fig, axs = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.25))
    w = (t > lead - 1.5)
    axs[0].plot(t[w], s[w], color=SKY, lw=1.56)
    axs[0].text(0.01, 0.95, "у каналі: x(t) + повідомлення", transform=axs[0].transAxes,
                color=SKY, fontsize=14, va="top")
    axs[1].plot(t[w], rec[w], color=GOLD, lw=1.16)
    axs[1].plot(t[w], A * env[w], color=TEAL, lw=1.97, alpha=0.9)
    axs[1].text(0.01, 0.95, f"приймач з ключем r = {RHO:g}  →  «{dec}»", transform=axs[1].transAxes,
                color=GOLD, fontsize=14, va="top")
    axs[1].set_ylim(-1.6 * A, 1.9 * A)
    axs[2].plot(t[w], rc_bad[w], color=ORANGE, lw=1.16)
    axs[2].text(0.01, 0.95, f"перехоплювач з ключем r = {RHO + bad['dr']:g}  →  «{bad['decoded']}»",
                transform=axs[2].transAxes, color=ORANGE, fontsize=14, va="top")
    lim = np.percentile(np.abs(rc_bad[w]), 99.5)
    axs[2].set_ylim(-1.1 * lim, 1.5 * lim)
    for ax in axs:
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
    axs[2].set_xlabel("час")
    save(fig, "secret")


# ---------------------------------------------------------------------------
def exp_wheel():
    """Водяне колесо Малкуса з 24 відер."""
    p = wheel_params_for(N=24)
    T, dt = 300.0, 0.002
    t, th, om, ms = waterwheel(p["N"], T, dt, p["Q"], p["K"], p["nu"], p["I"], p["gR"], 0.1, 5)
    L = wheel_to_lorenz(th, om, ms, p)
    keep = t > 30
    rev = int(np.sum(np.diff(np.sign(om[keep])) != 0))
    lam = lyapunov_spectrum(np.array([1.0, 1.0, 1.0]), 0.005, 2000, 400000, 10.0, 28.0, 1.0)[0]
    R["wheel_Q"] = float(p["Q"]); R["wheel_reversals"] = rev
    R["wheel_reversal_rate"] = rev / float(t[-1] - 30)
    R["wheel_lam"] = [float(v) for v in lam]
    R["wheel_hopf"] = 10.0 * (10 + 1 + 3) / (10 - 1 - 1)
    # наскільки дискретне колесо відрізняється від ідеального Лоренца на короткому відрізку
    i0 = np.searchsorted(t, 100.0)
    ref = integrate(L[i0].copy(), dt, int(3.0 / dt), 10.0, 28.0, 1.0, 5)
    R["wheel_vs_lorenz_3t"] = float(np.max(np.abs(ref[:, 0] - L[i0:i0 + len(ref), 0])))
    np.savez_compressed(os.path.join(ROOT, "results", "wheel_series.npz"), t=t, L=L)
    print(f"  Q={p['Q']:.2f}, реверсів {rev}, λ(β=1)={np.round(lam, 3)}, "
          f"|ω-x_Lorenz|max на 3 од. часу = {R['wheel_vs_lorenz_3t']:.2e}")

    fig = plt.figure(figsize=(13, 4.6))
    # --- малюнок колеса
    ax = fig.add_axes([0.0, 0.02, 0.30, 0.96]); ax.set_aspect("equal"); clean(ax)
    k = np.searchsorted(t, 151.3)
    N = p["N"]; mmax = ms[keep].max()
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=DIM, lw=1.97))
    ax.plot(0, 0, "o", color=DIM, ms=6)
    for i in range(N):
        phi = th[k] + 2 * np.pi * i / N
        cx, cy = -np.sin(phi), np.cos(phi)
        ax.plot([0, cx], [0, cy], color="#9C937C", lw=1.16)
        wcup, hcup = 0.13, 0.2
        ax.add_patch(Rectangle((cx - wcup / 2, cy - hcup / 2), wcup, hcup, fc="none", ec=INK, lw=1.56))
        lvl = hcup * ms[k, i] / mmax
        ax.add_patch(Rectangle((cx - wcup / 2, cy - hcup / 2), wcup, lvl, fc=SKY, ec="none", alpha=0.9))
    ax.annotate("", xy=(0, 1.13), xytext=(0, 1.5), arrowprops=dict(arrowstyle="-|>", color=SKY, lw=3.05))
    ax.text(0, 1.55, "вода", color=SKY, ha="center", va="bottom", fontsize=13)
    ax.text(0, -1.32, "відра протікають", color=DIM, ha="center", fontsize=12)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.45, 1.8)
    # --- кутова швидкість
    ax2 = fig.add_axes([0.36, 0.56, 0.62, 0.40])
    w = (t > 30) & (t < 90)
    ax2.plot(t[w] - 30, om[w], color=GOLD, lw=1.84)
    ax2.axhline(0, color=DIM, lw=1.16, ls=":")
    ax2.set_ylabel(r"$\omega$")
    ax2.text(0.5, 13.5, "проти годинникової", color=DIM, fontsize=11)
    ax2.text(0.5, -15.5, "за годинниковою", color=DIM, fontsize=11)
    ax2.set_ylim(-17, 17)
    # --- атрактор колеса проти Лоренца beta=1
    ax3 = fig.add_axes([0.36, 0.0, 0.29, 0.42]); clean(ax3)
    ax3.plot(L[keep, 0], L[keep, 2], color=GOLD, lw=0.69, alpha=0.8)
    ax3.set_title("колесо (24 відра)", color=GOLD, fontsize=12)
    ax4 = fig.add_axes([0.69, 0.0, 0.29, 0.42]); clean(ax4)
    ref = integrate(on_attractor(rho=28.0, beta=1.0), 0.005, 50000, 10.0, 28.0, 1.0)
    ax4.plot(ref[:, 0], ref[:, 2], color=SKY, lw=0.69, alpha=0.8)
    ax4.set_title("рівняння Лоренца, β = 1", color=SKY, fontsize=12)
    for a in (ax3, ax4):
        a.set_xlim(-17, 17); a.set_ylim(5, 48)
    save(fig, "wheel")


# ---------------------------------------------------------------------------
def exp_reservoir():
    """Мережа луни: прогноз без рівнянь."""
    dt, sub = 0.02, 4
    tr = integrate(on_attractor(), dt / sub, sub * 120000, stride=sub)
    mu, sd = tr.mean(0), tr.std(0)
    U = (tr - mu) / sd
    n_train = 10000                     # 200 одиниць часу ≈ 180 ляпуновських часів
    esn = EchoStateNetwork(n_res=800, radius=0.9, input_scale=0.2, ridge=1e-8, seed=2)
    fit = esn.train(U[:n_train])
    lt = dt * LAM1                      # ляпуновських часів на крок
    vts = []
    starts = n_train + 500 + 2500 * np.arange(30)
    for st in starts:
        esn.listen(U[st - 300:st])
        P = esn.predict(1500)
        vt, _ = valid_time(U[st:st + 1500], P)
        vts.append(vt * lt)
    vts = np.array(vts)
    R["esn_fit_rmse"] = fit
    R["esn_vt_mean"] = float(vts.mean()); R["esn_vt_median"] = float(np.median(vts))
    R["esn_vt_min"] = float(vts.min()); R["esn_vt_max"] = float(vts.max())
    R["esn_n_train_lyap"] = float(n_train * lt)

    # еталон: ідеальна модель з тією ж похибкою стану, що й ESN на першому кроці
    st = int(starts[np.argsort(vts)[len(vts) // 2]])       # медіанний випадок для графіка
    esn.listen(U[st - 300:st])
    P = esn.predict(1500)
    vt, err = valid_time(U[st:st + 1500], P)
    # «клімат»: довгий автономний прогін
    esn.listen(U[n_train - 300:n_train])
    Plong = esn.predict(60000) * sd + mu
    zm_true = z_maxima(on_attractor(), 0.005, 0, int(60000 * dt / 0.005))
    zl = Plong[:, 2]
    i = np.where((zl[1:-1] > zl[:-2]) & (zl[1:-1] > zl[2:]))[0] + 1
    zm_esn = zl[i]
    R["esn_climate_zmax_mean"] = [float(zm_true.mean()), float(zm_esn.mean())]
    R["esn_climate_zmax_std"] = [float(zm_true.std()), float(zm_esn.std())]
    print(f"  fit RMSE={fit:.1e}; valid time = {vts.mean():.2f} ± {vts.std():.2f} λ-часів "
          f"(мін {vts.min():.2f}, макс {vts.max():.2f}); z_max: {np.round(R['esn_climate_zmax_mean'], 2)}")

    fig = plt.figure(figsize=(13, 4.2))
    ax = fig.add_axes([0.0, 0.14, 0.62, 0.84])
    tl = np.arange(1500) * lt
    w = tl < 14
    glow_line(ax, tl[w], (U[st:st + 1500, 0] * sd[0] + mu[0])[w], SKY, lw=2.38)
    ax.plot(tl[w], (P[:, 0] * sd[0] + mu[0])[w], color=ORANGE, lw=2.24, ls=(0, (4, 2)))
    ax.axvline(vt * lt, color=GOLD, lw=1.70, ls=":")
    ax.text(vt * lt - 0.2, 25,  f"прогноз тримається\n≈ {vt * lt:.1f} ляпуновських часів", color=GOLD, fontsize=12, va="top", ha="right")
    ax.text(0.2, -20, "реальність", color=SKY, fontsize=13)
    ax.text(3.2, -20, "нейромережа (не знає рівнянь)", color=ORANGE, fontsize=13)
    ax.set_xlabel("час прогнозу,  λ₁t"); ax.set_yticks([]); ax.set_ylabel("x(t)")
    ax.set_ylim(-22, 26)
    ax2 = fig.add_axes([0.68, 0.14, 0.31, 0.84])
    ax2.plot(zm_true[:-1], zm_true[1:], ".", color=SKY, ms=2.2, alpha=0.7)
    ax2.plot(zm_esn[:-1], zm_esn[1:], ".", color=ORANGE, ms=1.4, alpha=0.6)
    ax2.set_xlabel("$z_{max}$ (n)"); ax2.set_ylabel("$z_{max}$ (n+1)")
    ax2.text(29.5, 47, "«клімат» збігається", color=INK, fontsize=12)
    ax2.set_xlim(29, 49); ax2.set_ylim(29, 49)
    save(fig, "reservoir")


# ---------------------------------------------------------------------------
def exp_sindy():
    """SINDy: відкрити рівняння з даних."""
    dt = 0.002
    X = integrate(on_attractor(), dt, int(20 / dt))
    true = true_coefficients()
    rng = np.random.default_rng(0)
    noises = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0])
    raw, smo, raw_ok, smo_ok = [], [], [], []
    cut = slice(300, -300)
    for nz in noises:
        Xn = X + nz * rng.standard_normal(X.shape)
        Xi = stlsq(poly_library(Xn[cut]), derivative(Xn, dt)[cut], 0.5)
        raw.append(np.max(np.abs(Xi - true) / np.maximum(np.abs(true), 1)))
        raw_ok.append(bool(np.array_equal(Xi != 0, true != 0)))
        Xs = savgol_filter(Xn, 51, 3, axis=0)
        Xi = stlsq(poly_library(Xs[cut]), derivative(Xn, dt, 51)[cut], 0.5)
        smo.append(np.max(np.abs(Xi - true) / np.maximum(np.abs(true), 1)))
        smo_ok.append(bool(np.array_equal(Xi != 0, true != 0)))
        if nz == 0.5:
            R["sindy_noisy_coefs"] = Xi.tolist()
    Xi0 = stlsq(poly_library(X[cut]), derivative(X, dt)[cut], 0.5)
    R["sindy_clean_coefs"] = Xi0.tolist()
    R["sindy_noise"] = noises.tolist()
    R["sindy_err_raw"] = [float(v) for v in raw]; R["sindy_err_smooth"] = [float(v) for v in smo]
    R["sindy_ok_raw"] = raw_ok; R["sindy_ok_smooth"] = smo_ok

    # колесо: вимірюємо (omega, b1, a1) — SINDy має знайти beta = 1
    if os.path.exists(os.path.join(ROOT, "results", "wheel_series.npz")):
        d = np.load(os.path.join(ROOT, "results", "wheel_series.npz"))
        tw, Lw = d["t"], d["L"]
        m = (tw > 30) & (tw < 60)
        dtw = tw[1] - tw[0]
        Xi_w = stlsq(poly_library(Lw[m][cut]), derivative(Lw[m], dtw)[cut], 0.5)
        R["sindy_wheel_coefs"] = Xi_w.tolist()
        R["sindy_wheel_beta"] = float(-Xi_w[3, 2])
        print(f"  колесо -> SINDy: σ={-Xi_w[1, 0]:.3f}, r={Xi_w[1, 1]:.3f}, β={-Xi_w[3, 2]:.3f}; "
              f"ненульових членів {int(np.sum(Xi_w != 0))}")
    print(f"  чисті дані: σ={-Xi0[1, 0]:.4f}, r={Xi0[1, 1]:.4f}, β={-Xi0[3, 2]:.4f}")
    for nz, a, b, oa, ob in zip(noises, raw, smo, raw_ok, smo_ok):
        print(f"    шум {nz:<4}: похибка {a:.3g} ({'OK' if oa else 'ні'}) / згладж. {b:.3g} ({'OK' if ob else 'ні'})")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(noises[1:], raw[1:], "o-", color=DIM, lw=2.24, ms=7, label="похідна «в лоб»")
    ax.loglog(noises[1:], smo[1:], "o-", color=GOLD, lw=2.51, ms=7, label="зі згладжуванням")
    for nz, a, ok in zip(noises[1:], raw[1:], raw_ok[1:]):
        if not ok:
            ax.plot(nz, a, "x", color=ORANGE, ms=12, mew=2.5)
    ax.axhline(0.05, color=TEAL, lw=1.70, ls=":")
    ax.text(0.011, 0.058, "5% — рівняння впізнавані", color=TEAL, fontsize=11)
    ax.set_xlabel("шум вимірювань (σ шуму)"); ax.set_ylabel("відносна похибка коефіцієнтів")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "× — знайдено зайві члени", transform=ax.transAxes, ha="right", color=ORANGE, fontsize=11)
    save(fig, "sindy")


# ---------------------------------------------------------------------------
def write_tex():
    f = lambda v, d: f"{v:.{d}f}"
    L = ["% автоматично згенеровано experiments/applications.py"]
    nc = lambda k, v: L.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    if "secret_ratio" in R:
        nc("secPct", f(100 * R["secret_ratio"], 0)); nc("secDecoded", R["secret_decoded"])
        nc("secErr", f(R["secret_err"], 3))
        bad = next(o for o in R["secret_attack"] if o["noise"] > 0.5)
        nc("secBadDr", f"{bad['dr']:g}")
        ok = [o for o in R["secret_attack"] if o["decoded"] == "ХАОС"]
        nc("secMaxOkDr", f"{max(o['dr'] for o in ok):g}")
    if "wheel_lam" in R:
        nc("wheelLam", f(R["wheel_lam"][0], 3)); nc("wheelRev", R["wheel_reversals"])
        nc("wheelHopf", f(R["wheel_hopf"], 1))
    if "esn_vt_mean" in R:
        nc("esnVT", f(R["esn_vt_mean"], 1)); nc("esnVTmin", f(R["esn_vt_min"], 1))
        nc("esnVTmax", f(R["esn_vt_max"], 1)); nc("esnTrain", f(R["esn_n_train_lyap"], 0))
    if "sindy_clean_coefs" in R:
        C = np.array(R["sindy_clean_coefs"])
        nc("sinSigma", f(-C[1, 0], 3)); nc("sinRho", f(C[1, 1], 3)); nc("sinBeta", f(-C[3, 2], 3))
        Cn = np.array(R["sindy_noisy_coefs"])
        nc("sinNSigma", f(-Cn[1, 0], 2)); nc("sinNRho", f(Cn[1, 1], 2)); nc("sinNBeta", f(-Cn[3, 2], 2))
    if "sindy_wheel_beta" in R:
        W = np.array(R["sindy_wheel_coefs"])
        nc("sinWSigma", f(-W[1, 0], 2)); nc("sinWRho", f(W[1, 1], 2)); nc("sinWBeta", f(R["sindy_wheel_beta"], 2))
    with open(os.path.join(ROOT, "results", "numbers_apps.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")


EXPS = [exp_secret, exp_wheel, exp_reservoir, exp_sindy]

if __name__ == "__main__":
    setup()
    sel = sys.argv[1:]
    for fn in EXPS:
        if sel and not any(s in fn.__name__ for s in sel):
            continue
        t0 = time.time()
        print(f"[{fn.__name__}] {fn.__doc__.strip()}", flush=True)
        fn()
        print(f"  done in {time.time() - t0:.1f}s")
        json.dump(R, open(RES, "w"), indent=2, ensure_ascii=False)
    write_tex()
    print("->", os.path.relpath(RES, ROOT))
