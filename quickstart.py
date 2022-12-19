#!/usr/bin/env python3

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import pi

import gfunc

rb = 0.060  # (m)
rpo = 0.032 / 2  # (m)
D = 0.035  # (m)
ks = 2.50  # (W m-1 K-1)
Cs = 3.0e6  # (J m-3 K-1)
kb = 0.73  # (W m-1 K-1)
Cb = 3.8e6  # (J m-3 K-1)

um = 200.0  # (m year-1)
ums = um / (3600 * 8760)  # (m s-1)
phi = pi  # (rad)


def plot_gfunc(t, G_ils, G_ics, G_mils, G_mics, G_cmils):
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        t,
        G_ils,
        ec="k",
        fc="none",
        marker="o",
        s=10,
        lw=0.5,
        label="ILS",
    )
    ax.scatter(
        t,
        G_ics,
        ec="k",
        fc="none",
        marker="^",
        s=10,
        lw=0.5,
        label="ICS",
    )
    ax.scatter(
        t,
        G_mils,
        ec="k",
        fc="k",
        marker="o",
        s=10,
        lw=0.5,
        label="MILS",
    )
    ax.scatter(
        t,
        G_mics,
        ec="k",
        fc="k",
        marker="^",
        s=10,
        lw=0.5,
        label="MICS",
    )
    ax.scatter(
        t,
        G_cmils,
        ec="k",
        fc="k",
        marker="+",
        s=10,
        lw=0.5,
        label="CM-ILS (at U-tube wall)",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("$G(t)$")
    ax.set_xscale("log")
    ax.set_title(
        r"$k_\mathrm{s} = %.2f$, $k_\mathrm{b} = %.2f$, $u_\mathrm{m} = %.0f$"
        % (ks, kb, um),
        fontsize=10,
    )
    ax.set_xticks(
        [
            60,
            60 * 60,
            60 * 60 * 24,
            60 * 60 * 24 * 10,
            60 * 60 * 24 * 365,
        ]
    )
    ax.set_xticklabels(["1 min", "1 hour", "1 day", "10 day", "1 year"])
    ax.minorticks_off()
    plt.legend(frameon=False)
    plt.show()


def main():
    tmin = 60.0  # (s)
    tmax = 60.0 * 60 * 24 * 365 * 1  # (s)
    t = np.logspace(np.log10(tmin), np.log10(tmax), num=50, endpoint=True)

    alphas = ks / Cs  # (m2 s-1)
    alphab = kb / Cb  # (m2 s-1)

    R = rb / rb

    RA = (D + rpo) / rb
    RB = (D - rpo) / rb
    RD = D / rb
    k = ks / kb
    alpha = np.sqrt(alphab / alphas)

    # ILS-------------------------------------------------
    start = time.time()
    Fo_ils = alphas * t / (rb * rb)
    f_ils = gfunc.ils(Fo_ils)
    G_ils = 1 / (4 * pi * ks) * f_ils
    print(f"  ils: {time.time() - start: >2.6f} sec")
    # ILS-------------------------------------------------

    # # ICS-------------------------------------------------
    start = time.time()
    Fo_ics = alphas * t / (rb * rb)
    f_ics = gfunc.ics(Fo_ics, R)
    G_ics = 1 / (pi * pi * ks) * f_ics
    print(f"  ics: {time.time() - start: >2.6f} sec")
    # # ICS-------------------------------------------------

    # MILS-------------------------------------------------
    start = time.time()
    Fo_mils = alphas * t / (rb * rb)
    Pe_mils = ums * rb / alphas
    # f_mils = gfunc.mils(Fo_mils, Pe_mils, phi)
    f_mils = gfunc.milsmean(Fo_mils, Pe_mils)
    G_mils = 1 / (4 * pi * ks) * f_mils
    print(f" mils: {time.time() - start: >2.6f} sec")
    # MILS-------------------------------------------------

    # MICS-------------------------------------------------
    start = time.time()
    Fo_mics = alphas * t / (rb * rb)
    Pe_mics = ums * rb / alphas
    # f_mics = gfunc.mics(Fo_mics, Pe_mics, phi, R)
    f_mics = gfunc.micsmean(Fo_mics, Pe_mics, R)
    G_mics = 1 / (2 * pi * ks) * f_mics
    print(f" mics: {time.time() - start: >2.6f} sec")
    # MICS-------------------------------------------------

    # # CMILS-------------------------------------------------
    start = time.time()
    Fo_cmils = alphab * t / (rb * rb)
    f_cmils = gfunc.cmils1u(Fo_cmils, RA, RB, RD, alpha, k)
    G_cmils = 1 / (2 * pi * kb) * f_cmils
    print(f"cmils: {time.time() - start: >2.6f} sec")
    # # CMILS-------------------------------------------------

    plot_gfunc(t, G_ils, G_ics, G_mils, G_mics, G_cmils)


if __name__ == "__main__":
    main()
