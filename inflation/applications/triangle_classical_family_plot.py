"""Explain the distinct curve and band in the E3=0 triangle summary.

Render already verified CSV data from triangle_e3_zero_asymmetric.py.
This is a presentation helper, not a new model or compatibility test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(data_directory, output):
    curve = np.loadtxt(data_directory/"eq5-curve.csv", delimiter=",", skiprows=1)
    upper = np.loadtxt(data_directory/"322-upper.csv", delimiter=",", skiprows=1)
    lower = np.loadtxt(data_directory/"322-lower.csv", delimiter=",", skiprows=1)
    assert curve.shape[1] == upper.shape[1] == lower.shape[1] == 2
    assert np.allclose(upper[:, 0], lower[:, 0])
    assert np.all(upper[:, 1] >= lower[:, 1]-1e-12)
    red, blue = "#A74455", "#3272A5"
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.30, top=.80, wspace=.19)
    for ax in axes:
        ax.set(xlim=(-.67, .67), ylim=(-.012, .34), xlabel=r"$E_1$",
               xticks=[-.6, -.3, 0, .3, .6], yticks=[0, .1, .2, .3])
        ax.set_axisbelow(True)
        ax.grid(color="#eeeeee", linewidth=.7)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(color="#888888")
    axes[0].set_ylabel(r"$E_2$  (with $E_3=0$)")
    axes[0].set_title("One model, varied local noise", pad=24, fontsize=13)
    axes[0].plot(curve[:, 0], curve[:, 1], color=red, linewidth=2.3)
    axes[0].scatter(curve[[0, -1], 0], curve[[0, -1], 1], color=red, s=25, zorder=3)
    axes[0].scatter([0], [0], color=red, s=17, zorder=3)
    axes[0].text(.5, .87, r"$E_2=\frac{\sqrt{5}}{3}E_1^2$",
                 transform=axes[0].transAxes, ha="center", fontsize=19, color=red)
    axes[0].text(-.61, .302, r"$\eta=-1$", ha="center", color=red, fontsize=11)
    axes[0].text(.61, .302, r"$\eta=1$", ha="center", color=red, fontsize=11)
    axes[0].annotate(r"$\eta=0$", xy=(0, 0), xytext=(-.22, .055),
                     arrowprops={"arrowstyle": "-", "color": red, "lw": .7},
                     color=red, fontsize=11)
    axes[1].set_title("A separate family fills a band", pad=24, fontsize=13)
    axes[1].fill_between(upper[:, 0], lower[:, 1], upper[:, 1],
                         color=blue, alpha=.2)
    polygon = np.vstack((upper, lower[::-1], upper[:1]))
    axes[1].plot(polygon[:, 0], polygon[:, 1], color=blue, linewidth=1.25)
    axes[1].plot([], [], color=blue, linewidth=5, alpha=.45, label=r"$322$ family: area in $C$")
    axes[1].plot(curve[:, 0], curve[:, 1], color=red, linewidth=2.3,
                 label="Single-example family: curve in $C$")
    axes[1].scatter(curve[[0, -1], 0], curve[[0, -1], 1], color=red, s=25, zorder=3)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.23),
                   frameon=False, fontsize=10, handlelength=2.4)
    fig.text(.25, .105, "Only points on the red curve are supplied by this family.",
             ha="center", fontsize=10)
    fig.text(.53, .96, "Two distinct constructive subsets of the same $E_3=0$ slice",
             ha="center", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white",
                metadata={"Title": "Curve versus band in the binary triangle",
                          "Description": "Machine-written visualization of verified classical model data. "
                                         "Red: Eq. (5) plus local channels and noise. "
                                         "Blue: a restricted 322 model family. "
                                         "Both establish C membership, with no exclusion from C_sym. "
                                         "Source models: da Silva et al., arXiv:2503.16654v2. "
                                         "Data: sources/data/261019 - asymmetric classical slice constructions."})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.data_directory, args.output)
