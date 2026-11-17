"""run.py の結果を図にして、論文と見比べる

python plot.py n100_pe10            # 形の移り変わり（温度の場つき）と、重心・縦横比
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import axial_xy


def shape(xs, ys):
    """風の向き（x）の長さと、それと直角の向き（y）の厚さ"""
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def main(name):
    out = Path(__file__).parent / "out"
    d = np.load(out / f"{name}.npz")
    sites, q = d["sites"], d["q"]
    steps = len(q)
    snaps = [k for k in (0, 10, 50, 100, 200, steps - 1) if k < steps]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, k in zip(axs.flat, snaps):
        xs, ys = axial_xy(sites[k, :, 0], sites[k, :, 1])
        tk = max([int(s[2:]) for s in d.files if s.startswith("T_") and int(s[2:]) <= k], default=None)
        if tk is not None:
            x0, y0, h = d[f"origin_{tk}"]
            T = d[f"T_{tk}"]
            ext = [x0, x0 + T.shape[0] * h, y0, y0 + T.shape[1] * h]
            ax.imshow(T.T, origin="lower", extent=ext, cmap="coolwarm", vmin=0, vmax=1, alpha=0.6)
        sc = ax.scatter(xs, ys, c=q[k], cmap="viridis", s=60, edgecolors="k", linewidths=0.3)
        L, W = shape(xs, ys)
        ax.set_title(f"step {k}  長さ{L:.1f} 厚さ{W:.1f}", fontname="Yu Gothic")
        ax.set_aspect("equal")
        cxm = xs.mean()
        ax.set_xlim(cxm - 18, cxm + 18)
        ax.set_ylim(ys.mean() - 12, ys.mean() + 12)
    fig.suptitle(f"N={int(d['n'])} Pe={float(d['pe'])} noise={float(d['noise'])}  風は左から（背景は温度の場の例）", fontname="Yu Gothic")
    fig.tight_layout()
    fig.savefig(out / f"{name}_shapes.png", dpi=80)

    cx = [axial_xy(s[:, 0], s[:, 1])[0].mean() for s in sites]
    ar = []
    for s in sites:
        xs, ys = axial_xy(s[:, 0], s[:, 1])
        L, W = shape(xs, ys)
        ar.append(L / W)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(cx)
    axs[0].set_title("重心の x（風下へ進むか）", fontname="Yu Gothic")
    axs[1].plot(ar)
    axs[1].set_title("長さ / 厚さ", fontname="Yu Gothic")
    fig.tight_layout()
    fig.savefig(out / f"{name}_drift.png", dpi=80)
    print("saved", out / f"{name}_shapes.png", out / f"{name}_drift.png")


if __name__ == "__main__":
    main(sys.argv[1])
