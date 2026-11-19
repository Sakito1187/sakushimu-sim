"""ハドルのモデルを回して、1 手ずつの記録を out/<名前>.npz に書く

python run.py --n 100 --pe 10 --noise 0 --steps 400 --name n100_pe10
"""
import argparse
import time
from pathlib import Path

import numpy as np

from model import Huddle, axial_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pe", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="selfish", choices=["selfish", "random", "kind"])
    ap.add_argument("--freeze", default="", help="動かない子。番号をカンマで（init のときはそのときの番号）。'auto:windward' = はじめにいちばん風上の子")
    ap.add_argument("--init", default="", help="<名前>:<手> の形のマスから始める（例 f_n0:300）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fields", type=int, default=0, help="この手数ごとに風と温度の場も残す（0 = 残さない）")
    a = ap.parse_args()

    h = Huddle(a.n, pe=a.pe, noise=a.noise, seed=a.seed, select=a.select)
    if a.init:
        src, k = a.init.split(":")
        h.sites = [tuple(int(v) for v in s) for s in np.load(Path(__file__).parent / "out" / f"{src}.npz")["sites"][int(k)]]
        assert len(h.sites) == a.n
    frozen = []
    if a.freeze == "auto:windward":
        xs, _ = h.xy()
        frozen = [int(np.argmin(xs))]
    elif a.freeze == "auto:center":
        xs, ys = h.xy()
        frozen = [int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))]
    elif a.freeze:
        frozen = [int(v) for v in a.freeze.split(",")]
    h.frozen = set(frozen)
    sites = np.zeros((a.steps + 1, a.n, 2), np.int32)
    qs = np.zeros((a.steps, a.n))
    movers = np.zeros(a.steps, np.int32)
    sites[0] = h.sites
    fields = {}
    t = time.time()
    for k in range(a.steps):
        r = h.step()
        qs[k] = r["q"]
        movers[k] = r["mover"]
        sites[k + 1] = h.sites
        if a.fields and k % a.fields == 0:
            f = r["field"]
            fields[f"T_{k}"] = f["T"].astype(np.float32)
            fields[f"origin_{k}"] = np.array([f["x0"], f["y0"], f["h"]])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{a.steps}  {time.time() - t:.0f}s", flush=True)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{a.name}.npz", sites=sites, q=qs, movers=movers, q_amb=h.q_amb,
                        n=a.n, pe=a.pe, noise=a.noise, seed=a.seed, select=a.select,
                        frozen=np.array(frozen, np.int32), **fields)
    print("saved", out / f"{a.name}.npz")


if __name__ == "__main__":
    main()
