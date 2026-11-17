"""公平さを測る（論文の Fig. 7 と同じ考え方）

- 動く番の間隔: ある子が動いてから次に動くまでの手数を N で割ったもの。でたらめに選ぶなら平均 1・ばらつきは幾何分布（約 1）
- 熱の失い方の合計: 1 羽ずつの Q の合計のばらつき（変動係数）
最初の skip 手（形が落ち着くまで）は数えない

python fairness.py f_n0 f_n1 f_n3 f_rand
"""
import sys
from pathlib import Path

import numpy as np


def measure(name, skip=300):
    d = np.load(Path(__file__).parent / "out" / f"{name}.npz")
    n = int(d["n"])
    movers, q = d["movers"], d["q"]
    last = -np.ones(n, int)
    waits = []
    for k in range(skip, len(movers)):
        m = movers[k]
        if last[m] >= 0:
            waits.append((k - last[m]) / n)
        last[m] = k
    waits = np.array(waits)
    total = q[skip:].sum(0)
    p = 1 / n
    geo_sd = np.sqrt(1 - p) / p / n
    return {
        "name": name, "select": str(d["select"]) if "select" in d.files else "selfish", "noise": float(d["noise"]),
        "wait_mean": waits.mean(), "wait_sd": waits.std(), "geo_sd": geo_sd,
        "never_moved": int((last < 0).sum()), "total_cv": total.std() / total.mean(),
    }


if __name__ == "__main__":
    for name in sys.argv[1:]:
        r = measure(name)
        print(f"{r['name']:8s} {r['select']:8s} noise={r['noise']:<4} 間隔 平均 {r['wait_mean']:.2f} ばらつき {r['wait_sd']:.2f}"
              f"（でたらめなら {r['geo_sd']:.2f}） 一度も動かない {r['never_moved']} 羽  熱の合計の変動係数 {r['total_cv']:.3f}")
