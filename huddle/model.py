"""ハドルのモデル（Waters, Blanchette, Kim 2012, PLoS ONE 7(11): e50277 の再現）

論文との違い:
- 論文は群れの外を単位円の外に写して（Schwarz-Christoffel 写像）風と温度を解いている。
  ここでは同じ方程式（ポテンシャル流と定常の移流拡散）を、直交格子の有限体積法で解く。
- 長さの単位はペンギンの直径（= 格子の間隔）。論文は群れを単位円に写して解くので、Pe の長さの基準は群れの大きさ。
  ここでは群れと同じ面積の円の半径 R を基準にし、Pe = U R / D とする（1 羽だけのときは R = 0.5）。

1 手の流れ（論文の Methods のとおり）:
1. 群れのまわりの風（ポテンシャル流）を解く
2. 温度（ペンギン = 1、遠く = 0）を移流拡散で解く
3. ふちのペンギンごとの熱の失い方 Q_i を求める（中のペンギンは 0）
4. ゆらぎを足した Q_i が最大の子が動く。行き先は、Q が最小のふちの子と、
   その子のとなりで Q が最小のふちの子の、両方にとなり合う空いたマス
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

SQ3 = np.sqrt(3.0)
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def axial_xy(q, r):
    """六角格子（間隔 1）の中心の座標。風は +x 向き"""
    q = np.asarray(q, float)
    r = np.asarray(r, float)
    return q + r / 2.0, r * SQ3 / 2.0


class Flow:
    """群れのまわりの風と温度を、格子の上で解く"""

    def __init__(self, h=0.25, margin=6.0, radius=0.5):
        self.h = h
        self.margin = margin
        self.radius = radius

    def solve(self, xs, ys, pe, tris=None, edge=None, length=0.5):
        """tris: 3 羽ずつ互いにとなり合う組（群れの形 = その三角形をすべて合わせた多角形。論文と同じく、ふちの子の中心を結んだ形）
        edge: ふちの子の番号。熱はいちばん近いふちの子のものとする（= となりとの中点で区切る）"""
        h = self.h
        x0, x1 = xs.min() - self.margin - 0.5, xs.max() + self.margin * 1.5 + 0.5
        y0, y1 = ys.min() - self.margin - 0.5, ys.max() + self.margin + 0.5
        nx, ny = int(np.ceil((x1 - x0) / h)), int(np.ceil((y1 - y0) / h))
        cx = x0 + (np.arange(nx) + 0.5) * h
        cy = y0 + (np.arange(ny) + 0.5) * h
        X, Y = np.meshgrid(cx, cy, indexing="ij")

        solid = np.zeros((nx, ny), bool)
        owner = -np.ones((nx, ny), int)
        if tris is None or len(tris) == 0:
            # 1 羽だけ: 半径 0.5 の円
            r = self.radius
            for px, py in zip(xs, ys):
                solid |= (X - px) ** 2 + (Y - py) ** 2 <= r * r
        else:
            for a, b, c in tris:
                ax_, ay_, bx_, by_, cx_, cy_ = xs[a], ys[a], xs[b], ys[b], xs[c], ys[c]
                i0, i1 = max(0, int((min(ax_, bx_, cx_) - x0) / h) - 1), min(nx, int((max(ax_, bx_, cx_) - x0) / h) + 2)
                j0, j1 = max(0, int((min(ay_, by_, cy_) - y0) / h) - 1), min(ny, int((max(ay_, by_, cy_) - y0) / h) + 2)
                px, py = X[i0:i1, j0:j1], Y[i0:i1, j0:j1]
                d1 = (px - bx_) * (ay_ - by_) - (ax_ - bx_) * (py - by_)
                d2 = (px - cx_) * (by_ - cy_) - (bx_ - cx_) * (py - cy_)
                d3 = (px - ax_) * (cy_ - ay_) - (cx_ - ax_) * (py - ay_)
                e = 1e-9
                inside = ~(((d1 < -e) | (d2 < -e) | (d3 < -e)) & ((d1 > e) | (d2 > e) | (d3 > e)))
                solid[i0:i1, j0:j1] |= inside
            # どの三角形にも入らない子（となりの 2 羽どうしが接していない）は、となりとの間を太さ 0.5 の線で結ぶ
            intri = np.zeros(len(xs), bool)
            for t in tris:
                intri[list(t)] = True
            for a in np.nonzero(~intri)[0]:
                for b in range(len(xs)):
                    if b == a or (xs[a] - xs[b]) ** 2 + (ys[a] - ys[b]) ** 2 > 1.01:
                        continue
                    vx, vy = xs[b] - xs[a], ys[b] - ys[a]
                    t = np.clip(((X - xs[a]) * vx + (Y - ys[a]) * vy) / (vx * vx + vy * vy), 0, 1)
                    solid |= (X - xs[a] - t * vx) ** 2 + (Y - ys[a] - t * vy) ** 2 <= 0.25 ** 2
            solid = ndimage.binary_fill_holes(solid)
        cand = np.arange(len(xs)) if edge is None else np.asarray(edge)
        pts = np.stack([xs[cand], ys[cand]], 1)
        si, sj = np.nonzero(solid)
        d = (X[si, sj, None] - pts[None, :, 0]) ** 2 + (Y[si, sj, None] - pts[None, :, 1]) ** 2
        owner[si, sj] = cand[np.argmin(d, 1)]

        fluid = ~solid
        idx = -np.ones((nx, ny), int)
        idx[fluid] = np.arange(fluid.sum())
        n = int(fluid.sum())
        fi, fj = np.nonzero(fluid)

        # --- 1. ポテンシャル流: ∇²φ = 0、固体の面で流れ 0、左右は φ = x（U = 1）、上下は流れ 0
        rows, cols, vals = [], [], []
        rhs = np.zeros(n)
        diag = np.zeros(n)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = fi + di, fj + dj
            inside = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
            c = idx[fi, fj]
            # 格子の中のとなり
            m = inside.copy()
            m[inside] &= fluid[ni[inside], nj[inside]]
            rows.append(c[m]); cols.append(idx[ni[m], nj[m]]); vals.append(np.ones(m.sum()))
            diag[c[m]] -= 1
            # 左右のはし: φ = x（面までの距離 h/2）
            if di != 0:
                mb = ~inside
                xb = cx[fi[mb]] + di * h / 2
                diag[c[mb]] -= 2
                rhs[c[mb]] -= 2 * xb
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        phi = spla.spsolve(A, rhs)
        P = np.full((nx, ny), np.nan)
        P[fluid] = phi

        # 面の速さ（x の面: (nx+1, ny)、y の面: (nx, ny+1)）
        ux = np.zeros((nx + 1, ny))
        uy = np.zeros((nx, ny + 1))
        both = fluid[1:, :] & fluid[:-1, :]
        ux[1:-1][both] = (P[1:, :][both] - P[:-1, :][both]) / h
        ux[0, fluid[0, :]] = 1.0
        ux[-1, fluid[-1, :]] = 1.0
        bothy = fluid[:, 1:] & fluid[:, :-1]
        uy[:, 1:-1][bothy] = (P[:, 1:][bothy] - P[:, :-1][bothy]) / h

        # --- 2. 温度: u·∇T = (1/Pe)∇²T。ペンギン T = 1、左と上下 T = 0、右は流れ出るだけ
        D = length / pe
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        rhs = np.zeros(n)
        c = idx[fi, fj]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = fi + di, fj + dj
            if di == 1:
                u = ux[fi + 1, fj]
            elif di == -1:
                u = -ux[fi, fj]
            elif dj == 1:
                u = uy[fi, fj + 1]
            else:
                u = -uy[fi, fj]
            inside = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
            nfl = inside.copy()
            nfl[inside] &= fluid[ni[inside], nj[inside]]
            nso = inside & ~nfl
            out = ~inside
            # 外へ出る量 = 移流（風上の値）+ 拡散。面の長さ h で割ってある
            # となりも空気
            m = nfl
            diag[c[m]] += np.maximum(u[m], 0) + D / h
            rows.append(c[m]); cols.append(idx[ni[m], nj[m]]); vals.append(np.minimum(u[m], 0) - D / h)
            # となりがペンギン（T = 1、面までの距離 h/2）
            m = nso
            diag[c[m]] += 2 * D / h
            rhs[c[m]] += 2 * D / h
            # 格子のはし
            m = out
            if di == 1:  # 右: 流れ出るだけ
                diag[c[m]] += np.maximum(u[m], 0)
            else:  # 左・上下: T = 0
                diag[c[m]] += np.maximum(u[m], 0) + 2 * D / h
        rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        T = spla.spsolve(A, rhs)
        TT = np.ones((nx, ny))
        TT[fluid] = T

        # --- 3. ペンギンごとの熱の失い方（固体の面を通って出ていく拡散の量）
        q = np.zeros(len(xs))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = fi + di, fj + dj
            inside = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
            m = inside.copy()
            m[inside] &= solid[ni[inside], nj[inside]]
            flux = 2 * D * (1 - TT[fi[m], fj[m]])  # 面の長さ h × 勾配 (1-T)/(h/2)
            np.add.at(q, owner[ni[m], nj[m]], flux)

        return {"q": q, "T": TT, "phi": P, "solid": solid, "x0": x0, "y0": y0, "h": h, "ux": ux, "uy": uy}


class Huddle:
    def __init__(self, n, pe=10.0, noise=0.0, seed=0, flow: Flow | None = None, select="selfish"):
        """select: "selfish" = 論文の規則 / "random" = 比べるため。ふちの子をでたらめに選び、でたらめな空いたマスへ"""
        self.select = select
        self.rng = np.random.default_rng(seed)
        self.pe = pe
        self.noise = noise
        self.flow = flow or Flow()
        self.sites: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 1), (1, -1), (-1, 1)]
        # 論文と同じく、5 羽から、2 羽以上ととなり合う空いたマスへ 1 羽ずつ足していく
        while len(self.sites) < n:
            occ = set(self.sites)
            cand = sorted({(q + a, r + b) for q, r in self.sites for a, b in DIRS} - occ)
            cand = [c for c in cand if self._nb(c, occ) >= 2]
            self.sites.append(cand[self.rng.integers(len(cand))])
        self.radius_eq = float(np.sqrt(n * SQ3 / 2 / np.pi))  # 群れと同じ面積の円の半径
        self.q_amb = self._ambient()
        self.step_no = 0

    @staticmethod
    def _nb(site, occ):
        q, r = site
        return sum((q + a, r + b) in occ for a, b in DIRS)

    def _ambient(self):
        """1 羽だけのときの熱の失い方（ゆらぎの大きさの基準）"""
        res = self.flow.solve(np.array([0.0]), np.array([0.0]), self.pe, length=self.radius_eq)
        return float(res["q"][0])

    @staticmethod
    def _intri(site, occ):
        """となりの 2 羽どうしも接している（三角形になる）か"""
        q, r = site
        nb = [(q + a, r + b) in occ for a, b in DIRS]
        return any(nb[k] and nb[(k + 1) % 6] for k in range(6))

    def xy(self):
        q, r = np.array(self.sites).T
        return axial_xy(q, r)

    def triangles(self):
        """3 羽ずつ互いにとなり合う組（上向きと下向きの三角形）"""
        pos = {s: i for i, s in enumerate(self.sites)}
        out = []
        for (q, r), i in pos.items():
            for (a1, b1), (a2, b2) in (((1, 0), (0, 1)), ((1, 0), (1, -1))):
                j, k = pos.get((q + a1, r + b1)), pos.get((q + a2, r + b2))
                if j is not None and k is not None:
                    out.append((i, j, k))
        return out

    def boundary(self):
        occ = set(self.sites)
        return [i for i, s in enumerate(self.sites) if self._nb(s, occ) < 6]

    def step(self):
        """1 羽を動かす。動いた子の番号・前と後のマス・そのときの Q を返す"""
        xs, ys = self.xy()
        b = self.boundary()
        res = self.flow.solve(xs, ys, self.pe, tris=self.triangles(), edge=b, length=self.radius_eq)
        q = res["q"]
        eff = q.copy()
        eff[b] += self.noise * self.q_amb * self.rng.uniform(-0.5, 0.5, len(b))
        if self.select == "random":
            mover = b[self.rng.integers(len(b))]
            old = self.sites[mover]
            occ = set(self.sites) - {old}
            cand = sorted({(qq + a, rr + bb) for qq, rr in occ for a, bb in DIRS} - occ - {old})
            cand = [c for c in cand if self._nb(c, occ) >= 2 and self._intri(c, occ)]
            new = cand[self.rng.integers(len(cand))]
            self.sites[mover] = new
            self.step_no += 1
            return {"mover": mover, "old": old, "new": new, "q": q, "field": res}
        mover = max(b, key=lambda i: eff[i])
        old = self.sites[mover]
        occ = set(self.sites) - {old}
        rest = [i for i in b if i != mover]
        first = min(rest, key=lambda i: eff[i])
        fq, fr = self.sites[first]
        adj = [i for i in rest if (self.sites[i][0] - fq, self.sites[i][1] - fr) in DIRS]
        new = None
        for second in sorted(adj, key=lambda i: eff[i]):
            sq, sr = self.sites[second]
            common = {(fq + a, fr + b) for a, b in DIRS} & {(sq + a, sr + b) for a, b in DIRS}
            empty = [c for c in common if c not in occ and c != old]
            if empty:
                new = empty[0]
                break
        if new is None:  # 論文に書かれていない場合: first のとなりで、となりがいちばん多い空いたマス
            empty = [(fq + a, fr + b) for a, b in DIRS if (fq + a, fr + b) not in occ and (fq + a, fr + b) != old]
            new = max(empty, key=lambda c: self._nb(c, occ))
        self.sites[mover] = new
        self.step_no += 1
        return {"mover": mover, "old": old, "new": new, "q": q, "field": res}
