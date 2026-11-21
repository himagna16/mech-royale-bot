"""Arena geometry shared by the bot and the local simulator.

Everything is sampled on the 1 m terrain vertex grid (see tools/build_grid.py):
  ground    terrain height
  solid_top top of the ground-standing solid over a point (buildings, stadium), -BIG if open
  over_*    underside/top of an elevated solid (bridge, overhang), +/-BIG if none
  wall_*    the same, dilated by the mech radius, for cylinder-vs-wall checks
"""
import json
import math
import os

import numpy as np

BIG = 1e9
NONE = -32768
HERE = os.path.dirname(os.path.abspath(__file__))


class Arena:
    def __init__(self, map_dir, grid_path=None):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            world = json.load(f)
        t = world["terrain"]
        self.minx, self.miny = float(t["minX"]), float(t["minY"])
        self.nx, self.ny = int(t["nx"]), int(t["ny"])
        self.bounds = world["bounds"]
        rules = world.get("rules", {})
        self.canopy_density = float(rules.get("canopyDensity", 0.35))
        self.canopy_block_t = float(rules.get("canopyBlockT", 0.25))
        raw = np.fromfile(os.path.join(map_dir, "v2", "terrain.i16"), dtype="<i2")
        self.ground = raw.reshape(self.ny, self.nx).astype(np.float32) * float(t.get("scale", 0.01))
        g = np.load(grid_path or os.path.join(HERE, "grid.npz"))

        def unpack(name, fill):
            a = g[name]
            f = a.astype(np.float32) / 100.0
            f[a == NONE] = fill
            return f

        self.solid_top = unpack("solid_top", -BIG)
        self.over_base = unpack("over_base", BIG)
        self.over_top = unpack("over_top", -BIG)
        self.wall_top = unpack("wall_top", -BIG)
        self.wall_over = unpack("wall_over", BIG)
        self.trees = g["trees"].astype(np.float64)  # cx, cy, cz, R, rz, x, y, trunkR, z, height
        self.surf = np.maximum(self.ground, self.solid_top)  # roof-or-ground, ignoring overhangs
        self.is_solid = self.solid_top > self.ground + 0.05

    # ---- scalar lookups -------------------------------------------------
    def ij(self, x, y):
        i = int(x - self.minx + 0.5)
        j = int(y - self.miny + 0.5)
        if i < 0:
            i = 0
        elif i >= self.nx:
            i = self.nx - 1
        if j < 0:
            j = 0
        elif j >= self.ny:
            j = self.ny - 1
        return i, j

    def ground_at(self, x, y):
        fx = min(max(x - self.minx, 0.0), self.nx - 1.001)
        fy = min(max(y - self.miny, 0.0), self.ny - 1.001)
        i, j = int(fx), int(fy)
        u, v = fx - i, fy - j
        g = self.ground
        return float(
            g[j, i] * (1 - u) * (1 - v) + g[j, i + 1] * u * (1 - v) + g[j + 1, i] * (1 - u) * v + g[j + 1, i + 1] * u * v
        )

    def in_bounds(self, x, y, margin=0.0):
        b = self.bounds
        return b["minX"] + margin <= x <= b["maxX"] - margin and b["minY"] + margin <= y <= b["maxY"] - margin

    def floor_at(self, x, y, z, step_up=1.2):
        """Highest standable surface at (x, y) that feet at height z can be on (<= z + step_up)."""
        i, j = self.ij(x, y)
        best = self.ground_at(x, y)
        lim = z + step_up
        s = float(self.solid_top[j, i])
        if best < s <= lim:
            best = s
        o = float(self.over_top[j, i])
        if best < o <= lim:
            best = o
        return best

    def top_at(self, x, y):
        """Roof (or ground) height at a point, ignoring overhangs."""
        i, j = self.ij(x, y)
        return float(self.surf[j, i])

    def blocked(self, x, y, z, height=9.0, step_up=1.2, radius_margin=2.5):
        """Would a mech cylinder with feet at z collide with a wall at (x, y)?"""
        if not self.in_bounds(x, y, radius_margin):
            return True
        i, j = self.ij(x, y)
        if self.wall_top[j, i] > z + step_up:
            return True
        ob = self.wall_over[j, i]
        if ob < z + height and ob > z + step_up:
            return True
        return False

    # ---- rays -----------------------------------------------------------
    def _samples(self, p, q, step):
        dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = max(2, int(d / step) + 2)
        t = np.linspace(0.0, 1.0, n)
        xs = p[0] + dx * t
        ys = p[1] + dy * t
        zs = p[2] + dz * t
        ii = np.clip((xs - self.minx + 0.5).astype(np.int32), 0, self.nx - 1)
        jj = np.clip((ys - self.miny + 0.5).astype(np.int32), 0, self.ny - 1)
        return t, d, ii, jj, zs

    def _hits(self, ii, jj, zs):
        return (zs < self.surf[jj, ii]) | ((zs > self.over_base[jj, ii]) & (zs < self.over_top[jj, ii]))

    def los(self, p, q, step=1.0):
        """True if the segment p -> q clears terrain, buildings and bridges."""
        _, _, ii, jj, zs = self._samples(p, q, step)
        return not bool(self._hits(ii, jj, zs).any())

    def ray_dist(self, p, q, step=0.75):
        """Distance along p -> q to the first solid, or the full length if clear."""
        t, d, ii, jj, zs = self._samples(p, q, step)
        h = self._hits(ii, jj, zs)
        k = int(h.argmax())
        if not h[k]:
            return d
        return float(t[max(k - 1, 0)] * d)

    def canopy_len(self, p, q):
        """Metres of the segment p -> q that pass through tree crowns."""
        tr = self.trees
        dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-6:
            return 0.0
        # cheap reject: crowns farther from the segment's midpoint than half its length + radius
        mx, my = (p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5
        near = (np.abs(tr[:, 0] - mx) < d * 0.5 + tr[:, 3]) & (np.abs(tr[:, 1] - my) < d * 0.5 + tr[:, 3])
        if not near.any():
            return 0.0
        c = tr[near]
        ux, uy, uz = (p[0] - c[:, 0]) / c[:, 3], (p[1] - c[:, 1]) / c[:, 3], (p[2] - c[:, 2]) / c[:, 4]
        vx, vy, vz = dx / c[:, 3], dy / c[:, 3], dz / c[:, 4]
        a = vx * vx + vy * vy + vz * vz
        b = 2 * (ux * vx + uy * vy + uz * vz)
        cc = ux * ux + uy * uy + uz * uz - 1.0
        disc = b * b - 4 * a * cc
        ok = disc > 0
        if not ok.any():
            return 0.0
        sq = np.sqrt(disc[ok])
        t1 = np.clip((-b[ok] - sq) / (2 * a[ok]), 0.0, 1.0)
        t2 = np.clip((-b[ok] + sq) / (2 * a[ok]), 0.0, 1.0)
        return float((t2 - t1).sum() * d)

    def sees(self, eye, target):
        """Line of sight including canopy fade."""
        if not self.los(eye, target):
            return False
        return math.exp(-self.canopy_density * self.canopy_len(eye, target)) >= self.canopy_block_t
