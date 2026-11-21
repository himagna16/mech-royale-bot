"""Ground-level navigation: a coarse walkability grid and numpy wavefront distance fields."""
import math

import numpy as np

INF = 1e9


class Nav:
    def __init__(self, arena, cell=4, step_up=1.2, height=9.0, max_slope_deg=50.0):
        self.arena = arena
        self.cell = cell
        a = arena
        g = a.ground
        blocked = a.wall_top > g + step_up
        blocked |= (a.wall_over < g + height) & (a.wall_over > g + step_up)
        tan = math.tan(math.radians(max_slope_deg))
        steep = np.zeros_like(blocked)
        dx = np.abs(np.diff(g, axis=1)) > tan
        dy = np.abs(np.diff(g, axis=0)) > tan
        steep[:, :-1] |= dx
        steep[:, 1:] |= dx
        steep[:-1, :] |= dy
        steep[1:, :] |= dy
        blocked |= steep
        # keep the mech radius inside the map bounds
        b = a.bounds
        xs = a.minx + np.arange(a.nx)
        ys = a.miny + np.arange(a.ny)
        blocked[:, (xs < b["minX"] + 3) | (xs > b["maxX"] - 3)] = True
        blocked[(ys < b["minY"] + 3) | (ys > b["maxY"] - 3), :] = True
        self.blocked1 = blocked
        h = cell // 2
        self.free = ~blocked[h::cell, h::cell]
        self.ny, self.nx = self.free.shape
        # links between neighbouring cells: the straight 1 m samples between the two centres must be clear
        le = self.free[:, :-1] & self.free[:, 1:]
        ln = self.free[:-1, :] & self.free[1:, :]
        for k in range(1, cell):
            le &= ~blocked[h::cell, h + k::cell][:, : le.shape[1]]
            ln &= ~blocked[h + k::cell, h::cell][: ln.shape[0], :]
        self.link_e = le  # (ny, nx-1): cell (j,i) <-> (j,i+1)
        self.link_n = ln  # (ny-1, nx): cell (j,i) <-> (j+1,i)
        self.x0 = a.minx + h
        self.y0 = a.miny + h

    def cell_of(self, x, y):
        i = int(round((x - self.x0) / self.cell))
        j = int(round((y - self.y0) / self.cell))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def center(self, i, j):
        return self.x0 + i * self.cell, self.y0 + j * self.cell

    def nearest_free(self, i, j, rmax=12):
        if self.free[j, i]:
            return i, j
        best, bd = None, 1e9
        for r in range(1, rmax + 1):
            j0, j1 = max(0, j - r), min(self.ny, j + r + 1)
            i0, i1 = max(0, i - r), min(self.nx, i + r + 1)
            sub = self.free[j0:j1, i0:i1]
            if sub.any():
                jj, ii = np.nonzero(sub)
                d = (jj + j0 - j) ** 2 + (ii + i0 - i) ** 2
                k = int(d.argmin())
                if d[k] < bd:
                    bd, best = d[k], (int(ii[k] + i0), int(jj[k] + j0))
                break
        return best if best else (i, j)

    def field(self, gx, gy, max_iter=900, until=None):
        """Walking distance in cells (4-connected, so every reachable cell has a strictly lower neighbour)."""
        free = self.free
        dist = np.full(free.shape, INF, dtype=np.float32)
        i, j = self.nearest_free(*self.cell_of(gx, gy))
        front = np.zeros(free.shape, dtype=bool)
        front[j, i] = True
        dist[j, i] = 0.0
        open_ = free.copy()
        open_[j, i] = False
        d = 0.0
        le, ln = self.link_e, self.link_n
        stop_at = None
        if until is not None:  # only needed around one walker: stop soon after the wave passes it
            ui, uj = self.nearest_free(*self.cell_of(until[0], until[1]))
        for it in range(max_iter):
            if until is not None:
                if stop_at is None and dist[uj, ui] < INF:
                    stop_at = it + 6
                if stop_at is not None and it >= stop_at:
                    break
            nb = np.zeros_like(front)
            nb[1:, :] |= front[:-1, :] & ln
            nb[:-1, :] |= front[1:, :] & ln
            nb[:, 1:] |= front[:, :-1] & le
            nb[:, :-1] |= front[:, 1:] & le
            nb &= open_
            if not nb.any():
                break
            d += 1.0
            dist[nb] = d
            open_ &= ~nb
            front = nb
        return dist

    def linked(self, i, j, di, dj):
        if di == 1:
            return i < self.nx - 1 and bool(self.link_e[j, i])
        if di == -1:
            return i > 0 and bool(self.link_e[j, i - 1])
        if dj == 1:
            return j < self.ny - 1 and bool(self.link_n[j, i])
        return j > 0 and bool(self.link_n[j - 1, i])

    def _clear(self, x0, y0, x1, y1):
        """Is the straight walk between two points free of walls (1 m samples)?"""
        a = self.arena
        d = math.hypot(x1 - x0, y1 - y0)
        n = int(d) + 1
        b = self.blocked1
        for k in range(1, n + 1):
            t = k / n
            i = int(x0 + (x1 - x0) * t - a.minx + 0.5)
            j = int(y0 + (y1 - y0) * t - a.miny + 0.5)
            if i < 0 or j < 0 or i >= a.nx or j >= a.ny or b[j, i]:
                return False
        return True

    def step_dir(self, field, x, y, look=7):
        """Unit direction to walk from (x, y) toward the field's goal; (0, 0) at the goal or if unreachable."""
        i, j = self.cell_of(x, y)
        free = self.free
        if not free[j, i] or field[j, i] >= INF:
            i2, j2 = self.nearest_free(i, j)
            cx, cy = self.center(i2, j2)
            dx, dy = cx - x, cy - y
            n = math.hypot(dx, dy)
            return (dx / n, dy / n) if n > 0.5 else (0.0, 0.0)
        if field[j, i] <= 0:
            return 0.0, 0.0
        tx, ty = None, None
        ci, cj = i, j
        for _ in range(look):
            v = float(field[cj, ci])
            if v <= 0:
                break
            best = None
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < self.nx and 0 <= jj < self.ny and field[jj, ii] < v and self.linked(ci, cj, di, dj):
                    # prefer continuing diagonally when both sides are open
                    w = float(field[jj, ii])
                    if best is None or w < best[0]:
                        best = (w, ii, jj)
            if best is None:
                break
            ci, cj = best[1], best[2]
            cx, cy = self.center(ci, cj)
            if tx is None or self._clear(x, y, cx, cy):
                tx, ty = cx, cy
            else:
                break
        if tx is None:
            return 0.0, 0.0
        dx, dy = tx - x, ty - y
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)

    def walk_dist(self, field, x, y):
        i, j = self.cell_of(x, y)
        return float(field[j, i]) * self.cell
