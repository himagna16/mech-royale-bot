"""Mech Royale bot: decision logic. `Bot(hello)` then `act(obs) -> action` ten times a second."""
import math
import os
import random

import numpy as np

from geom import Arena
from nav import Nav

TAU = 2 * math.pi
WEAPON_RANK = {"laser": 3.2, "lightning": 2.5, "mortar": 1, "gun": 0}
UPGRADE_ORDER = [
    "plating", "armor", "nanites", "lightning_efficiency", "laser_damage", "laser_charge", "reactor", "capacitor",
    "lightning_chain", "lightning_range", "gun_damage", "servos", "gun_mag", "gun_velocity", "laser_pierce",
    "jets", "gyros", "sensors", "gun_rate", "mortar_reload", "mortar_splash", "mortar_velocity",
]

# tunables (the MR_* environment overrides exist only for local experiments)
def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


FIGHT_RANGE = _env("MR_RANGE", 72.0)      # preferred duel distance (m)
RETREAT_HP = _env("MR_RETREAT", 0.35)     # break contact below this health fraction
REJOIN_HP = _env("MR_REJOIN", 0.9)       # ... and come back above this one
ZONE_MARGIN = 18.0
FARM = _env("MR_FARM", 2.0)
SWAP_GUN = _env("MR_SWAPGUN", 0.0) > 0     # replace the starting gun once we can carry laser + lightning
LASER_RANGE = _env("MR_LRANGE", 110.0)
ANGLE = _env("MR_ANGLE", 0.0) > 0
ROAM = _env("MR_ROAM", 1.0) > 0
BUNKER = _env("MR_BUNKER", 0.0) > 0        # hold a tucked-away spot and make hunters come to us
BUNKER_LEASH = _env("MR_LEASH", 16.0)
GRAB = _env("MR_GRAB", 70.0)
RUSH = _env("MR_RUSH", 1.0) > 0
BREAK_N = _env("MR_BREAKN", 8.0)          # break off on numbers alone only against this many attackers
DODGE = _env("MR_DODGE", 1.0) > 0
FFA_DEATHS = _env("MR_FFADEATHS", 4.0)    # deaths we did not cause before we believe it is a free-for-all
FARM_HP = _env("MR_FARMHP", 0.8)
PURSUE = _env("MR_PURSUE", 1.0) > 0
DROPS = _env("MR_DROPS", 0.0)           # go for supply drops within this many metres (0 = only when idle)
ZAP_RANGE = _env("MR_ZRANGE", 28.0)
LATENCY = _env("MR_LAT", 0.1)
ACCEL_LEAD = _env("MR_ALEAD", 0.5)


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def intercept(dx, dy, dz, vx, vy, vz, speed):
    """Time for a shot of `speed` to meet a target at offset d moving at v (constant velocity)."""
    a = vx * vx + vy * vy + vz * vz - speed * speed
    b = 2 * (dx * vx + dy * vy + dz * vz)
    c = dx * dx + dy * dy + dz * dz
    if abs(a) < 1e-9:
        return -c / b if b < 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None


class Bot:
    name = "longhorn"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.me_id = hello["slot"]
        self.rng = random.Random(hello.get("seed", 1))
        self.arena = Arena(hello["map"]["dir"])
        self.nav = Nav(self.arena)
        try:
            self.exposure = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exposure.npy")).astype(np.float32)
        except Exception:
            self.exposure = np.full(self.nav.free.shape, 120.0, dtype=np.float32)
        self.cell_x = (self.nav.x0 + np.arange(self.nav.nx) * self.nav.cell)[None, :]
        self.cell_y = (self.nav.y0 + np.arange(self.nav.ny) * self.nav.cell)[:, None]
        self.hide = None
        self.hide_t = -100.0
        self.hide_key = None
        self.mode = "idle"
        self.zone_field = None
        self.zone_key = None
        self.goal_field = None
        self.goal_key = None
        self.goal_t = -100.0
        self.strafe = 1.0
        self.strafe_until = 0.0
        self.retreating = False
        self.threat_bearing = None
        self.threat_t = -100.0
        self.known = {}          # enemy id -> last obs dict + 't'
        self.pick_memo = {}      # pickup id -> (kind, pos)
        self.cube_memo = {}      # cube id -> pos
        self.farm_memo = {}      # crate / drone id -> (pos, height)
        self.rear_t, self.rear_bearing, self.rear_by = -100.0, 0.0, None
        self.angle_pt, self.angle_t = None, -100.0
        self.zapping = False
        self.ffa_latched = False
        self.prev_vel = {}       # mech id -> (t, velocity) from the previous observation
        self.hit_by = {}         # mech id -> last time it damaged us
        self.tgt_id = None
        self.now = 0.0
        self.my_kills = 0
        self.deaths_seen = 0
        self.last_alive = None
        self.scan_dir = 1.0
        self.last_move = (1.0, 0.0)
        self.stuck_t = 0.0
        self.last_pos = None
        self.fields_left = 1

    # ------------------------------------------------------------------ navigation helpers
    def field_to(self, x, y, t, me):
        """Distance field to a nearby goal. At most one field is built per decision (they cost ~10 ms)."""
        key = (round(x / 6), round(y / 6))
        stale = self.goal_field is None or (key != self.goal_key and t - self.goal_t >= 0.5) or t - self.goal_t > 3.0
        if stale and self.fields_left > 0:
            self.fields_left -= 1
            self.goal_field = self.nav.field(x, y, until=(me["pos"][0], me["pos"][1]))
            self.goal_key = key
            self.goal_t = t
        return self.goal_field if self.goal_key == key else None

    def go(self, x, y, me, t):
        """Unit move toward a point, around buildings."""
        mx, my = me["pos"][0], me["pos"][1]
        d = math.hypot(x - mx, y - my)
        if d < 1.0:
            return (0.0, 0.0)
        if d < 30 and self.nav._clear(mx, my, x, y):
            return ((x - mx) / d, (y - my) / d)
        f = self.field_to(x, y, t, me)
        if f is None:
            return ((x - mx) / d, (y - my) / d)
        mv = self.nav.step_dir(f, mx, my)
        if mv == (0.0, 0.0):
            return ((x - mx) / d, (y - my) / d)
        return mv

    def hide_spot(self, obs, me, t):
        """A tucked-away walkable cell inside the next circle, not too far from us. Cheap: one numpy pass."""
        z = obs["zone"]
        key = (round(z["next_center"][0]), round(z["next_center"][1]), z["next_radius"])
        if self.hide is not None and key == self.hide_key and t - self.hide_t < 6.0:
            return self.hide
        x, y = me["pos"][0], me["pos"][1]
        d_me = np.hypot(self.cell_x - x, self.cell_y - y)
        d_zone = np.hypot(self.cell_x - z["next_center"][0], self.cell_y - z["next_center"][1])
        inner = max(z["next_radius"] - 22.0, z["next_radius"] * 0.5)
        cost = self.exposure + 0.22 * d_me + np.maximum(0.0, d_zone - inner) * 6.0
        if self.hide is not None:
            cost = cost + 12.0 * (np.hypot(self.cell_x - self.hide[0], self.cell_y - self.hide[1]) > 10.0)  # stay put unless clearly better
        cost[~self.nav.free] = 1e9
        j, i = np.unravel_index(int(cost.argmin()), cost.shape)
        self.hide = self.nav.center(int(i), int(j))
        self.hide_key, self.hide_t = key, t
        return self.hide

    def zone_dir(self, obs, me):
        z = obs["zone"]
        key = (round(z["next_center"][0] / 4), round(z["next_center"][1] / 4))
        if key != self.zone_key and self.fields_left > 0:
            self.fields_left -= 1
            self.zone_key = key
            self.zone_field = self.nav.field(*z["next_center"])
        mv = self.nav.step_dir(self.zone_field, me["pos"][0], me["pos"][1]) if self.zone_field is not None else (0.0, 0.0)
        if mv == (0.0, 0.0):
            dx, dy = z["next_center"][0] - me["pos"][0], z["next_center"][1] - me["pos"][1]
            n = math.hypot(dx, dy)
            mv = (dx / n, dy / n) if n > 1 else (0.0, 0.0)
        return mv

    def open_dir(self, me, mv):
        """Keep a strafe out of walls: return mv if the next few metres are walkable, else None."""
        x, y, z = me["pos"]
        for k in (3.0, 6.0):
            if self.arena.blocked(x + mv[0] * k, y + mv[1] * k, z):
                return None
        return mv

    # ------------------------------------------------------------------ aiming
    def aim_at(self, me, tgt, speed=None, height=None):
        """(yaw, pitch, distance, lead point) to hit tgt; speed None = hitscan.

        The shot leaves about LATENCY seconds after this observation (the action applies on the next
        ticks), so both we and the target are advanced by that much before solving the intercept."""
        x, y, z = me["pos"]
        mvx, mvy = me["vel"][0], me["vel"][1]
        x, y = x + mvx * LATENCY, y + mvy * LATENCY
        mz = z + me["stats"].get("muzzle_height", 7.0)
        h = height if height is not None else tgt.get("h", 9.0 * (1 + 0.05 * tgt.get("cubes", 0)))
        vx, vy, vz = tgt.get("vel", (0, 0, 0))
        vz_use = vz if abs(vz) > 0.5 else 0.0   # a grounded target does not sink: ignore tiny vertical speed
        tx = tgt["pos"][0] + vx * LATENCY
        ty = tgt["pos"][1] + vy * LATENCY
        tz = tgt["pos"][2] + h * tgt.get("frac", 0.5) + vz_use * LATENCY
        if speed:
            tt = intercept(tx - x, ty - y, tz - mz, vx, vy, vz_use, speed)
            if tt is not None:
                tt = min(tt, 1.2)
                ddx, ddy = self.drift(tgt, tt)
                # one refinement: flight time to where the target will actually be
                tt = min(1.2, math.sqrt((tx + ddx - x) ** 2 + (ty + ddy - y) ** 2 + (tz - mz) ** 2) / speed)
                ddx, ddy = self.drift(tgt, tt)
                tx, ty, tz = tx + ddx, ty + ddy, tz + vz_use * tt
        hd = math.hypot(tx - x, ty - y)
        return math.atan2(ty - y, tx - x), math.atan2(tz - mz, max(hd, 0.1)), math.hypot(hd, tz - mz), (tx, ty, tz)

    def drift(self, tgt, horizon):
        """Where a target ends up after `horizon` seconds, relative to now.

        Dodging bots keep accelerating toward a new velocity, so plain velocity x time overshoots.
        Checked against the recorded real match: half the observed acceleration, speed capped, hits more."""
        vx, vy = tgt.get("vel", (0, 0, 0))[:2]
        prev = self.prev_vel.get(tgt.get("id"))
        ax = ay = 0.0
        if ACCEL_LEAD and prev is not None and 0.0 < self.now - prev[0] <= 0.11:
            dt = self.now - prev[0]
            ax, ay = (vx - prev[1][0]) / dt * ACCEL_LEAD, (vy - prev[1][1]) / dt * ACCEL_LEAD
        if ax == 0.0 and ay == 0.0:
            return vx * horizon, vy * horizon
        cap = max(11.7, math.hypot(vx, vy))
        x = y = 0.0
        left = horizon
        while left > 1e-6:
            h = 0.05 if left > 0.05 else left
            vx += ax * h
            vy += ay * h
            sp = math.hypot(vx, vy)
            if sp > cap:
                vx, vy = vx * cap / sp, vy * cap / sp
            x += vx * h
            y += vy * h
            left -= h
        return x, y

    # ------------------------------------------------------------------ main
    def act(self, obs):
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        st = me["stats"]
        act = {"tick": obs["tick"], "move": [0.0, 0.0], "fire": [False, False]}
        hp_frac = me["hp"] / max(me["max_hp"], 1)
        zone = obs["zone"]
        self.fields_left = 1
        self.now = t

        # ---- bookkeeping
        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("rear") and ev.get("bearing") is not None:
                self.rear_t, self.rear_bearing, self.rear_by = t, ev["bearing"], ev.get("by")
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                self.threat_bearing, self.threat_t = ev["bearing"], t
                if isinstance(ev.get("by"), int) and ev["by"] < 100:
                    self.hit_by[ev["by"]] = t
            elif ev["type"] == "kill":
                self.my_kills += 1
        if self.last_alive is not None and obs["alive"] < self.last_alive:
            self.deaths_seen += self.last_alive - obs["alive"]
        self.last_alive = obs["alive"]
        other_deaths = self.deaths_seen - self.my_kills
        if other_deaths >= FFA_DEATHS or (other_deaths >= 2 and t < 45) or (other_deaths >= 3 and t < 90):
            self.ffa_latched = True      # others are dying without us: a free-for-all, not the practice squad
        ffa = self.ffa_latched
        for p in obs["pickups"]:
            self.pick_memo[p["id"]] = (p["kind"], p["pos"])
        for c in obs["crates"]:
            self.farm_memo[c["id"]] = (c["pos"], 2.4)
        for n in obs["mechs"]:
            if n["kind"] == "npc":
                self.farm_memo[n["id"]] = (n["pos"], 3.5)
        seen_farm = {c["id"] for c in obs["crates"]} | {n["id"] for n in obs["mechs"] if n["kind"] == "npc"}
        for fid, (fpos, _h) in list(self.farm_memo.items()):
            # gone if we are close with a clear view and it is not listed any more
            if fid not in seen_farm and math.hypot(fpos[0] - x, fpos[1] - y) < 40 and self.arena.los(
                    (x, y, z + 8.0), (fpos[0], fpos[1], fpos[2] + 1.5)):
                del self.farm_memo[fid]
        seen_cubes = {c["id"] for c in obs["cubes"]}
        for c in obs["cubes"]:
            self.cube_memo[c["id"]] = c["pos"]
        for cid, cpos in list(self.cube_memo.items()):
            # forget a remembered cube once we are standing where it was and it is not there
            if cid not in seen_cubes and math.hypot(cpos[0] - x, cpos[1] - y) < 12:
                del self.cube_memo[cid]
        for ev in obs["events"]:
            if ev["type"] in ("took", "take_failed") and ev.get("reason") in (None, "gone"):
                self.pick_memo.pop(ev["id"], None)

        if obs["offer"]:
            ids = [c["id"] for c in obs["offer"]["choices"]]
            order = UPGRADE_ORDER
            act["pick"] = min(range(len(ids)), key=lambda i: order.index(ids[i]) if ids[i] in order else 99)

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        eye = (x, y, z + st.get("eye_height", 8.0))
        for e in enemies:
            e["d"] = math.dist(e["pos"], me["pos"])
            eh = 9.0 * (1 + 0.05 * e.get("cubes", 0))
            muz = (x, y, z + st.get("muzzle_height", 7.0))
            if self.arena.los(muz, (e["pos"][0], e["pos"][1], e["pos"][2] + eh * 0.5)):
                e["los"], e["frac"] = True, 0.5
            elif self.arena.los(muz, (e["pos"][0], e["pos"][1], e["pos"][2] + eh * 0.85)):
                e["los"], e["frac"] = True, 0.85     # only the upper body shows over cover: shoot that
            else:
                e["los"] = False
        enemies.sort(key=lambda e: e["d"])
        visible = [e for e in enemies if e["los"]]

        # ---- weapons on the ground
        slots = me["weapons"]
        held = [w["kind"] if w else None for w in slots]
        ranks = [WEAPON_RANK.get(k, 0) if k else -1 for k in held]
        low_slot = 0 if ranks[0] <= ranks[1] else 1          # the slot a new weapon would go into
        low_rank = ranks[low_slot]
        if not SWAP_GUN and held[0] == "gun" and held[1] is not None:
            low_slot, low_rank = 1, ranks[1]
        want = None
        for pid, (kind, pos) in self.pick_memo.items():
            if kind in held or WEAPON_RANK.get(kind, 0) <= low_rank:
                continue
            d = math.hypot(pos[0] - x, pos[1] - y)
            score = d - 25 * WEAPON_RANK[kind]
            if abs(pos[2] - self.arena.ground_at(pos[0], pos[1])) > 2.0:
                continue  # on a roof: skip for now
            if want is None or score < want[0]:
                want = (score, pid, kind, pos, d)
        if want and want[4] < 3.6 and abs(want[3][2] - z) < 4:
            act["take"] = want[1]
            if held[low_slot] is not None:
                act["drop"] = low_slot

        # ---- zone pressure
        zc = zone["next_center"]
        dz_next = math.hypot(x - zc[0], y - zc[1])
        final = zone["next_radius"] <= 30
        margin = min(ZONE_MARGIN, zone["next_radius"] * 0.5)
        outside_next = dz_next > zone["next_radius"] - margin
        time_left = zone["shrink_end"] - t
        need = (dz_next - (zone["next_radius"] - margin)) / (st["walk_speed"] * 0.8)
        must_move = (outside_next and need > time_left - 12) or not me["in_zone"]

        # ---- choose a target: stay on the same mech, answer whoever is hitting us, prefer weak and near
        tgt, best = None, None
        for e in enemies:
            sc = e["d"] + 60.0 * e.get("hp", 1400) / max(e.get("max_hp", 1400), 1) + (0.0 if e["los"] else 45.0)
            if t - self.hit_by.get(e["id"], -100.0) < 3.0:
                sc -= 45.0
            if e["id"] == self.tgt_id:
                sc -= 35.0
            if best is None or sc < best:
                best, tgt = sc, e
        self.tgt_id = tgt["id"] if tgt is not None else None
        n_threat = len([e for e in visible if e["d"] < 170])
        tfrac = tgt.get("hp", 1400) / max(tgt.get("max_hp", 1400), 1) if tgt is not None else 1.0
        losing = (hp_frac < RETREAT_HP and (tgt is None or tfrac > hp_frac + 0.1)) or (n_threat >= 2 and hp_frac < 0.5) \
            or (n_threat >= BREAK_N and hp_frac < 0.75)
        finishing = tgt is not None and tgt["los"] and tfrac < 0.2 and hp_frac > 0.2 and n_threat <= 2
        if finishing:
            self.retreating = False      # never walk away from a kill
        elif losing:
            self.retreating = True
        elif hp_frac > REJOIN_HP or (not enemies and hp_frac > 0.6 and t - self.threat_t > 6.0):
            self.retreating = False

        mv = (0.0, 0.0)
        boost = False
        mode = "idle"
        hold_fire = False
        late = obs["alive"] <= 3 or zone["next_radius"] <= 30
        engaged = t - self.threat_t < 4.0 or (tgt is not None and tgt["d"] < 65)
        survive = ffa and not late     # free-for-all: placement only, so stay out of other people's fights
        cubes = [cp for cp in self.cube_memo.values()
                 if abs(cp[2] - self.arena.ground_at(cp[0], cp[1])) < 2.5 and math.hypot(cp[0] - zc[0], cp[1] - zc[1]) < zone["radius"] + 40]
        cube = min(cubes, key=lambda c: math.hypot(c[0] - x, c[1] - y), default=None)
        cube_d = math.hypot(cube[0] - x, cube[1] - y) if cube else 1e9
        armed = max(ranks) > 0
        if must_move:
            mv = self.zone_dir(obs, me)
            boost = not me["in_zone"] and me["energy"] > 5
            mode = "zone"
        elif self.retreating and (enemies or t - self.threat_t < 6.0):
            if enemies:
                mv = self.retreat_dir(obs, me, enemies, t)
            elif self.threat_bearing is not None:
                b = self.threat_bearing
                mv = self.retreat_dir(obs, me, [{"pos": [x + 90 * math.cos(b), y + 90 * math.sin(b), z], "los": True, "d": 90.0}], t)
            boost = me["energy"] > 25 and any(e["los"] and e["d"] < 120 for e in enemies)
            mode = "retreat"
        elif survive and tgt is not None and not engaged:
            mv = self.retreat_dir(obs, me, visible or enemies, t)   # seen someone who is not on us yet: slip out of sight
            hold_fire = True
            mode = "slip"
        elif survive and tgt is None and t - self.threat_t >= 4.0:
            hx, hy = self.hide_spot(obs, me, t)
            if math.hypot(hx - x, hy - y) > 5.0:
                mv = self.go(hx, hy, me, t)
                quiet = st["walk_speed"] > 0 and 4.6 / st["walk_speed"]
                mv = (mv[0] * quiet, mv[1] * quiet)       # under 5 m/s makes no footstep sound
            mode = "hide"
        elif tgt is not None:
            if not tgt["los"]:
                if PURSUE and not survive:
                    mv = self.go(tgt["pos"][0], tgt["pos"][1], me, t)     # it ducked behind something: go get a line on it
                    boost = any(k == "lightning" for k in held) and me["energy"] > 70 and tgt["d"] < 90
                mode = "pursue"
            else:
                mv, boost = self.duel_move(obs, me, tgt, t)
                mode = "fight"
            if not survive and tgt["d"] > 40:
                # the economy keeps running during a fight: step onto nearby weapons and cubes while shooting
                if want and want[4] < (GRAB if not armed else 35.0):
                    mv = self.go(want[3][0], want[3][1], me, t)
                    mode = "grab"
                elif cube and cube_d < (70.0 if hp_frac > 0.5 else 35.0):
                    mv = self.go(cube[0], cube[1], me, t)
                    mode = "snack"
        elif t - self.threat_t < 3.0 and self.threat_bearing is not None:
            # being shot by something we cannot see: do not stand there. Break its line of fire.
            b = self.threat_bearing
            ghost = {"pos": [x + 90 * math.cos(b), y + 90 * math.sin(b), z], "los": True, "d": 90.0}
            mv = self.retreat_dir(obs, me, [ghost], t)
            mode = "evade"
        else:
            # nobody in sight: weapons, cubes, drops, then drones and crates, drifting toward the next circle
            drop = next((d for d in obs["drops"] if d["lands_in"] < 12 and d.get("holder") in (None, me["id"])
                         and math.hypot(d["pos"][0] - zc[0], d["pos"][1] - zc[1]) < zone["next_radius"]), None)
            if want and want[4] < (230.0 if not armed else 90.0) and hp_frac > 0.4:
                mv = self.go(want[3][0], want[3][1], me, t)
                mode = "loot"
                # sprint for the first real weapon: everything else in the match goes better with one
                boost = RUSH and not armed and me["energy"] > 25 and want[4] > 15
            elif cube and cube_d < 150:
                mv = self.go(cube[0], cube[1], me, t)
                mode = "cube"
            elif drop and hp_frac > 0.4 and not ffa and math.hypot(drop["pos"][0] - x, drop["pos"][1] - y) < 220:
                if math.hypot(drop["pos"][0] - x, drop["pos"][1] - y) > 3.0:
                    mv = self.go(drop["pos"][0], drop["pos"][1], me, t)
                mode = "drop"
            elif outside_next or final:
                mv = self.zone_dir(obs, me)
                mode = "zone"
            elif FARM >= 2 and hp_frac >= 0.5 and not ffa and self.farm_memo:
                # walk toward the nearest remembered crate or drone that is on the ground inside the next circle
                bestf = None
                for fid, (fpos, fh) in self.farm_memo.items():
                    if abs(fpos[2] - self.arena.ground_at(fpos[0], fpos[1])) > 3.0:
                        continue
                    if math.hypot(fpos[0] - zc[0], fpos[1] - zc[1]) > zone["next_radius"] - 10:
                        continue
                    d = math.hypot(fpos[0] - x, fpos[1] - y)
                    if d < (170 if fh > 3 else 150) and (bestf is None or d < bestf[0]):
                        bestf = (d, fpos)
                if bestf is not None and (bestf[0] > 70 or not self.arena.los((x, y, z + 7.0), (bestf[1][0], bestf[1][1], bestf[1][2] + 1.5))):
                    mv = self.go(bestf[1][0], bestf[1][1], me, t)
                    mode = "farm"
        if mode == "idle" and not ffa and ROAM and dz_next > zone["next_radius"] * 0.45 and hp_frac > 0.6:
            mv = self.zone_dir(obs, me)
            mode = "roam"
        if final and not must_move and mode in ("idle", "zone", "roam"):
            mv = self.go(zc[0], zc[1], me, t) if dz_next > 2.0 else (0.0, 0.0)

        # ---- stuck recovery
        if self.last_pos is not None and math.hypot(mv[0], mv[1]) > 0.5:
            if math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 0.15:
                self.stuck_t += 0.1
            else:
                self.stuck_t = 0.0
            if self.stuck_t > 0.6:
                a = self.rng.uniform(0, TAU)
                mv = (math.cos(a), math.sin(a))
                act["jump"] = True
                if self.stuck_t > 1.5:
                    self.stuck_t = 0.0
        self.last_pos = (x, y)
        self.mode = mode
        act["move"] = [mv[0], mv[1]]
        act["boost"] = bool(boost)
        if math.hypot(mv[0], mv[1]) > 0.1:
            self.last_move = mv

        # ---- aim and fire
        fire = [False, False]
        aim = None
        if tgt is not None:
            aim, fire = self.shoot(me, tgt, slots, t)
            if hold_fire:
                fire = [False, False]
        else:
            farm = None
            under_fire = t - self.threat_t < 3.0 and self.threat_bearing is not None
            if not under_fire and not survive and not enemies:
                cands = []
                if FARM and hp_frac >= FARM_HP:   # dealing damage resets the 5 s regen timer: heal first, farm after
                    cands = [(math.dist(n["pos"], me["pos"]), n, 3.5) for n in npcs]
                    cands += [(math.dist(c["pos"], me["pos"]), c, 2.4) for c in obs["crates"]]
                cands = [c for c in cands if c[0] < 110]
                if cands:
                    farm = min(cands, key=lambda c: c[0])
            if farm is not None:
                d, obj, h = farm
                tg = {"pos": obj["pos"], "vel": obj.get("vel", (0, 0, 0)), "cubes": 0, "h": h, "r": 1.4, "farm": True}
                aim, fire = self.shoot(me, tg, slots, t)
            elif under_fire:
                aim = [self.threat_bearing, 0.0]     # face whoever is shooting us (rear hits are x1.5)
            elif enemies:
                e = enemies[0]
                aim = [math.atan2(e["pos"][1] - y, e["pos"][0] - x), 0.0]
            else:
                # sweep the torso so the 120 degree wedge covers everything every few seconds
                base = math.atan2(self.last_move[1], self.last_move[0])
                aim = [base + 1.9 * math.sin(t * 1.1), 0.0]
            gun = next((w for w in slots if w and w["kind"] == "gun"), None)
            if gun and gun["ammo"] < gun["stats"]["mag"] and gun["reload"] <= 0 and not any(fire):
                act["reload"] = True
        if tgt is None:
            for i, w in enumerate(slots):
                if w and w["kind"] == "laser" and 0 < w.get("charge", 0) < w["stats"]["charge"] + 1.5 and not any(fire):
                    fire[i] = True       # hold the charge: letting go would fire it at nothing
        if aim is not None:
            act["aim"] = [float(aim[0]), float(aim[1])]
        act["fire"] = [bool(fire[0]), bool(fire[1])]
        self.prev_vel = {m["id"]: (t, m["vel"]) for m in obs["mechs"]}
        return act

    # ------------------------------------------------------------------ behaviours
    def dodge_sign(self, obs, me, ux, uy):
        """Which way to sidestep (+1 / -1 along the left-hand perpendicular of the target bearing) so that
        incoming shots pass as far from us as possible; None if nothing is about to arrive."""
        x, y, _ = me["pos"]
        px_, py_ = -uy, ux
        v_lat = me["vel"][0] * px_ + me["vel"][1] * py_
        walk = me["stats"]["walk_speed"]
        acc = me["stats"].get("ground_accel", 58.5)
        threats = []
        for p in obs["projectiles"]:
            vx, vy = p["vel"][0], p["vel"][1]
            sp2 = vx * vx + vy * vy
            if sp2 < 1.0:
                continue
            rx, ry = x - p["pos"][0], y - p["pos"][1]
            tca = (rx * vx + ry * vy) / sp2
            if tca < 0.04 or tca > (1.6 if p["kind"] == "mortar" else 0.75):
                continue
            cx, cy = p["pos"][0] + vx * tca - x, p["pos"][1] + vy * tca - y
            reach = 10.0 if p["kind"] == "mortar" else 7.0
            if cx * cx + cy * cy > reach * reach:
                continue
            threats.append((tca, cx * px_ + cy * py_, cx * ux + cy * uy))
        if not threats:
            return None
        best = None
        for sgn in (1.0, -1.0):
            worst = 1e9
            for tca, clat, clon in threats:
                dv = sgn * walk - v_lat
                ta = min(tca, abs(dv) / acc)
                lat = v_lat * ta + 0.5 * math.copysign(acc, dv) * ta * ta + sgn * walk * (tca - ta)
                worst = min(worst, math.hypot(lat - clat, clon))
            if best is None or worst > best[0]:
                best = (worst, sgn)
        return best[1]

    def duel_move(self, obs, me, tgt, t):
        """Strafe at the range our weapons like. Returns (move, boost)."""
        x, y, _ = me["pos"]
        dx, dy = tgt["pos"][0] - x, tgt["pos"][1] - y
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        if t >= self.strafe_until:
            if self.rng.random() < 0.65:
                self.strafe = -self.strafe
            self.strafe_until = t + self.rng.uniform(0.3, 0.9)
        rng_want = FIGHT_RANGE
        held = [w["kind"] for w in me["weapons"] if w]
        boost = False
        if "laser" in held:
            rng_want = LASER_RANGE
        if "lightning" in held:
            # brawl inside arc range while there is energy for it; fall back to the other weapon's range when dry
            if me["energy"] > 35 or (self.zapping and me["energy"] > 8):
                rng_want, self.zapping = ZAP_RANGE, True
                boost = me["energy"] > 70 and 45 < d < 90
            else:
                self.zapping = False
        radial = 0.0
        side = 1.0
        if d > rng_want + 15:
            radial = 0.8
            if rng_want == ZAP_RANGE:
                radial, side = 1.0, 0.45     # close the gap quickly: arcs only reach 45 m
        elif d < rng_want - 15:
            radial = -0.7
        if DODGE:
            sgn = self.dodge_sign(obs, me, ux, uy)
            if sgn is not None:
                self.strafe = sgn
                self.strafe_until = max(self.strafe_until, t + 0.2)
            for e in obs["mechs"]:
                if e.get("charging") and e["kind"] == "mech":
                    back = math.atan2(y - e["pos"][1], x - e["pos"][0])
                    if abs(wrap(back - e["aim"][0])) < 0.25 and me["energy"] > 30 and "lightning" not in held:
                        boost = True          # a charged laser is lined up on us: make it track a faster target
        for sgn in (self.strafe, -self.strafe):
            mv = (ux * radial - uy * sgn * side, uy * radial + ux * sgn * side)
            n = math.hypot(mv[0], mv[1])
            mv = (mv[0] / n, mv[1] / n)
            if self.open_dir(me, mv):
                if sgn != self.strafe:
                    self.strafe = sgn
                return mv, boost
        return (ux * radial, uy * radial), boost

    def angle_dir(self, me, tgt, visible, t):
        """Several enemies can see us: find a nearby spot where only the target has a line on us."""
        x, y, z = me["pos"]
        others = [(e["pos"][0], e["pos"][1], e["pos"][2] + 8.0) for e in visible if e is not tgt][:3]
        teye = (tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + 8.0)
        if t - self.angle_t < 0.4 and self.angle_pt is not None:
            best = self.angle_pt
        else:
            best = None
            for r in (0.0, 10.0, 20.0, 34.0):
                for k in range(12 if r else 1):
                    a = k * TAU / 12
                    px, py = x + r * math.cos(a), y + r * math.sin(a)
                    g = self.arena.ground_at(px, py)
                    if self.arena.blocked(px, py, g):
                        continue
                    c = (px, py, g + 5.0)
                    seen = sum(1 for o in others if self.arena.los(o, c))
                    tsee = self.arena.los(teye, c)
                    cost = seen * 100 + (0 if tsee else 45) + r * 0.6
                    if best is None or cost < best[0]:
                        best = (cost, px, py, seen)
            self.angle_pt, self.angle_t = best, t
        if best is None or best[3] >= len(others):
            return None      # no cover nearby: fight in the open as usual
        d = math.hypot(best[1] - x, best[2] - y)
        if d < 2.5:
            return (0.0, 0.0)
        return ((best[1] - x) / d, (best[2] - y) / d) if d < 12 else self.go(best[1], best[2], me, t)

    def retreat_dir(self, obs, me, enemies, t):
        """Head for the nearest spot no known enemy can see, inside the next circle if possible."""
        x, y, z = me["pos"]
        zc, zr = obs["zone"]["next_center"], obs["zone"]["next_radius"]
        eyes = [(e["pos"][0], e["pos"][1], e["pos"][2] + 8.0) for e in enemies[:4]]
        ax = sum(e[0] for e in eyes) / len(eyes)
        ay = sum(e[1] for e in eyes) / len(eyes)
        away = math.atan2(y - ay, x - ax)
        best = None
        for r in (18.0, 36.0, 60.0):
            for k in range(-5, 6):
                a = away + k * 0.5
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if self.arena.blocked(px, py, self.arena.ground_at(px, py)):
                    continue
                pz = self.arena.ground_at(px, py) + 4.5
                seen = sum(1 for e in eyes if self.arena.los(e, (px, py, pz)))
                out = max(0.0, math.hypot(px - zc[0], py - zc[1]) - (zr - 10))
                cost = seen * 100 + out * 3 + r * 0.5 + abs(k) * 4
                if best is None or cost < best[0]:
                    best = (cost, px, py)
        if best is None:
            return (math.cos(away), math.sin(away))
        return self.go(best[1], best[2], me, t)

    def shoot(self, me, tgt, slots, t):
        x, y, z = me["pos"]
        st = me["stats"]
        fire = [False, False]
        gun_i = next((i for i, w in enumerate(slots) if w and w["kind"] == "gun"), None)
        las_i = next((i for i, w in enumerate(slots) if w and w["kind"] == "laser"), None)
        lit_i = next((i for i, w in enumerate(slots) if w and w["kind"] == "lightning"), None)
        mor_i = next((i for i, w in enumerate(slots) if w and w["kind"] == "mortar"), None)
        muzzle = (x, y, z + st.get("muzzle_height", 7.0))
        gspeed = slots[gun_i]["stats"]["speed"] if gun_i is not None else None
        yaw, pitch, dist, pt = self.aim_at(me, tgt, gspeed)
        aim = [yaw, pitch]
        cur_yaw, cur_pitch = me["aim"]
        r = tgt.get("r", 2.5 * (1 + 0.05 * tgt.get("cubes", 0)))
        tol = max(0.012, math.atan2(r * 1.0, dist))
        on = abs(wrap(yaw - cur_yaw)) < tol and abs(pitch - cur_pitch) < max(0.02, math.atan2(4.0, dist))
        clear = self.arena.los(muzzle, pt)
        if gun_i is not None:
            g = slots[gun_i]
            if on and clear and dist < g["stats"]["range"] * 0.97:
                fire[gun_i] = True
        if las_i is not None:
            w = slots[las_i]
            ls = w["stats"]
            dyaw, dpitch, ddist, dpt = self.aim_at(me, tgt, None)
            charge = w.get("charge", 0)
            if (w["cooldown"] <= 0.15 or charge > 0) and ddist < ls["range"] * 0.95:
                # the beam is instant: point straight at the target for the whole charge, let go when full and lined up
                aim = [dyaw, dpitch]
                full = charge >= ls["charge"] - 0.05
                up_tol = 1.8 if tgt.get("frac", 0.5) > 0.6 else 3.5     # only shoulders showing: do not sail over its head
                lined = abs(wrap(dyaw - cur_yaw)) < math.atan2(r * 0.75, ddist) and abs(dpitch - cur_pitch) < math.atan2(up_tol, ddist)
                fire[las_i] = not (full and lined and self.arena.los(muzzle, dpt))
        if lit_i is not None:
            w = slots[lit_i]
            floor_e = 55 if tgt.get("farm") else w["stats"]["energy"] + 8
            if dist < w["stats"]["range"] and abs(wrap(yaw - cur_yaw)) < math.radians(w["stats"]["cone_deg"]) * 0.9 and me["energy"] >= floor_e and clear:
                fire[lit_i] = True
        if mor_i is not None and 20 < dist < 150 and not tgt.get("farm"):
            w = slots[mor_i]
            ms = w["stats"]
            v, g = ms["speed"], ms["gravity"]
            tx, ty = tgt["pos"][0], tgt["pos"][1]
            tv = tgt.get("vel", (0, 0, 0))
            tof = dist / (v * 0.92)
            px, py = tx + tv[0] * tof, ty + tv[1] * tof
            R = math.hypot(px - x, py - y)
            dzz = tgt["pos"][2] - (z + st.get("height", 9.0) + 0.5)
            disc = v ** 4 - g * (g * R * R + 2 * dzz * v * v)
            if disc > 0 and w["cooldown"] <= 0.1:
                mp = math.atan((v * v - math.sqrt(disc)) / (g * R))
                myaw = math.atan2(py - y, px - x)
                gun_busy = gun_i is None or slots[gun_i]["reload"] > 0.3
                if gun_busy and las_i is None:
                    aim = [myaw, mp]
                if abs(wrap(myaw - cur_yaw)) < 0.05 and abs(mp - cur_pitch) < 0.04:
                    fire[mor_i] = True
        return aim, fire
