"""Mech Royale bot: decision logic. `Bot(hello)` then `act(obs) -> action` ten times a second."""
import math
import os
import random

import numpy as np

from geom import Arena
from nav import Nav

TAU = 2 * math.pi
WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
UPGRADE_ORDER = [
    "plating", "armor", "nanites", "laser_damage", "gun_damage", "servos", "gun_mag", "laser_charge",
    "gun_velocity", "lightning_efficiency", "lightning_range", "lightning_chain", "laser_pierce",
    "jets", "capacitor", "reactor", "gyros", "sensors", "gun_rate", "mortar_reload", "mortar_splash", "mortar_velocity",
]

# tunables (the MR_* environment overrides exist only for local experiments)
def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


FIGHT_RANGE = _env("MR_RANGE", 85.0)      # preferred duel distance (m)
RETREAT_HP = _env("MR_RETREAT", 0.38)     # break contact below this health fraction
REJOIN_HP = _env("MR_REJOIN", 0.85)       # ... and come back above this one
ZONE_MARGIN = 18.0
FARM = _env("MR_FARM", 2.0)
if os.environ.get("MR_WTOP") == "lightning":
    WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 1, "gun": 0}
LASER_RANGE = _env("MR_LRANGE", 115.0)
ANGLE = _env("MR_ANGLE", 1.0) > 0
ROAM = _env("MR_ROAM", 1.0) > 0
GRAB = _env("MR_GRAB", 70.0)
ZAP_RANGE = _env("MR_ZRANGE", 36.0)


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
        """(yaw, pitch, distance, lead point) to hit tgt; speed None = hitscan."""
        x, y, z = me["pos"]
        mz = z + me["stats"].get("muzzle_height", 7.0)
        h = height if height is not None else 9.0 * (1 + 0.05 * tgt.get("cubes", 0))
        tx, ty, tz = tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + h * tgt.get("frac", 0.5)
        vx, vy, vz = tgt.get("vel", (0, 0, 0))
        if speed:
            # the target cannot fall through the floor: ignore vertical speed unless it is airborne and rising
            vz_use = vz if abs(vz) > 0.5 else 0.0
            tt = intercept(tx - x, ty - y, tz - mz, vx - 0.0, vy - 0.0, vz_use, speed)
            if tt is not None:
                tt = min(tt, 1.2)
                tx, ty, tz = tx + vx * tt, ty + vy * tt, tz + vz_use * tt
        hd = math.hypot(tx - x, ty - y)
        return math.atan2(ty - y, tx - x), math.atan2(tz - mz, max(hd, 0.1)), math.hypot(hd, tz - mz), (tx, ty, tz)

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

        # ---- bookkeeping
        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("rear") and ev.get("bearing") is not None:
                self.rear_t, self.rear_bearing, self.rear_by = t, ev["bearing"], ev.get("by")
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                self.threat_bearing, self.threat_t = ev["bearing"], t
            elif ev["type"] == "kill":
                self.my_kills += 1
        if self.last_alive is not None and obs["alive"] < self.last_alive:
            self.deaths_seen += self.last_alive - obs["alive"]
        self.last_alive = obs["alive"]
        ffa = True  # experiment: always play the hide-and-survive posture
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
        second_rank = WEAPON_RANK.get(held[1], -1) if len(held) > 1 else -1
        want = None
        for pid, (kind, pos) in self.pick_memo.items():
            if kind in held or WEAPON_RANK.get(kind, 0) <= second_rank:
                continue
            d = math.hypot(pos[0] - x, pos[1] - y)
            score = d - 25 * WEAPON_RANK[kind]
            if abs(pos[2] - self.arena.ground_at(pos[0], pos[1])) > 2.0:
                continue  # on a roof: skip for now
            if want is None or score < want[0]:
                want = (score, pid, kind, pos, d)
        if want and want[4] < 3.6 and abs(want[3][2] - z) < 4:
            act["take"] = want[1]
            if held[1] is not None:
                act["drop"] = 1

        # ---- zone pressure
        zc = zone["next_center"]
        dz_next = math.hypot(x - zc[0], y - zc[1])
        final = zone["next_radius"] <= 30
        margin = min(ZONE_MARGIN, zone["next_radius"] * 0.5)
        outside_next = dz_next > zone["next_radius"] - margin
        time_left = zone["shrink_end"] - t
        need = (dz_next - (zone["next_radius"] - margin)) / (st["walk_speed"] * 0.8)
        must_move = (outside_next and need > time_left - 12) or not me["in_zone"]

        # ---- choose a target and stance
        tgt = visible[0] if visible else None
        if tgt is not None and len(visible) > 1:
            # prefer the one we can finish, then the nearest
            tgt = min(visible, key=lambda e: e["d"] + 0.08 * e.get("hp", 1400))
        n_threat = len([e for e in visible if e["d"] < 160])
        if hp_frac < RETREAT_HP or (n_threat >= 3 and hp_frac < 0.7):
            self.retreating = True
        elif hp_frac > REJOIN_HP or (not enemies and hp_frac > 0.6):
            self.retreating = False
        if tgt is not None and tgt.get("hp", 9999) < me["hp"] * 0.5 and n_threat <= 1:
            self.retreating = False  # we win this trade: finish it

        mv = (0.0, 0.0)
        boost = False
        mode = "idle"
        hold_fire = False
        late = obs["alive"] <= 3 or zone["next_radius"] <= 30
        engaged = t - self.threat_t < 4.0 or (tgt is not None and tgt["d"] < 65)
        survive = ffa and not late     # free-for-all: placement only, so stay out of other people's fights
        if must_move:
            mv = self.zone_dir(obs, me)
            boost = not me["in_zone"] and me["energy"] > 5
            mode = "zone"
        elif self.retreating and enemies:
            mv = self.retreat_dir(obs, me, enemies, t)
            boost = me["energy"] > 25 and any(e["los"] and e["d"] < 120 for e in enemies)
            mode = "retreat"
        elif survive and tgt is not None and not engaged:
            mv = self.retreat_dir(obs, me, visible, t)   # seen someone who is not on us yet: slip out of sight
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
            if t - self.rear_t < 1.5 and self.rear_by != tgt["id"]:
                b = self.rear_bearing
                ghost = {"pos": [x + 80 * math.cos(b), y + 80 * math.sin(b), z], "los": True, "d": 80.0}
                mv = self.retreat_dir(obs, me, visible + [ghost], t)
                mode = "unflank"
            elif ANGLE and len(visible) >= 2:
                mv = self.angle_dir(me, tgt, visible, t)
                mode = "angle"
                if mv is None:
                    mv = self.duel_move(me, tgt, t, ffa)
                    mode = "fight"
            else:
                mv = self.duel_move(me, tgt, t, ffa)
                mode = "fight"
            if GRAB and held[1] is None and want and want[4] < GRAB and tgt["d"] > 45 and mode in ("fight", "angle"):
                mv = self.go(want[3][0], want[3][1], me, t)   # keep shooting, but walk onto the weapon
                mode = "grab"
        elif t - self.threat_t < 3.0 and self.threat_bearing is not None and not enemies:
            # being shot by something we cannot see: do not stand there. Break its line of fire.
            b = self.threat_bearing
            ghost = {"pos": [x + 90 * math.cos(b), y + 90 * math.sin(b), z], "los": True, "d": 90.0}
            mv = self.retreat_dir(obs, me, [ghost], t)
            mode = "evade"
        elif enemies and not ffa:
            # known but hidden: hold an angle on where it will appear rather than walking into it
            e = enemies[0]
            if t - self.threat_t < 3.0:
                if hp_frac > 0.55:
                    mv = self.go(e["pos"][0], e["pos"][1], me, t)   # it can hit us and we cannot hit it: close in for a line
                else:
                    mv = self.retreat_dir(obs, me, enemies, t)
            elif hp_frac > 0.7 and e["d"] > FIGHT_RANGE + 40:
                mv = self.go(e["pos"][0], e["pos"][1], me, t)
            mode = "stalk"
        else:
            # nobody around: heal first, then loot, then drift to the next circle
            cubes = [{"pos": cp} for cp in self.cube_memo.values()
                     if abs(cp[2] - self.arena.ground_at(cp[0], cp[1])) < 2.5 and math.hypot(cp[0] - zc[0], cp[1] - zc[1]) < zone["radius"] + 40]
            cube = min(cubes, key=lambda c: math.dist(c["pos"][:2], (x, y)), default=None)
            drop = next((d for d in obs["drops"] if d["lands_in"] < 12 and math.hypot(d["pos"][0] - zc[0], d["pos"][1] - zc[1]) < zone["next_radius"]), None)
            if cube and math.dist(cube["pos"][:2], (x, y)) < 170:
                mv = self.go(cube["pos"][0], cube["pos"][1], me, t)
                mode = "cube"
            elif want and want[4] < 230 and hp_frac > 0.4:
                mv = self.go(want[3][0], want[3][1], me, t)
                mode = "loot"
            elif drop and hp_frac > 0.5 and not ffa and math.hypot(drop["pos"][0] - x, drop["pos"][1] - y) < 220:
                mv = self.go(drop["pos"][0], drop["pos"][1], me, t)
                mode = "drop"
            elif outside_next or final:
                mv = self.zone_dir(obs, me)
                mode = "zone"
            elif FARM >= 2 and hp_frac >= 0.9 and not ffa and self.farm_memo:
                # walk toward the nearest remembered crate or drone that is on the ground inside the next circle
                best = None
                for fid, (fpos, _h) in self.farm_memo.items():
                    if abs(fpos[2] - self.arena.ground_at(fpos[0], fpos[1])) > 3.0:
                        continue
                    if math.hypot(fpos[0] - zc[0], fpos[1] - zc[1]) > zone["next_radius"] - 10:
                        continue
                    d = math.hypot(fpos[0] - x, fpos[1] - y)
                    if d < 260 and (best is None or d < best[0]):
                        best = (d, fpos)
                if best is not None and (best[0] > 70 or not self.arena.los((x, y, z + 7.0), (best[1][0], best[1][1], best[1][2] + 1.5))):
                    mv = self.go(best[1][0], best[1][1], me, t)
                    mode = "farm"
        if mode == "idle" and not ffa and ROAM and dz_next > zone["next_radius"] * 0.45 and hp_frac > 0.6:
            mv = self.zone_dir(obs, me)
            mode = "roam"
        if final and not must_move and mode in ("idle", "stalk", "zone", "roam"):
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
            if not under_fire and not survive and (not enemies or mode in ("idle", "cube", "loot", "zone", "farm", "roam")):
                cands = []
                if FARM and hp_frac >= 0.97:   # dealing damage resets the 5 s regen timer: heal first, farm after
                    cands = [(math.dist(n["pos"], me["pos"]), n, 3.5) for n in npcs]
                    cands += [(math.dist(c["pos"], me["pos"]), c, 2.4) for c in obs["crates"]]
                cands = [c for c in cands if c[0] < 110]
                if cands:
                    farm = min(cands, key=lambda c: c[0])
            if farm is not None:
                d, obj, h = farm
                gun = next((w for w in slots if w and w["kind"] == "gun"), None)
                tg = {"pos": obj["pos"], "vel": obj.get("vel", (0, 0, 0)), "cubes": 0}
                yaw, pitch, dist, pt = self.aim_at(me, tg, gun["stats"]["speed"] if gun else None, h)
                aim = [yaw, pitch]
                on = abs(wrap(yaw - me["aim"][0])) < max(0.012, math.atan2(0.9, dist)) and abs(pitch - me["aim"][1]) < max(0.012, math.atan2(0.9, dist))
                muzzle = (x, y, z + st.get("muzzle_height", 7.0))
                if on and self.arena.los(muzzle, pt):
                    for i, w in enumerate(slots):
                        if w and (w["kind"] == "gun" or (w["kind"] == "lightning" and dist < w["stats"]["range"] and me["energy"] > 40)):
                            fire[i] = True
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
        if aim is not None:
            act["aim"] = [float(aim[0]), float(aim[1])]
        act["fire"] = [bool(fire[0]), bool(fire[1])]
        return act

    # ------------------------------------------------------------------ behaviours
    def duel_move(self, me, tgt, t, ffa):
        x, y, _ = me["pos"]
        dx, dy = tgt["pos"][0] - x, tgt["pos"][1] - y
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        if t >= self.strafe_until:
            if self.rng.random() < 0.65:
                self.strafe = -self.strafe
            self.strafe_until = t + self.rng.uniform(0.25, 0.75)
        rng_want = FIGHT_RANGE
        held = [w["kind"] for w in me["weapons"] if w]
        if "lightning" in held and me["energy"] > 30 and "laser" not in tgt.get("weapons", []):
            rng_want = ZAP_RANGE
        if "laser" in held:
            rng_want = LASER_RANGE
        radial = 0.0
        if d > rng_want + 12:
            radial = 0.8
        elif d < rng_want - 20:
            radial = -0.8
        for s in (self.strafe, -self.strafe):
            mv = (ux * radial - uy * s, uy * radial + ux * s)
            n = math.hypot(mv[0], mv[1])
            mv = (mv[0] / n, mv[1] / n)
            if self.open_dir(me, mv):
                if s != self.strafe:
                    self.strafe = s
                return mv
        return (ux * radial, uy * radial)

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
        r = 2.5 * (1 + 0.05 * tgt.get("cubes", 0))
        tol = max(0.01, math.atan2(r * 0.8, dist))
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
            if w["cooldown"] <= 0.2 and ddist < ls["range"] * 0.95:
                full = w.get("charge", 0) >= ls["charge"] - 0.05
                if not full:
                    fire[las_i] = True
                else:
                    # fully charged: swing onto the target itself (hitscan needs no lead) and let go when lined up
                    aim = [dyaw, dpitch]
                    lined = abs(wrap(dyaw - cur_yaw)) < math.atan2(r * 0.7, ddist) and abs(dpitch - cur_pitch) < math.atan2(3.5, ddist)
                    fire[las_i] = not (lined and self.arena.los(muzzle, dpt))
        if lit_i is not None:
            w = slots[lit_i]
            if dist < w["stats"]["range"] and abs(wrap(yaw - cur_yaw)) < math.radians(w["stats"]["cone_deg"]) * 0.9 and me["energy"] >= w["stats"]["energy"] + 8 and clear:
                fire[lit_i] = True
        if mor_i is not None and 20 < dist < 150:
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
