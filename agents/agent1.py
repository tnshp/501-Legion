import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow
import numpy as np

MAX_SPEED = 6.0
MIN_SHIPS_MINE_ATTACK = 5
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP = 8

FORMULA_DIST = 100
FORMULA_PROD_MULT = 15
FORMULA_ENEMY_BONUS_MULT = 10
FORMULA_TOTAL_SHIPS_PERCENT = 0.7


# ── pure helpers (no mutable state) ──────────────────────────────────────────

def get_custom_score(m, t):
    dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)
    min_ships = t.ships + 1
    fleet_speed = get_fleet_speed(max(1, min_ships))
    eta = dist / fleet_speed
    enemy_produced = 0
    enemy_bonus = 0
    if t.owner != -1:
        enemy_produced = eta * t.production
        enemy_bonus = t.production
    total_ships = min_ships + enemy_produced
    return (
        (FORMULA_DIST - dist)
        + (FORMULA_PROD_MULT * t.production)
        + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2 * eta)
    )


def refresh_local_obs(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    mine    = [p for p in planets if p.owner == obs.get("player", [])]
    targets = [p for p in planets if p.owner != obs.get("player", [])]
    player  = obs.get("player", -2)
    fleets  = [ow.Fleet(*f) for f in obs.get("fleets", [])]
    return {"planets": planets, "mine": mine, "targets": targets,
            "player": player, "fleets": fleets}


def get_fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def sun_collision(m, fleet_speed, angle, ticks=61):
    prev_x, prev_y = m.x, m.y
    for tick in range(1, ticks):
        x = m.x + math.cos(angle) * fleet_speed * tick
        y = m.y + math.sin(angle) * fleet_speed * tick
        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True
        prev_x, prev_y = x, y
    return False


def calculate_angle(m, t):
    return math.atan2(t.y - m.y, t.x - m.x)


def find_angle_to_planet(p, t, ships, vel, moving=False):
    fleet_speed = get_fleet_speed(ships)
    if moving:
        for tick, (tx, ty) in enumerate(get_planet_trajectories(t, vel), start=1):
            dx = tx - p.x
            dy = ty - p.y
            dist_to_target = math.sqrt(dx**2 + dy**2) - p.radius
            if abs(fleet_speed * tick - dist_to_target) > t.radius:
                continue
            angle = math.atan2(dy, dx)
            if sun_collision(p, fleet_speed, angle):
                return None, None
            return angle, tick
        return None, None
    else:
        angle = calculate_angle(p, t)
        if sun_collision(p, fleet_speed, angle):
            return None, None
        dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
        return angle, math.floor(dist / fleet_speed)


def collides(x1, y1, x2, y2, cx, cy, r):
    vec_x, vec_y = x2 - x1, y2 - y1
    vec_to_cx, vec_to_cy = cx - x1, cy - y1
    vec_length_sq = vec_x**2 + vec_y**2
    if vec_length_sq == 0:
        return (x1 - cx)**2 + (y1 - cy)**2 <= r**2
    t = max(0, min(1, (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq))
    closest_x = x1 + t * vec_x
    closest_y = y1 + t * vec_y
    return (closest_x - cx)**2 + (closest_y - cy)**2 <= r**2


def get_closest_planets_to_target(mine, t):
    planets = [(m, math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)) for m in mine]
    return sorted(planets, key=lambda k: k[1])


def get_planet_trajectories(p, vel):
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    return [
        (50 + r * math.cos(angle + vel * tick),
         50 + r * math.sin(angle + vel * tick))
        for tick in range(1, 61)
    ]


def predict_total_ships(m, t, vel, base_ships, m_ships, moving=False):
    total_ships = base_ships
    for _ in range(5):
        angle, arrive_tick = find_angle_to_planet(m, t, total_ships, vel, moving=moving)
        if angle is None:
            return None, None, None
        new_total = base_ships + (arrive_tick * t.production if t.owner != -1 else 0)
        if new_total > m_ships:
            return None, None, None
        if new_total == total_ships:
            break
        total_ships = new_total
    return total_ships, angle, arrive_tick


def plan_coop_attack(attacking_planets, t, base_ships, vel, moving=False):
    remainder = base_ships
    planned = []
    for a_p in attacking_planets:
        p = a_p["planet"]
        p_ships = min(a_p["ships"], remainder)
        if p_ships > 0:
            p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
        if p_ships <= 0:
            continue
        angle, arrive_tick = find_angle_to_planet(p, t, p_ships, vel, moving=moving)
        remainder -= p_ships
        if angle is None or arrive_tick is None:
            continue
        planned.append([p, angle, p_ships, arrive_tick])
    return remainder, planned


def get_candidate_targets(m, targets, comet_planet_ids):
    candidates = [
        (m, t, get_custom_score(m, t))
        for t in targets if t.id not in comet_planet_ids
    ]
    return sorted(candidates, key=lambda x: x[2], reverse=True)


# ── stateful agent class ──────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    Encapsulates the rule-based agent with per-instance mutable state.

    Use as a kaggle opponent::

        agent = RuleBasedAgent()
        env.train([None, agent])   # kaggle calls agent(obs, config) each step

    Call reset() between episodes when reusing the same instance::

        env.reset()
        agent.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.fleet_trajectories = []
        self.reinforcement_trajectories = []
        self.moving_planets = []
        self.steps = 0

    def __call__(self, obs, config=None):
        return self._step(obs)

    # ── stateful helpers ──────────────────────────────────────────────────────

    def _fill_moving_planets(self, obs):
        planets = [ow.Planet(*p) for p in obs.get("planets", [])]
        initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
        for p in planets:
            i = initial_by_id.get(p.id)
            if i and (p.x, p.y) != (i.x, i.y) and p.id not in self.moving_planets:
                self.moving_planets.append(p.id)

    def _update_fleet_trajectories(self, fleets):
        for f_t in self.fleet_trajectories[:]:
            found = any(
                f.from_planet_id == f_t["mine"].id and abs(f.angle - f_t["angle"]) < 1e-6
                for f in fleets
            )
            if found:
                f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)
            else:
                self.fleet_trajectories.remove(f_t)

    def _update_reinforcement_trajectories(self, planets):
        for r_t in self.reinforcement_trajectories[:]:
            r_t["arrive_tick"] -= 1
            if r_t["arrive_tick"] <= 0:
                self.reinforcement_trajectories.remove(r_t)

    def _get_planets_under_attack(self, mine, fleets, player, vel):
        mov_pl_traj = {
            m.id: get_planet_trajectories(m, vel)
            for m in mine if m.id in self.moving_planets
        }
        under_attack = {}
        seen = set()
        enemy_fleets = [f for f in fleets if f.owner != player]

        for f in enemy_fleets:
            fleet_speed = get_fleet_speed(f.ships)
            prev_x, prev_y = f.x, f.y
            for tick in range(1, 61):
                next_x = f.x + math.cos(f.angle) * fleet_speed * tick
                next_y = f.y + math.sin(f.angle) * fleet_speed * tick
                for m in mine:
                    m_x, m_y = (mov_pl_traj[m.id][tick - 1]
                                 if m.id in self.moving_planets else (m.x, m.y))
                    if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, m.radius):
                        if (m.id, f.id) not in seen:
                            under_attack.setdefault(m.id, {"planet": m, "fleets": []})
                            under_attack[m.id]["fleets"].append({"fleet": f, "arrive_tick": tick})
                            seen.add((m.id, f.id))
                prev_x, prev_y = next_x, next_y

        return under_attack

    def _get_reinforcement_plans(self, mine, under_attack):
        reinforcement_plans = {}
        for p in mine:
            if p.id not in under_attack:
                continue
            attacking_fleets = sorted(under_attack[p.id]["fleets"],
                                      key=lambda a: a["arrive_tick"])
            incoming = sorted(
                [r for r in self.reinforcement_trajectories if r["target"].id == p.id],
                key=lambda r: r["arrive_tick"]
            )
            p_ships = p.ships
            prev_tick = 0
            r_idx = 0
            for att in attacking_fleets:
                tick = att["arrive_tick"]
                p_ships += (tick - prev_tick) * p.production
                while r_idx < len(incoming) and incoming[r_idx]["arrive_tick"] <= tick:
                    p_ships += incoming[r_idx]["ships"]
                    r_idx += 1
                p_ships -= att["fleet"].ships
                prev_tick = tick
                if p_ships < 0:
                    reinforcement_plans[p] = {
                        "ships_needed": max(MIN_SHIPS_MINE_ATTACK, abs(p_ships)),
                        "needed_by_tick": tick,
                    }
                    break
        return reinforcement_plans

    # ── main decision step ────────────────────────────────────────────────────

    def _step(self, obs):
        moves = []

        if self.steps < 2:
            self.steps += 1
            return []
        if self.steps == 2:
            self._fill_moving_planets(obs)
            self.steps = 3

        lobs = refresh_local_obs(obs)
        self._update_fleet_trajectories(lobs["fleets"])
        self._update_reinforcement_trajectories(lobs["planets"])
        comet_planet_ids = obs.get("comet_planet_ids", [])
        under_attack = self._get_planets_under_attack(
            lobs["mine"], lobs["fleets"], lobs["player"], obs.angular_velocity
        )
        exhausted = set()

        if not lobs["targets"]:
            return []

        # ── reinforcement phase ───────────────────────────────────────────────
        reinforcement_plans = self._get_reinforcement_plans(lobs["mine"], under_attack)
        for p, plan in reinforcement_plans.items():
            if any(r["target"].id == p.id and r["arrive_tick"] >= 0
                   for r in self.reinforcement_trajectories):
                continue

            ships_needed   = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]

            for p_np, _ in get_closest_planets_to_target(lobs["mine"], p):
                if p_np.id == p.id or p_np.id in exhausted:
                    continue

                avail = p_np.ships - sum(
                    r["ships"] for r in self.reinforcement_trajectories
                    if r["mine"].id == p_np.id
                )
                if p_np.id in under_attack:
                    avail = max(0, avail - sum(
                        a["fleet"].ships for a in under_attack[p_np.id]["fleets"]
                    ))

                sent = max(MIN_SHIPS_MINE_ATTACK, ships_needed)
                if avail < sent:
                    continue

                angle, arrive_tick = find_angle_to_planet(
                    p_np, p, sent, obs.angular_velocity,
                    moving=p.id in self.moving_planets
                )
                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue

                moves.append([p_np.id, angle, sent])
                exhausted.add(p_np.id)
                self.reinforcement_trajectories.append({
                    "mine": p_np, "target": p,
                    "angle": angle, "ships": sent, "arrive_tick": arrive_tick,
                })
                break

        # ── attack phase ──────────────────────────────────────────────────────
        for m in sorted(lobs["mine"], key=lambda p: p.ships, reverse=True):
            if m.id in exhausted or m.ships < MIN_SHIPS_MINE_ATTACK:
                continue

            for m, t, _ in get_candidate_targets(m, lobs["targets"], comet_planet_ids)[:3]:
                m_avail = m.ships
                if m.id in under_attack:
                    m_avail = max(0, m.ships - sum(
                        a["fleet"].ships for a in under_attack[m.id]["fleets"]
                    ))
                if m_avail < MIN_SHIPS_MINE_ATTACK:
                    continue

                safe_nearest = []
                for p, dist in get_closest_planets_to_target(lobs["mine"], t):
                    if p.id == m.id or p.id in exhausted:
                        continue
                    avail = p.ships
                    if p.id in under_attack:
                        avail = max(0, avail - sum(
                            a["fleet"].ships for a in under_attack[p.id]["fleets"]
                        ))
                    if avail >= MIN_SHIPS_MINE_ATTACK:
                        safe_nearest.append((p, dist, avail))

                owned_count = len(lobs["mine"])
                total_count = len(lobs["planets"])
                en_route = sum(f["ships"] for f in self.fleet_trajectories
                               if f["target"].id == t.id)
                needed_now = t.ships + 1 + (3 * t.production if t.owner != -1 else 0)

                if owned_count < total_count * 0.75 and en_route >= needed_now:
                    continue

                base_ships = max(MIN_SHIPS_MINE_ATTACK, needed_now - en_route)
                t_moving = t.id in self.moving_planets

                if m_avail >= base_ships:
                    total, angle, arrive_tick = predict_total_ships(
                        m, t, obs.angular_velocity, base_ships, m_avail, moving=t_moving
                    )
                    if angle is not None and not sun_collision(m, get_fleet_speed(max(1, total)), angle):
                        moves.append([m.id, angle, total])
                        exhausted.add(m.id)
                        self.fleet_trajectories.append({
                            "mine": m, "target": t,
                            "angle": angle, "ships": total, "arrive_tick": arrive_tick,
                        })

                elif (m_avail < base_ships
                      and len(lobs["mine"]) > 1
                      and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK):
                    accum = m_avail
                    attacking_planets = [{"planet": m, "ships": m_avail}]
                    coop_sent = False

                    for p, _, p_avail in safe_nearest:
                        if coop_sent:
                            break
                        attacking_planets.append({"planet": p, "ships": p_avail})
                        accum += p_avail
                        if len(attacking_planets) > COOP_PLANET_CAP or accum < base_ships:
                            continue

                        remainder, planned = plan_coop_attack(
                            attacking_planets, t, base_ships, obs.angular_velocity, moving=False
                        )
                        if remainder > 0:
                            continue

                        for move in planned:
                            self.fleet_trajectories.append({
                                "mine": move[0], "target": t,
                                "angle": move[1], "ships": move[2], "arrive_tick": move[3],
                            })
                            exhausted.add(move[0].id)
                            move[0] = move[0].id
                            moves.append(move)
                        coop_sent = True
                        break

        return moves


# ── kaggle submission compatibility ──────────────────────────────────────────
# A single module-level instance keeps state across steps of one kaggle game.
# Import RuleBasedAgent directly when you need per-instance reset (training).

_kaggle_agent = RuleBasedAgent()

def agent(obs, config=None):
    return _kaggle_agent(obs)
