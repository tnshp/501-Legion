

import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
DECOY_THRESHOLD = 8


def fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist / fleet_speed(ships) if ships > 0 else 999.0


def line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


def predict_orbit(x: float, y: float, omega: float, dt: float):
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):
    if not orbiting:
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    theta = math.atan2(ty - SUN_Y, tx - SUN_X)
    r = math.hypot(tx - SUN_X, ty - SUN_Y)
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t


def safe_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    return cw if adiff(cw) < adiff(ccw) else ccw


def is_decoy_fleet(fleet, planets, omega):
    if fleet['ships'] < DECOY_THRESHOLD:
        return True
    tgt_id = None
    best_dist = float('inf')
    for p in planets.values():
        d = math.hypot(fleet['x'] - p['x'], fleet['y'] - p['y'])
        if d < best_dist:
            best_dist = d
            tgt_id = p['id']
    if tgt_id is None:
        return True
    tgt = planets.get(tgt_id)
    if tgt is None:
        return True
    r = math.hypot(tgt['x'] - SUN_X, tgt['y'] - SUN_Y)
    is_orb = (r + tgt['radius']) < 48.0
    ships_needed = tgt['ships'] + 1
    if fleet['ships'] < ships_needed * 0.4:
        return True
    return False


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=1.05):
    if owner == -1:
        return int(tgt_ships * margin) + 1
    growth = tgt_prod * tt
    return int((tgt_ships + growth) * margin) + 1


def planet_under_threat(p_id, fleets, planets, player, omega):
    incoming = 0
    for f in fleets.values():
        if f['owner'] == player:
            continue
        best_tgt, best_d = None, float('inf')
        for p in planets.values():
            if p['id'] == f['from']:
                continue
            d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
            if d < best_d:
                best_d = d
                best_tgt = p['id']
        if best_tgt == p_id:
            r = math.hypot(planets[p_id]['x'] - SUN_X, planets[p_id]['y'] - SUN_Y)
            is_orbiting = (r + planets[p_id]['radius']) < 48.0
            if is_orbiting:
                ix, iy = predict_orbit(planets[p_id]['x'], planets[p_id]['y'], omega, travel_time(f['x'], f['y'], planets[p_id]['x'], planets[p_id]['y'], int(f['ships'])))
                d = math.hypot(ix - planets[p_id]['x'], iy - planets[p_id]['y'])
            else:
                d = math.hypot(f['x'] - planets[p_id]['x'], f['y'] - planets[p_id]['y'])
            if d < 50:
                incoming += f['ships']
    return incoming


# =============================================================================
# MULTI-LEG PATH PLANNER (minimal - just for hard targets)
# =============================================================================

def compute_tangent_points(x1: float, y1: float, margin: float = 2.0):
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + margin:
        return None, None
    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    return to_sun + half_angle, to_sun - half_angle