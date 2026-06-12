"""JAX implementations of the geometric primitives used by the game loop."""
import jax
import jax.numpy as jnp


def dist2(ax, ay, bx, by):
    """Squared distance between two points."""
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def dist(ax, ay, bx, by):
    return jnp.sqrt(dist2(ax, ay, bx, by))


def swept_pair_hit_time(
    ax, ay, bx, by,   # fleet: start → end
    px, py, qx, qy,   # planet: start → end
    r,                # collision radius (planet.radius)
):
    """Continuous-collision time for fleet A→B vs planet P→Q.

    Returns the earliest t ∈ [0, 1] at which the moving fleet comes within
    radius r of the moving planet centre, or jnp.inf if no collision.

    Both paths are linearised over the tick (planet orbit → chord).
    """
    d0x = ax - px
    d0y = ay - py
    dvx = (bx - ax) - (qx - px)
    dvy = (by - ay) - (qy - py)

    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    # Already overlapping (a ~ 0 or relative motion tiny)
    t_static = jnp.where(c <= 0.0, 0.0, jnp.inf)

    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)

    hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    t_entry = jnp.where(t1 >= 0.0, t1, 0.0)  # already inside if t1 < 0
    t_hit = jnp.where(hit, t_entry, jnp.inf)

    return jnp.where(a < 1e-12, t_static, t_hit)


def point_to_segment_dist_sq(
    px, py,     # point
    ax, ay,     # segment start
    bx, by,     # segment end
):
    """Squared distance from point P to line segment AB."""
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    t = ((px - ax) * abx + (py - ay) * aby) / jnp.maximum(l2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return dist2(px, py, cx, cy)


def fleet_speed(ships, max_speed):
    """Speed formula matching orbit_wars.py: 1 ship = 1/turn, scales to max_speed."""
    log_ships = jnp.log(jnp.maximum(ships, 1.0))
    log_max = jnp.log(1000.0)
    ratio = jnp.clip(log_ships / log_max, 0.0, 1.0)
    speed = 1.0 + (max_speed - 1.0) * ratio ** 1.5
    return jnp.minimum(speed, max_speed)
