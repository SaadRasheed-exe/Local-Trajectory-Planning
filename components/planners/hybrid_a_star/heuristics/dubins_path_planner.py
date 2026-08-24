from math import sin, cos, atan2, sqrt, acos, pi, hypot

# Toleranz für Gleitkomma-Ungenauigkeiten (Floating Point Precision)
EPSILON = 1e-8

def mod2pi(theta):
    """
    Normalizes an angle to the range [0, 2*pi).
    """
    return theta % (2 * pi)

def _calc_trig_funcs(alpha, beta):
    """
    Precalculates trigonometric values to avoid redundant computations.
    """
    return sin(alpha), sin(beta), cos(alpha), cos(beta), cos(alpha - beta)

def _LSL(alpha, beta, d):
    """Calculates path lengths for the Left-Straight-Left curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    
    if p_squared < 0:
        if p_squared > -EPSILON:
            p_squared = 0.0
        else:
            return None, None, None
            
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    return mod2pi(-alpha + tmp), sqrt(p_squared), mod2pi(beta - tmp)

def _RSR(alpha, beta, d):
    """Calculates path lengths for the Right-Straight-Right curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    
    if p_squared < 0:
        if p_squared > -EPSILON:
            p_squared = 0.0
        else:
            return None, None, None
            
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    return mod2pi(alpha - tmp), sqrt(p_squared), mod2pi(-beta + tmp)

def _LSR(alpha, beta, d):
    """Calculates path lengths for the Left-Straight-Right curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    
    if p_squared < 0:
        if p_squared > -EPSILON:
            p_squared = 0.0
        else:
            return None, None, None
            
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    return mod2pi(-alpha + tmp), d1, mod2pi(-mod2pi(beta) + tmp)

def _RSL(alpha, beta, d):
    """Calculates path lengths for the Right-Straight-Left curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    
    if p_squared < 0:
        if p_squared > -EPSILON:
            p_squared = 0.0
        else:
            return None, None, None
            
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    return mod2pi(alpha - tmp), d1, mod2pi(beta - tmp)

def _RLR(alpha, beta, d):
    """Calculates path lengths for the Right-Left-Right curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    
    if abs(tmp) > 1.0:
        if abs(tmp) - 1.0 < EPSILON:
            tmp = 1.0 if tmp > 0 else -1.0
        else:
            return None, None, None
            
    d2 = mod2pi(2 * pi - acos(tmp))
    d1 = mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    return d1, d2, mod2pi(alpha - beta - d1 + d2)

def _LRL(alpha, beta, d):
    """Calculates path lengths for the Left-Right-Left curve combination."""
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    
    if abs(tmp) > 1.0:
        if abs(tmp) - 1.0 < EPSILON:
            tmp = 1.0 if tmp > 0 else -1.0
        else:
            return None, None, None
            
    d2 = mod2pi(2 * pi - acos(tmp))
    d1 = mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    return d1, d2, mod2pi(mod2pi(beta) - alpha - d1 + mod2pi(d2))


def get_dubins_path_length(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature):
    """
    Calculates the minimum path length of a Dubins curve connecting two poses.
    This version is heavily optimized for A* heuristic calculations and 
    returns only the physical length of the shortest path, omitting waypoint generation.

    Parameters
    ----------
    s_x : float
        x position of the start point [m]
    s_y : float
        y position of the start point [m]
    s_yaw : float
        yaw angle of the start point [rad]
    g_x : float
        x position of the goal point [m]
    g_y : float
        y position of the goal point [m]
    g_yaw : float
        yaw angle of the goal point [rad]
    curvature : float
        curvature of the path, which is 1 / minimum_turning_radius [1/m]

    Returns
    -------
    path_length : float
        The physical length of the shortest Dubins path in meters.
    """

    # Prevent division by zero: curvature ~0 means infinite turning radius.
    # Fallback to straight-line Euclidean distance (admissible heuristic for A*).
    if abs(curvature) < 1e-6:
        return hypot(g_x - s_x, g_y - s_y)
    
    # 1. Calculate local coordinates of the goal relative to the start point
    # We use pure math here instead of numpy matrix multiplication for higher speed
    dx = g_x - s_x
    dy = g_y - s_y
    l_x = cos(-s_yaw) * dx - sin(-s_yaw) * dy
    l_y = sin(-s_yaw) * dx + cos(-s_yaw) * dy
    l_yaw = g_yaw - s_yaw

    # 2. Normalize distances and angles for the geometric Dubins calculation
    d = hypot(l_x, l_y) * curvature
    theta = mod2pi(atan2(l_y, l_x))
    alpha = mod2pi(-theta)
    beta = mod2pi(l_yaw - theta)

    best_cost = float("inf")
    
    # List of all 6 possible Dubins curve combinations
    planners = [_LSL, _RSR, _LSR, _RSL, _RLR, _LRL]

    # 3. Iterate through all possible curves to find the shortest one
    for planner in planners:
        d1, d2, d3 = planner(alpha, beta, d)
        
        # If the function returns None, this curve combination is mathematically invalid
        if d1 is None:
            continue

        # The cost is the sum of the absolute angles/lengths of the segments
        cost = abs(d1) + abs(d2) + abs(d3)
        if cost < best_cost:
            best_cost = cost

    # 4. Convert the normalized angular cost back to physical length [m]
    path_length = best_cost / curvature
    
    return path_length