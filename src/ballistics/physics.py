import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def derivative(state, g, use_drag, k_drag):
    _, _, vx, vy = state

    if use_drag and k_drag > 0.0:
        speed = np.hypot(vx, vy)
        ax = -k_drag * speed * vx
        ay = -g - k_drag * speed * vy
    else:
        ax = 0.0
        ay = -g

    return np.array([vx, vy, ax, ay], dtype=float)


def analytical_vacuum(v0, angle, h0=0.0, g=9.81):
    """Точное аналитическое решение для полёта в вакууме (Cd = 0)."""
    angle_rad = np.radians(angle)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)

    discriminant = vy0**2 + 2.0 * g * h0
    flight_time = (vy0 + np.sqrt(max(0.0, discriminant))) / g
    flight_range = vx0 * flight_time
    max_height = h0 + (vy0**2) / (2.0 * g)

    vy_impact = vy0 - g * flight_time
    impact_speed = float(np.hypot(vx0, vy_impact))
    impact_angle = float(np.degrees(np.arctan2(vy_impact, vx0)))

    return {
        "time": float(flight_time),
        "range": float(flight_range),
        "max_height": float(max_height),
        "impact_speed": impact_speed,
        "impact_angle": impact_angle,
    }


def _cubic_hermite_root(y0, vy0, y1, vy1, dt):
    """Находит долю шага s in (0, 1), где кубический сплайн Эрмита y(s) = 0 (точность O(dt^4))."""
    m0 = vy0 * dt
    m1 = vy1 * dt

    def y_hermite(s):
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if y_hermite(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cubic_hermite_interp(s0, ds0, s1, ds1, frac, dt):
    """Интерполирует вектор состояния на доле шага frac с 4-м порядком точности."""
    s = frac
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return h00 * s0 + h10 * (ds0 * dt) + h01 * s1 + h11 * (ds1 * dt)


def simulate(
    v0, angle, h0, mass, cd, area, rho, g, use_drag, dt=0.005, max_time=300.0
):
    """Численное интегрирование траектории методом Рунге-Кутты 4-го порядка (RK4)."""
    angle_rad = np.radians(angle)
    state = np.array(
        [0.0, h0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)], dtype=float
    )

    k_drag = 0.0
    if use_drag and mass > 0.0:
        k_drag = 0.5 * rho * cd * area / mass

    t = 0.0
    states = [state.copy()]
    times = [0.0]

    for _ in range(int(max_time / dt)):
        _, y_curr, _, vy_curr = state
        if y_curr < 0.0:
            break

        def f(s):
            return derivative(s, g, use_drag, k_drag)

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        _, y_next, _, vy_next = next_state
        if y_next <= 0.0 <= y_curr:
            frac = _cubic_hermite_root(y_curr, vy_curr, y_next, vy_next, dt)
            land_state = _cubic_hermite_interp(
                state, k1, next_state, f(next_state), frac, dt
            )
            x_land, _, vx_land, vy_land = land_state
            states.append(np.array([x_land, 0.0, vx_land, vy_land], dtype=float))
            times.append(t + frac * dt)
            break

        state = next_state
        t += dt
        states.append(state.copy())
        times.append(t)

    states_arr = np.array(states)
    x = states_arr[:, 0]
    y = states_arr[:, 1]
    vx = states_arr[:, 2]
    vy = states_arr[:, 3]

    impact_speed = float(np.hypot(vx[-1], vy[-1]))
    impact_angle = float(np.degrees(np.arctan2(vy[-1], vx[-1])))

    return {
        "x": x,
        "y": y,
        "time": float(times[-1]),
        "range": float(x[-1]),
        "max_height": float(np.max(y)),
        "impact_speed": impact_speed,
        "impact_angle": impact_angle,
    }


def simulate_ivp(
    v0, angle, h0, mass, cd, area, rho, g, use_drag, max_time=300.0
):
    """Интегрирование через scipy.integrate.solve_ivp с событием касания земли."""
    angle_rad = np.radians(angle)
    y0 = np.array(
        [0.0, h0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)], dtype=float
    )

    k_drag = 0.0
    if use_drag and mass > 0.0:
        k_drag = 0.5 * rho * cd * area / mass

    def rhs(t, state):
        return derivative(state, g, use_drag, k_drag)

    def hit_ground(t, state):
        _, y_val, _, _ = state
        return y_val

    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(
        rhs,
        t_span=(0.0, max_time),
        y0=y0,
        method="RK45",
        events=hit_ground,
        rtol=1e-9,
        atol=1e-11,
    )

    x_arr, y_arr, vx_arr, vy_arr = sol.y
    y_clean = np.maximum(0.0, y_arr)

    impact_speed = float(np.hypot(vx_arr[-1], vy_arr[-1]))
    impact_angle = float(np.degrees(np.arctan2(vy_arr[-1], vx_arr[-1])))

    return {
        "x": x_arr,
        "y": y_clean,
        "time": float(sol.t[-1]),
        "range": float(x_arr[-1]),
        "max_height": float(np.max(y_clean)),
        "impact_speed": impact_speed,
        "impact_angle": impact_angle,
    }


def find_optimal_angle(
    v0,
    h0=0.0,
    mass=0.145,
    cd=0.35,
    area=0.0043,
    rho=1.225,
    g=9.81,
    use_drag=True,
    dt=0.01,
):
    """Находит угол броска (в градусах), обеспечивающий максимальную дальность полёта."""

    def neg_range(angle_deg):
        res = simulate(
            v0=v0,
            angle=float(angle_deg),
            h0=h0,
            mass=mass,
            cd=cd,
            area=area,
            rho=rho,
            g=g,
            use_drag=use_drag,
            dt=dt,
        )
        return -res["range"]

    opt = minimize_scalar(neg_range, bounds=(1.0, 89.0), method="bounded")
    best_angle = float(opt.x)
    best_traj = simulate(
        v0=v0,
        angle=best_angle,
        h0=h0,
        mass=mass,
        cd=cd,
        area=area,
        rho=rho,
        g=g,
        use_drag=use_drag,
        dt=dt,
    )
    return {
        "optimal_angle": best_angle,
        "max_range": best_traj["range"],
        "time": best_traj["time"],
    }
