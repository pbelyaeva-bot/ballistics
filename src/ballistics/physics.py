import numpy as np
from scipy.integrate import solve_ivp


def derivative(state, g, use_drag, k_drag):
    vx = state[2]
    vy = state[3]

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


def simulate(
    v0, angle, h0, mass, cd, area, rho, g, use_drag, dt=0.005, max_time=300.0
):
    """Численное интегрирование траектории методом Рунге-Кутты 4 порядка."""
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
        if state[1] < 0.0:
            break

        def f(s):
            return derivative(s, g, use_drag, k_drag)

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if next_state[1] <= 0.0 <= state[1]:
            denom = state[1] - next_state[1]
            frac = state[1] / denom if denom > 1e-15 else 0.0
            frac = min(1.0, max(0.0, frac))

            land_state = state + frac * (next_state - state)
            land_state[1] = 0.0
            states.append(land_state)
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
        return state[1]

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

    x = sol.y[0]
    y = np.maximum(0.0, sol.y[1])
    vx = sol.y[2]
    vy = sol.y[3]

    impact_speed = float(np.hypot(vx[-1], vy[-1]))
    impact_angle = float(np.degrees(np.arctan2(vy[-1], vx[-1])))

    return {
        "x": x,
        "y": y,
        "time": float(sol.t[-1]),
        "range": float(x[-1]),
        "max_height": float(np.max(y)),
        "impact_speed": impact_speed,
        "impact_angle": impact_angle,
    }
