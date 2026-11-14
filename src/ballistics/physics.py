import numpy as np


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


def simulate(v0, angle, h0, mass, cd, area, rho, g, use_drag,
             dt=0.005, max_time=300.0):
    angle_rad = np.radians(angle)
    state = np.array([0.0, h0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])

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

    states = np.array(states)
    x = states[:, 0]
    y = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]

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
