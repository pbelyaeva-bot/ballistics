import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def _validate_sim_params(v0, angle, h0, mass, cd, area, rho, g, dt=None, max_time=None):
    if v0 <= 0.0:
        raise ValueError("Начальная скорость v0 должна быть больше нуля.")
    if not (0.0 <= angle <= 90.0):
        raise ValueError("Угол броска должен лежать в диапазоне [0, 90] градусов.")
    if h0 < 0.0:
        raise ValueError("Начальная высота h0 не может быть отрицательной.")
    if mass <= 0.0:
        raise ValueError("Масса тела mass должна быть больше нуля.")
    if g <= 0.0:
        raise ValueError("Ускорение свободного падения g должно быть больше нуля.")
    if cd < 0.0 or area < 0.0 or rho < 0.0:
        raise ValueError("Параметры cd, area и rho не могут быть отрицательными.")
    if dt is not None and dt <= 0.0:
        raise ValueError("Шаг интегрирования dt должен быть больше нуля.")
    if max_time is not None and max_time <= 0.0:
        raise ValueError("Максимальное время расчёта max_time должно быть больше нуля.")


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
    if v0 <= 0.0 or g <= 0.0 or h0 < 0.0:
        raise ValueError("Некорректные входные параметры для аналитического расчёта.")

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


def _find_ground_fraction(y0, vy0, y1, vy1, dt):
    """Ищет долю шага s in (0, 1) до пересечения y = 0 по кубическому многочлену Эрмита."""
    m0 = vy0 * dt
    m1 = vy1 * dt

    def y_poly(s):
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if y_poly(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _interp_state(s0, ds0, s1, ds1, s, dt):
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return h00 * s0 + h10 * (ds0 * dt) + h01 * s1 + h11 * (ds1 * dt)


def simulate(
    v0, angle, h0, mass, cd, area, rho, g, use_drag, dt=0.005, max_time=300.0
):
    """Численное интегрирование траектории методом Рунге-Кутты 4-го порядка (RK4)."""
    _validate_sim_params(v0, angle, h0, mass, cd, area, rho, g, dt=dt, max_time=max_time)

    angle_rad = np.radians(angle)
    state = np.array(
        [0.0, h0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)], dtype=float
    )
    k_drag = (0.5 * rho * cd * area / mass) if use_drag else 0.0

    t = 0.0
    states = [state.copy()]
    times = [0.0]
    steps = int(np.ceil(max_time / dt))

    for _ in range(steps):
        _, y_curr, _, vy_curr = state
        if y_curr < 0.0:
            break

        k1 = derivative(state, g, use_drag, k_drag)
        k2 = derivative(state + 0.5 * dt * k1, g, use_drag, k_drag)
        k3 = derivative(state + 0.5 * dt * k2, g, use_drag, k_drag)
        k4 = derivative(state + dt * k3, g, use_drag, k_drag)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        _, y_next, _, vy_next = next_state
        if y_next <= 0.0 <= y_curr:
            frac = _find_ground_fraction(y_curr, vy_curr, y_next, vy_next, dt)
            ds_next = derivative(next_state, g, use_drag, k_drag)
            land_state = _interp_state(state, k1, next_state, ds_next, frac, dt)
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
    _validate_sim_params(v0, angle, h0, mass, cd, area, rho, g, max_time=max_time)

    angle_rad = np.radians(angle)
    y0 = np.array(
        [0.0, h0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)], dtype=float
    )
    k_drag = (0.5 * rho * cd * area / mass) if use_drag else 0.0

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
        rtol=1e-12,
        atol=1e-13,
    )

    x_arr, y_arr, vx_arr, vy_arr = sol.y
    y_clean = np.maximum(0.0, y_arr)

    if sol.t_events and len(sol.t_events[0]) > 0:
        t_end = float(sol.t_events[0][-1])
        x_end, _, vx_end, vy_end = sol.y_events[0][-1]
        x_arr = x_arr.copy()
        y_clean = y_clean.copy()
        x_arr[-1] = x_end
        y_clean[-1] = 0.0
    else:
        t_end = float(sol.t[-1])
        x_end = float(x_arr[-1])
        vx_end = float(vx_arr[-1])
        vy_end = float(vy_arr[-1])

    impact_speed = float(np.hypot(vx_end, vy_end))
    impact_angle = float(np.degrees(np.arctan2(vy_end, vx_end)))

    return {
        "x": x_arr,
        "y": y_clean,
        "time": t_end,
        "range": float(x_end),
        "max_height": float(np.max(y_clean)),
        "impact_speed": impact_speed,
        "impact_angle": impact_angle,
    }


def find_optimal_angle(
    v0,
    h0=0.0,
    mass=0.145,
    cd=0.3392,
    area=0.0043,
    rho=1.225,
    g=9.81,
    use_drag=True,
    dt=0.01,
):
    """Находит угол броска (в градусах), дающий максимальную дальность полёта."""
    _validate_sim_params(v0, 45.0, h0, mass, cd, area, rho, g, dt=dt)

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
