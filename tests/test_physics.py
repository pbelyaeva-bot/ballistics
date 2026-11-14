import pytest

from ballistics.physics import (
    analytical_vacuum,
    find_optimal_angle,
    simulate,
    simulate_ivp,
)


def test_vacuum_matches_analytical():
    params = dict(
        v0=45.0,
        angle=35.0,
        h0=1.2,
        mass=0.145,
        cd=0.0,
        area=0.0043,
        rho=1.225,
        g=9.81,
        use_drag=False,
    )
    exact = analytical_vacuum(v0=params["v0"], angle=params["angle"], h0=params["h0"], g=params["g"])
    rk4_res = simulate(**params, dt=0.001)
    ivp_res = simulate_ivp(**params)

    assert rk4_res["range"] == pytest.approx(exact["range"], rel=1e-5)
    assert rk4_res["time"] == pytest.approx(exact["time"], rel=1e-5)
    assert rk4_res["max_height"] == pytest.approx(exact["max_height"], rel=1e-4)

    assert ivp_res["range"] == pytest.approx(exact["range"], rel=1e-7)
    assert ivp_res["time"] == pytest.approx(exact["time"], rel=1e-7)


def test_drag_rk4_matches_solve_ivp():
    params = dict(
        v0=42.0,
        angle=28.0,
        h0=1.0,
        mass=0.145,
        cd=0.35,
        area=0.0043,
        rho=1.225,
        g=9.81,
        use_drag=True,
    )
    rk4_res = simulate(**params, dt=0.001)
    ivp_res = simulate_ivp(**params)

    assert rk4_res["range"] == pytest.approx(ivp_res["range"], rel=1e-4)
    assert rk4_res["time"] == pytest.approx(ivp_res["time"], rel=1e-4)
    assert rk4_res["impact_speed"] == pytest.approx(ivp_res["impact_speed"], rel=1e-4)


def test_rk4_convergence_order():
    params = dict(
        v0=40.0,
        angle=40.0,
        h0=0.0,
        mass=0.145,
        cd=0.0,
        area=0.0043,
        rho=1.225,
        g=9.81,
        use_drag=False,
    )
    exact_range = analytical_vacuum(v0=40.0, angle=40.0, h0=0.0, g=9.81)["range"]
    dts = [0.1, 0.05, 0.025, 0.0125]
    errors = [abs(simulate(**params, dt=dt)["range"] - exact_range) for dt in dts]

    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))


def test_optimal_angle_vacuum_and_drag():
    vac = find_optimal_angle(v0=45.0, h0=0.0, use_drag=False)
    assert vac["optimal_angle"] == pytest.approx(45.0, abs=0.2)

    drag = find_optimal_angle(v0=45.0, h0=0.0, cd=0.35, use_drag=True)
    assert 25.0 < drag["optimal_angle"] < 44.0
    assert drag["max_range"] < vac["max_range"]
