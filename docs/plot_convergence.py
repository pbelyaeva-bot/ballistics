import matplotlib.pyplot as plt
import numpy as np

from ballistics.physics import analytical_vacuum, simulate, simulate_ivp


def main():
    params = dict(
        v0=45.0,
        angle=35.0,
        h0=1.0,
        mass=0.145,
        cd=0.35,
        area=0.0043,
        rho=1.225,
        g=9.81,
        use_drag=True,
    )

    ref_range = simulate_ivp(**params)["range"]

    vac_params = {**params, "use_drag": False, "h0": 0.0}
    exact_vac_range = analytical_vacuum(
        v0=vac_params["v0"], angle=vac_params["angle"], h0=0.0, g=vac_params["g"]
    )["range"]

    dts = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
    errors_drag = [abs(simulate(**params, dt=dt)["range"] - ref_range) for dt in dts]
    errors_vac = [abs(simulate(**vac_params, dt=dt)["range"] - exact_vac_range) for dt in dts]

    plt.figure(figsize=(8, 5))
    plt.loglog(dts, errors_vac, "o-", label="RK4 vs Аналитика (вакуум)")
    plt.loglog(dts, errors_drag, "s-", label="RK4 vs solve_ivp (с сопротивлением)")

    ref_line = errors_drag[0] * (dts / dts[0]) ** 4
    plt.loglog(dts, ref_line, "--", color="gray", label="Теоретический наклон $O(dt^4)$")

    plt.xlabel("Шаг интегрирования dt, с")
    plt.ylabel("Абсолютная ошибка дальности, м")
    plt.title("Сходимость метода Рунге-Кутты 4-го порядка (RK4)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/convergence.png", dpi=150)
    print("График сохранён в docs/convergence.png")


if __name__ == "__main__":
    main()
