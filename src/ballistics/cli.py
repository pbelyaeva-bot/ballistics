import argparse
from pathlib import Path

import pandas as pd

from ballistics.fitting import evaluate_models, fit_cd, split_train_test
from ballistics.physics import find_optimal_angle, simulate, simulate_ivp
from ballistics.statcast import (
    BASEBALL_AREA_M2,
    BASEBALL_MASS_KG,
    DEFAULT_HIT_HEIGHT_M,
    fetch_statcast_data,
    inspect_columns,
    prepare_fly_balls,
)


def cmd_fetch(args):
    print(f"Загрузка данных Statcast с {args.start} по {args.end}...")
    raw = fetch_statcast_data(args.start, args.end, save_path=args.raw_out)
    report = inspect_columns(raw)
    print(f"Всего записей: {report['total_rows']}, типы ударов: {report['bb_type_counts']}")

    clean = prepare_fly_balls(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    print(f"Сохранено чистых флай-болов: {len(clean)} -> {out}")


def cmd_fit(args):
    df = pd.read_csv(args.data)
    train_df, test_df = split_train_test(df, train_ratio=args.train_ratio)
    fit_res = fit_cd(train_df)
    best_cd = fit_res["cd"]

    print(f"Подобранный коэффициент Cd = {best_cd:.4f} (train={len(train_df)}, test={len(test_df)})")
    eval_res = evaluate_models(train_df, test_df, fitted_cd=best_cd)

    print(f"\n{'Модель':<28} | {'MAE, м':>8} | {'RMSE, м':>8} | {'Bias, м':>9}")
    print("-" * 62)
    for name, m in eval_res.items():
        print(f"{name:<28} | {m['mae_m']:>8.2f} | {m['rmse_m']:>8.2f} | {m['bias_m']:>+9.2f}")


def cmd_simulate(args):
    fn = simulate_ivp if args.solver == "ivp" else simulate
    res = fn(
        v0=args.v0,
        angle=args.angle,
        h0=args.h0,
        mass=args.mass,
        cd=args.cd,
        area=args.area,
        rho=args.rho,
        g=args.g,
        use_drag=not args.vacuum,
    )
    mode = "вакуум" if args.vacuum else f"с сопротивлением (Cd={args.cd})"
    print(f"Расчёт ({args.solver}, {mode}):")
    print(f"  Дальность:          {res['range']:.2f} м")
    print(f"  Макс. высота:       {res['max_height']:.2f} м")
    print(f"  Время полёта:       {res['time']:.2f} с")
    print(f"  Скорость падения:   {res['impact_speed']:.2f} м/с")
    print(f"  Угол падения:       {res['impact_angle']:.2f}°")


def cmd_optimal_angle(args):
    res = find_optimal_angle(
        v0=args.v0,
        h0=args.h0,
        mass=args.mass,
        cd=args.cd,
        area=args.area,
        rho=args.rho,
        g=args.g,
        use_drag=not args.vacuum,
    )
    mode = "вакуум" if args.vacuum else f"Cd={args.cd}"
    print(f"Оптимальный угол броска ({mode}, v0={args.v0} м/с):")
    print(f"  Угол макс. дальности: {res['optimal_angle']:.2f}°")
    print(f"  Максимальная дальность: {res['max_range']:.2f} м")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ballistics",
        description="Баллистический калькулятор и калибровка по данным Statcast",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. fetch
    p_fetch = sub.add_parser("fetch", help="Скачать и очистить данные флай-болов Statcast")
    p_fetch.add_argument("--start", required=True, help="Дата начала YYYY-MM-DD")
    p_fetch.add_argument("--end", required=True, help="Дата окончания YYYY-MM-DD")
    p_fetch.add_argument("--raw-out", default="data/statcast_raw.csv", help="Путь для сырого CSV")
    p_fetch.add_argument("--out", default="data/fly_balls_clean.csv", help="Путь для очищенного CSV")
    p_fetch.set_defaults(func=cmd_fetch)

    # 2. fit
    p_fit = sub.add_parser("fit", help="Подобрать Cd по данным и оценить ошибку на тесте")
    p_fit.add_argument("--data", default="data/fly_balls_clean.csv", help="Путь к очищенному CSV")
    p_fit.add_argument("--train-ratio", type=float, default=0.8, help="Доля обучающей выборки")
    p_fit.set_defaults(func=cmd_fit)

    # 3. simulate
    p_sim = sub.add_parser("simulate", help="Рассчитать траекторию одного броска")
    p_sim.add_argument("--v0", type=float, required=True, help="Начальная скорость, м/с")
    p_sim.add_argument("--angle", type=float, required=True, help="Угол броска, градусы")
    p_sim.add_argument("--h0", type=float, default=DEFAULT_HIT_HEIGHT_M, help="Начальная высота, м")
    p_sim.add_argument("--mass", type=float, default=BASEBALL_MASS_KG, help="Масса, кг")
    p_sim.add_argument("--cd", type=float, default=0.35, help="Коэффициент сопротивления Cd")
    p_sim.add_argument("--area", type=float, default=BASEBALL_AREA_M2, help="Площадь сечения, м²")
    p_sim.add_argument("--rho", type=float, default=1.225, help="Плотность воздуха, кг/м³")
    p_sim.add_argument("--g", type=float, default=9.81, help="Ускорение свободного падения, м/с²")
    p_sim.add_argument("--vacuum", action="store_true", help="Считать без сопротивления воздуха")
    p_sim.add_argument("--solver", choices=["rk4", "ivp"], default="rk4", help="Метод интегрирования")
    p_sim.set_defaults(func=cmd_simulate)

    # 4. optimal-angle
    p_opt = sub.add_parser("optimal-angle", help="Найти угол максимальной дальности")
    p_opt.add_argument("--v0", type=float, required=True, help="Начальная скорость, м/с")
    p_opt.add_argument("--h0", type=float, default=0.0, help="Начальная высота, м")
    p_opt.add_argument("--mass", type=float, default=BASEBALL_MASS_KG, help="Масса, кг")
    p_opt.add_argument("--cd", type=float, default=0.35, help="Коэффициент сопротивления Cd")
    p_opt.add_argument("--area", type=float, default=BASEBALL_AREA_M2, help="Площадь сечения, м²")
    p_opt.add_argument("--rho", type=float, default=1.225, help="Плотность воздуха, кг/м³")
    p_opt.add_argument("--g", type=float, default=9.81, help="Ускорение свободного падения, м/с²")
    p_opt.add_argument("--vacuum", action="store_true", help="Считать без сопротивления воздуха")
    p_opt.set_defaults(func=cmd_optimal_angle)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
