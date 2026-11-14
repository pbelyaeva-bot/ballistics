from ballistics.statcast import (
    fetch_statcast_data,
    inspect_columns,
    prepare_fly_balls,
)


def main():
    raw_path = "data/statcast_raw.csv"
    print("Загрузка данных Statcast за 2024-06-01 .. 2024-06-07...")
    df_raw = fetch_statcast_data("2024-06-01", "2024-06-07", save_path=raw_path)

    report = inspect_columns(df_raw)
    print(f"\nВсего записей таблице: {report['total_rows']}")
    print(f"Найденные целевые столбцы: {report['found_columns']}")
    print("\nРаспределение типов ударов (bb_type):")
    for k, v in report["bb_type_counts"].items():
        print(f"  {k}: {v}")

    print("\nСтатистика по числовым полям:")
    for col, stats in report["summary"].items():
        print(f"  {col}: {stats}")

    fly_balls = prepare_fly_balls(df_raw)
    fly_balls.to_csv("data/fly_balls_clean.csv", index=False)
    print(f"\nОтобрано чистых флай-болов: {len(fly_balls)}")
    print("\nПервые 5 строк после перевода в СИ (v0_ms, angle_deg, distance_m):")
    print(fly_balls[["v0_ms", "angle_deg", "distance_m", "events"]].head())


if __name__ == "__main__":
    main()
