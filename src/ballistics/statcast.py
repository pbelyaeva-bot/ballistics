from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

# Константы перевода в СИ
MPH_TO_MS = 0.44704   # 1 миля в час = 0.44704 м/с
FT_TO_M = 0.3048      # 1 фут = 0.3048 м

# Физические параметры стандартного бейсбольного мяча (MLB)
BASEBALL_MASS_KG = 0.145
BASEBALL_RADIUS_M = 0.037
BASEBALL_AREA_M2 = 3.141592653589793 * (BASEBALL_RADIUS_M**2)  # ~0.0043 м²
DEFAULT_HIT_HEIGHT_M = 1.0  # средняя высота точки удара битой (зона страйка)


def fetch_statcast_data(start_dt: str, end_dt: str, save_path: str | None = None) -> pd.DataFrame:
    df = statcast(start_dt=start_dt, end_dt=end_dt)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


def inspect_columns(df: pd.DataFrame) -> dict:
    candidate_cols = [
        "bb_type",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "events",
        "home_team",
        "game_date",
    ]
    found = [col for col in candidate_cols if col in df.columns]
    missing = [col for col in candidate_cols if col not in df.columns]

    report = {
        "total_rows": len(df),
        "found_columns": found,
        "missing_columns": missing,
        "bb_type_counts": df["bb_type"].value_counts(dropna=False).to_dict()
        if "bb_type" in df.columns
        else {},
        "summary": {},
    }

    for col in ["launch_speed", "launch_angle", "hit_distance_sc"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            report["summary"][col] = {
                "count": int(series.count()),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }

    return report


def prepare_fly_balls(df: pd.DataFrame) -> pd.DataFrame:
    required = ["bb_type", "launch_speed", "launch_angle", "hit_distance_sc"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"В таблице отсутствует обязательный столбец: {col}")

    clean = df.copy()

    # Оставляем только флай-болы (ground_ball катятся по траве, popup/line_drive имеют иную динамику)
    clean = clean[clean["bb_type"] == "fly_ball"].copy()

    # Приводим к числам и удаляем строки с пропусками
    for col in ["launch_speed", "launch_angle", "hit_distance_sc"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=["launch_speed", "launch_angle", "hit_distance_sc"])

    # Фильтруем физически некорректные выбросы и отрицательные углы вылета
    clean = clean[
        (clean["launch_speed"] > 30.0)
        & (clean["launch_angle"] > 5.0)
        & (clean["launch_angle"] < 85.0)
        & (clean["hit_distance_sc"] > 30.0)
    ].copy()

    # Перевод в единицы СИ
    clean["v0_ms"] = clean["launch_speed"] * MPH_TO_MS
    clean["angle_deg"] = clean["launch_angle"]
    clean["distance_m"] = clean["hit_distance_sc"] * FT_TO_M

    keep_cols = ["v0_ms", "angle_deg", "distance_m", "launch_speed", "launch_angle", "hit_distance_sc"]
    for extra in ["events", "home_team", "game_date"]:
        if extra in clean.columns:
            keep_cols.append(extra)

    return clean[keep_cols].reset_index(drop=True)
