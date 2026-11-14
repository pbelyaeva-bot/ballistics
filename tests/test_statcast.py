import pandas as pd
import pytest

from ballistics.statcast import FT_TO_M, MPH_TO_MS, inspect_columns, prepare_fly_balls


def test_prepare_fly_balls_filtering_and_units():
    sample = pd.DataFrame(
        {
            "bb_type": ["fly_ball", "ground_ball", "fly_ball", "line_drive"],
            "launch_speed": [100.0, 95.0, 90.0, 98.0],
            "launch_angle": [28.0, -5.0, None, 15.0],
            "hit_distance_sc": [400.0, 20.0, 320.0, 250.0],
        }
    )

    report = inspect_columns(sample)
    assert report["total_rows"] == 4
    assert report["bb_type_counts"]["fly_ball"] == 2

    clean = prepare_fly_balls(sample)
    # Должна остаться только 1 валидная запись
    assert len(clean) == 1
    assert clean.loc[0, "v0_ms"] == pytest.approx(100.0 * MPH_TO_MS)
    assert clean.loc[0, "distance_m"] == pytest.approx(400.0 * FT_TO_M)
    assert clean.loc[0, "angle_deg"] == pytest.approx(28.0)
