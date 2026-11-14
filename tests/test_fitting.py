import pandas as pd
import pytest

from ballistics.fitting import fit_cd, predict_ranges, split_train_test


def test_fit_cd_recovers_known_value():
    true_cd = 0.36
    synthetic = pd.DataFrame(
        {
            "v0_ms": [38.0, 42.0, 45.0, 48.0, 40.0, 44.0, 46.0, 50.0, 41.0, 47.0],
            "angle_deg": [25.0, 30.0, 32.0, 28.0, 35.0, 27.0, 33.0, 29.0, 31.0, 26.0],
        }
    )
    synthetic["distance_m"] = predict_ranges(synthetic, cd=true_cd, dt=0.01)

    train_df, test_df = split_train_test(synthetic, train_ratio=0.7)
    res = fit_cd(train_df, initial_cd=0.50, dt=0.01)

    assert res["success"]
    assert res["cd"] == pytest.approx(true_cd, abs=1e-3)

    test_preds = predict_ranges(test_df, cd=res["cd"], dt=0.01)
    assert test_preds == pytest.approx(test_df["distance_m"].to_numpy(), abs=0.05)
