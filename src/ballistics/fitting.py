import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ballistics.physics import simulate
from ballistics.statcast import (
    BASEBALL_AREA_M2,
    BASEBALL_MASS_KG,
    DEFAULT_HIT_HEIGHT_M,
)


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    shuffle: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит выборку на обучающую и тестовую части."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio должен быть в интервале (0, 1).")

    work_df = (
        df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        if shuffle
        else df.reset_index(drop=True)
    )

    split_idx = int(len(work_df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(work_df):
        raise ValueError("Недостаточно данных для разделения на train и test.")

    train_df = work_df.iloc[:split_idx].reset_index(drop=True)
    test_df = work_df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def predict_ranges(
    df: pd.DataFrame,
    cd: float,
    h0: float = DEFAULT_HIT_HEIGHT_M,
    mass: float = BASEBALL_MASS_KG,
    area: float = BASEBALL_AREA_M2,
    rho: float = 1.225,
    g: float = 9.81,
    dt: float = 0.01,
) -> np.ndarray:
    use_drag = cd > 0.0
    preds = np.empty(len(df), dtype=float)

    for i, (v0, angle) in enumerate(zip(df["v0_ms"], df["angle_deg"])):
        res = simulate(
            v0=float(v0),
            angle=float(angle),
            h0=h0,
            mass=mass,
            cd=float(cd),
            area=area,
            rho=rho,
            g=g,
            use_drag=use_drag,
            dt=dt,
        )
        preds[i] = res["range"]

    return preds


def compute_error_metrics(actual_m: np.ndarray, predicted_m: np.ndarray) -> dict:
    errors = predicted_m - actual_m
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    return {"mae_m": mae, "rmse_m": rmse, "bias_m": bias}


def fit_cd(
    train_df: pd.DataFrame,
    initial_cd: float = 0.40,
    h0: float = DEFAULT_HIT_HEIGHT_M,
    mass: float = BASEBALL_MASS_KG,
    area: float = BASEBALL_AREA_M2,
    rho: float = 1.225,
    g: float = 9.81,
    dt: float = 0.01,
) -> dict:
    """Подбирает коэффициент сопротивления Cd методом наименьших квадратов."""
    actual = train_df["distance_m"].to_numpy(dtype=float)

    def residuals(params):
        cd_val = float(params[0])
        pred = predict_ranges(train_df, cd=cd_val, h0=h0, mass=mass, area=area, rho=rho, g=g, dt=dt)
        return pred - actual

    result = least_squares(
        residuals,
        x0=[initial_cd],
        bounds=([0.0], [2.0]),
        xtol=1e-4,
        ftol=1e-4,
    )

    best_cd = float(result.x[0])
    train_preds = predict_ranges(train_df, cd=best_cd, h0=h0, mass=mass, area=area, rho=rho, g=g, dt=dt)
    train_metrics = compute_error_metrics(actual, train_preds)

    return {
        "cd": best_cd,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "train_metrics": train_metrics,
    }


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fitted_cd: float,
    h0: float = DEFAULT_HIT_HEIGHT_M,
    mass: float = BASEBALL_MASS_KG,
    area: float = BASEBALL_AREA_M2,
    rho: float = 1.225,
    g: float = 9.81,
    dt: float = 0.01,
) -> dict:
    """Сравнивает вакуум, гладкий шар (Cd=0.47) и подобранный Cd на тестовой выборке."""
    actual_test = test_df["distance_m"].to_numpy(dtype=float)

    models = {
        "vacuum (Cd=0.0)": 0.0,
        "smooth_sphere (Cd=0.47)": 0.47,
        f"fitted (Cd={fitted_cd:.4f})": fitted_cd,
    }

    evaluation = {}
    for name, cd_val in models.items():
        preds = predict_ranges(
            test_df, cd=cd_val, h0=h0, mass=mass, area=area, rho=rho, g=g, dt=dt
        )
        evaluation[name] = {
            "cd": cd_val,
            "predictions": preds,
            **compute_error_metrics(actual_test, preds),
        }

    return evaluation
