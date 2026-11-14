from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ballistics.fitting import evaluate_models, fit_cd, split_train_test
from ballistics.statcast import fetch_statcast_data, prepare_fly_balls


def main():
    clean_path = Path("data/fly_balls_clean.csv")
    if clean_path.exists():
        df = pd.read_csv(clean_path)
    else:
        raw = fetch_statcast_data("2024-06-01", "2024-06-07", save_path="data/statcast_raw.csv")
        df = prepare_fly_balls(raw)
        df.to_csv(clean_path, index=False)

    print(f"Загружено чистых флай-болов: {len(df)}")
    train_df, test_df = split_train_test(df, train_ratio=0.8)
    print(f"Размер обучающей выборки (train): {len(train_df)}")
    print(f"Размер тестовой выборки (test):   {len(test_df)}")

    print("\nЗапуск оптимизации Cd (least_squares)...")
    fit_res = fit_cd(train_df)
    best_cd = fit_res["cd"]
    print(f"Подобранный коэффициент Cd = {best_cd:.4f} (вызовов функции: {fit_res['nfev']})")

    eval_res = evaluate_models(train_df, test_df, fitted_cd=best_cd)
    print("\nРезультаты на тестовой выборке (ошибка в метрах):")
    print(f"{'Модель':<28} | {'MAE, м':>8} | {'RMSE, м':>8} | {'Bias, м':>9}")
    print("-" * 62)
    for model_name, metrics in eval_res.items():
        print(
            f"{model_name:<28} | {metrics['mae_m']:>8.2f} | "
            f"{metrics['rmse_m']:>8.2f} | {metrics['bias_m']:>+9.2f}"
        )

    # Строим график: факт Statcast vs предсказание модели на тестовой выборке
    actual = test_df["distance_m"].to_numpy()
    pred_vac = eval_res["vacuum (Cd=0.0)"]["predictions"]
    pred_fit = eval_res[f"fitted (Cd={best_cd:.4f})"]["predictions"]

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, pred_vac, alpha=0.25, color="gray", label="Вакуум (Cd = 0)")
    plt.scatter(
        actual,
        pred_fit,
        alpha=0.5,
        color="tab:blue",
        label=f"Модель с сопротивлением (Cd = {best_cd:.3f})",
    )

    lims = [min(actual.min(), pred_fit.min()) - 5, max(actual.max(), pred_fit.max()) + 5]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Идеальное совпадение (y = x)")
    plt.xlim(lims)
    plt.ylim(lims)

    plt.xlabel("Реальная дальность Statcast, м")
    plt.ylabel("Предсказанная дальность модели, м")
    plt.title("Сравнение физической модели с данными Statcast (тестовая выборка)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/statcast_fit.png", dpi=150)
    print("\nГрафик сохранён в docs/statcast_fit.png")


if __name__ == "__main__":
    main()
