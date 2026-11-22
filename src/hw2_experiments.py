"""HW2: воспроизводимое сравнение двух групп признаков и две абляции.

Запуск из корня исходного проекта ДЗ1:
    python -m src.hw2_experiments --config configs/hw2_experiments.json

Основной внешний holdout ДЗ1 не используется в этом эксперименте.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from .identity_ablation import read_identity_selected
from .utils import load_config, load_transaction_columns, make_model, prepare_holdout


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_experiment_config(config, baseline):
    """Фиксация протокола: ровно две группы улучшений, четыре варианта."""
    expected = {
        "A_baseline": {"add_identity": False, "add_numeric": False},
        "B_identity": {"add_identity": True, "add_numeric": False},
        "C_C1_C2": {"add_identity": False, "add_numeric": True},
        "D_full": {"add_identity": True, "add_numeric": True},
    }
    if config["variants"] != expected:
        raise ValueError("Состав вариантов должен совпадать с предварительно зарегистрированным протоколом.")
    if config["fold_quantiles"] != [0.4, 0.6, 0.8]:
        raise ValueError("Границы фолдов зафиксированы как 0.4, 0.6 и 0.8.")
    if config["primary_metric"] != "roc_auc":
        raise ValueError("Основная метрика должна совпадать с ДЗ1: ROC-AUC.")
    if config["random_state"] != baseline["model"]["random_state"]:
        raise ValueError("Random state должен совпадать с исходным baseline.")
    if config["top_fraction"] != baseline["top_fraction"]:
        raise ValueError("Доля top-k должна совпадать с ДЗ1.")
    if config["additional_numeric_features"] != ["C1", "C2"]:
        raise ValueError("Новая числовая группа зафиксирована как [C1, C2].")
    if config["identity_features"] != ["DeviceType", "id_12", "id_15", "id_31"]:
        raise ValueError("Состав признаков identity должен совпадать с ДЗ1.")
    if set(config["additional_numeric_features"]) & set(baseline["numeric_features"]):
        raise ValueError("Повторяющиеся числовые признаки.")
    if baseline["train_fraction"] != 0.8:
        raise ValueError("Внешнее временное разделение ДЗ1 должно остаться 80/20.")
    if not 0 < config["top_fraction"] < 1:
        raise ValueError("top_fraction должен находиться в (0, 1).")


def prepare_training_data(baseline, config):
    """Читаются только необходимые колонки; identity присоединяется после внешнего split."""
    selected = list(dict.fromkeys([
        "TransactionID", baseline["time_column"], baseline["target_column"],
        *baseline["numeric_features"], *baseline["categorical_features"],
        *config["additional_numeric_features"],
    ]))
    transactions = load_transaction_columns(baseline["archive"], selected)
    if transactions["TransactionID"].isna().any() or not transactions["TransactionID"].is_unique:
        raise ValueError("TransactionID обязан быть уникальным и непустым.")
    if transactions[baseline["target_column"]].isna().any():
        raise ValueError("Обнаружены пропуски целевой переменной.")

    training, validation, cutoff = prepare_holdout(transactions, baseline)
    if len(training) + len(validation) != len(transactions):
        raise AssertionError("Внешнее разделение изменило количество объектов.")
    if len(training) != 472432 or len(validation) != 118108:
        raise ValueError("Размеры внешнего разделения отличаются от ДЗ1; проверьте архив.")
    for col in [*baseline["numeric_features"], *config["additional_numeric_features"]]:
        numeric = pd.to_numeric(training[col], errors="raise")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Признак {col} содержит бесконечности.")
        if (numeric.dropna() < 0).any():
            raise ValueError(f"Для log1p в признаке {col} есть отрицательные значения.")

    identity = read_identity_selected(baseline["archive"], config["identity_features"])
    training = training.merge(
        identity, on="TransactionID", how="left", validate="one_to_one",
        indicator=True, sort=False
    )
    if len(training) != 472432:
        raise AssertionError("Соединение identity изменило количество обучающих транзакций.")
    training["identity_status"] = np.where(training.pop("_merge") == "both", "present", "absent")
    for field in config["identity_features"]:
        s = training[field].astype(object)
        training[field] = s.where(s.isna(), s.astype(str))
    return training, {"outer_train_rows": len(training),
                      "outer_validation_rows": len(validation),
                      "outer_cutoff_dt": float(cutoff),
                      "outer_train_max_dt": float(training[baseline["time_column"]].max()),
                      "outer_validation_min_dt": float(validation[baseline["time_column"]].min())}


def _metric_record(y_true, scores, fraction):
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    selected = np.argsort(-scores, kind="mergesort")[:k]
    tp = int(y_true[selected].sum())
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision_at_1pct": float(tp / k),
        "recall_at_1pct": float(tp / int(y_true.sum())),
        "k": k,
        "tp_at_1pct": tp,
    }


def run_experiments(training, baseline, config):
    """Каждый из четырёх вариантов обучается на трёх идентичных временных фолдах."""
    time_col, target = baseline["time_column"], baseline["target_column"]
    t = training[time_col]
    cuts = t.quantile(config["fold_quantiles"]).to_numpy(dtype=float)
    if not np.all(np.diff(cuts) > 0):
        raise ValueError("Внутренние временные границы совпали.")
    fold_bounds = [(cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], np.inf)]
    rows = []
    for fold, (train_end, val_end) in enumerate(fold_bounds, start=1):
        fit = training.loc[t < train_end]
        val = training.loc[(t >= train_end) & (t < val_end)]
        if not len(fit) or not len(val):
            raise ValueError(f"Fold {fold}: пустая выборка.")
        if fit[time_col].max() >= val[time_col].min():
            raise AssertionError(f"Fold {fold}: нарушена хронология.")
        y_fit = fit[target].astype(int).to_numpy()
        y_val = val[target].astype(int).to_numpy()
        if np.unique(y_fit).size != 2 or np.unique(y_val).size != 2:
            raise ValueError(f"Fold {fold}: требуются оба класса.")

        for variant, options in config["variants"].items():
            num = list(baseline["numeric_features"])
            cat = list(baseline["categorical_features"])
            if options["add_numeric"]:
                num += config["additional_numeric_features"]
            if options["add_identity"]:
                cat += ["identity_status", *config["identity_features"]]
            cfg = {**baseline, "numeric_features": num, "categorical_features": cat}
            features = num + cat
            model = make_model(cfg)  # Та же LogisticRegression и та же схема preprocessing, что в ДЗ1.
            tic = time.perf_counter()
            model.fit(fit[features], y_fit)
            scores = model.predict_proba(val[features])[:, 1]
            elapsed = time.perf_counter() - tic
            if not np.isfinite(scores).all():
                raise ValueError(f"Некорректные предсказания: fold={fold}, variant={variant}")
            metrics = _metric_record(y_val, scores, config["top_fraction"])
            record = {
                "run_id": f"hw2_f{fold}_{variant}",
                "fold": fold,
                "variant": variant,
                "hypothesis": (
                    "Контроль: исходный baseline" if variant == "A_baseline" else
                    "H_identity: вклад полей identity" if variant == "B_identity" else
                    "H_numeric: вклад C1 и C2" if variant == "C_C1_C2" else
                    "H_joint: совместное использование обеих групп"
                ),
                "add_identity": options["add_identity"],
                "add_numeric": options["add_numeric"],
                "numeric_features": ",".join(num),
                "categorical_features": ",".join(cat),
                "seed": config["random_state"],
                "model": "LogisticRegression(liblinear, C=1.0, max_iter=300)",
                "preprocessing": "median + log1p + StandardScaler; constant + OneHotEncoder",
                "train_rows": int(len(fit)),
                "validation_rows": int(len(val)),
                "train_max_dt": float(fit[time_col].max()),
                "validation_min_dt": float(val[time_col].min()),
                "validation_max_dt": float(val[time_col].max()),
                "validation_fraud_rate": float(y_val.mean()),
                "fit_predict_seconds": float(elapsed),
                **metrics,
            }
            rows.append(record)
            print(f"Fold {fold} · {variant}: ROC-AUC={metrics['roc_auc']:.5f}, "
                  f"AP={metrics['average_precision']:.5f}, "
                  f"P@1%={metrics['precision_at_1pct']:.4f}")
    return pd.DataFrame(rows)


def summarize_experiments(runs):
    metrics = ["roc_auc", "average_precision", "precision_at_1pct", "recall_at_1pct"]
    summary = runs.groupby("variant", as_index=False, sort=False).agg(**{
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        "n_folds": ("fold", "size"),
    })
    baseline = summary.loc[summary["variant"] == "A_baseline"].iloc[0]
    for metric in metrics:
        summary[f"delta_{metric}_vs_A"] = summary[f"{metric}_mean"] - baseline[f"{metric}_mean"]
    return summary


def build_ablations(runs):
    """Две абляции: полная модель минус identity; полная модель минус C1/C2."""
    m = ["roc_auc", "average_precision", "precision_at_1pct", "recall_at_1pct"]
    full = runs.loc[runs["variant"] == "D_full"].set_index("fold")
    rows = []
    for label, reduced in [
        ("without_identity", "C_C1_C2"),
        ("without_C1_C2", "B_identity"),
    ]:
        part = runs.loc[runs["variant"] == reduced].set_index("fold")
        if not full.index.equals(part.index):
            raise AssertionError("У абляций не совпадают временные фолды.")
        for fold in full.index:
            row = {"fold": int(fold), "ablation": label, "full_variant": "D_full", "reduced_variant": reduced}
            for metric in m:
                row["full_" + metric] = float(full.loc[fold, metric])
                row["without_" + metric] = float(part.loc[fold, metric])
                row["delta_full_minus_without_" + metric] = float(full.loc[fold, metric] - part.loc[fold, metric])
            rows.append(row)
    by_fold = pd.DataFrame(rows)
    agg = by_fold.groupby("ablation", sort=False)[
        ["delta_full_minus_without_" + metric for metric in m]
    ].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    return by_fold, agg


def check_original_baseline(runs, comparison_path="results/temporal_cv_results.csv"):
    path = Path(comparison_path)
    if not path.is_file():
        return {"comparison_available": False, "reason": f"Не найден {path}"}
    previous = pd.read_csv(path).sort_values("fold")
    current = runs.loc[runs["variant"] == "A_baseline"].sort_values("fold")
    if not np.array_equal(previous["fold"].to_numpy(), current["fold"].to_numpy()):
        raise AssertionError("Не совпадают номера фолдов контрольной модели.")
    diffs = {}
    for metric in ["roc_auc", "average_precision"]:
        diff = np.abs(previous[metric].to_numpy() - current[metric].to_numpy())
        diffs[metric] = float(np.max(diff))
        if not np.allclose(previous[metric], current[metric], atol=1e-5, rtol=0):
            raise AssertionError(f"Контрольная модель ДЗ2 не воспроизводит ДЗ1 по {metric}.")
    return {"comparison_available": True, "max_abs_difference": diffs}


def save_outputs(runs, summary, abl_fold, abl_summary, config, baseline, config_path, outer_info, reference):
    output = Path(config["results_dir"])
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "experiments_by_fold.csv", index=False)
    summary.to_csv(output / "experiments_summary.csv", index=False)
    abl_fold.to_csv(output / "ablation_by_fold.csv", index=False)
    abl_summary.to_csv(output / "ablation_summary.csv", index=False)
    runs[["run_id", "hypothesis", "variant", "fold", "seed", "model", "preprocessing",
          "numeric_features", "categorical_features", "train_rows", "validation_rows",
          "roc_auc", "average_precision", "precision_at_1pct", "recall_at_1pct",
          "fit_predict_seconds"]].to_csv(output / "experiment_registry.csv", index=False)
    # Финальный manifest делает условия эксперимента доступными без исходных CSV.
    manifest = {
        "experiment_id": config["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "baseline_config": baseline,
        "config_sha256": _sha256(config_path),
        "baseline_config_sha256": _sha256(config["baseline_config"]),
        "original_baseline_control": reference,
        "outer_split": outer_info,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        },
        "selection_protocol": "Comparison on 3 fixed expanding-window folds inside HW1 train; outer HW1 validation is not used.",
        "note": "Только агрегированные результаты. Построчные данные и предсказания не сохраняются.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {"A_baseline": "A: baseline", "B_identity": "B: identity",
              "C_C1_C2": "C: C1/C2", "D_full": "D: full"}
    for metric, filename, title in [
        ("roc_auc", "roc_auc_by_fold.png", "ROC-AUC на одинаковых временных фолдах"),
        ("average_precision", "ap_by_fold.png", "Average Precision на одинаковых временных фолдах"),
    ]:
        plt.figure(figsize=(9, 4.7))
        for name, group in runs.groupby("variant", sort=False):
            plt.plot(group["fold"], group[metric], marker="o", label=labels[name])
        plt.xticks([1, 2, 3]); plt.xlabel("Номер временного фолда")
        plt.ylabel("ROC-AUC" if metric == "roc_auc" else "Average Precision")
        plt.title(title); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
        plt.savefig(output / filename, dpi=150); plt.close()

    for metric, filename in [("roc_auc", "ablation_delta_roc_auc.png"),
                             ("average_precision", "ablation_delta_ap.png")]:
        subset = abl_summary.set_index("ablation")
        metric_col = f"delta_full_minus_without_{metric}_mean"
        plt.figure(figsize=(8, 4))
        plt.bar(subset.index, subset[metric_col])
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Метрика полной модели − метрика без компонента")
        plt.title("Вклад компонента: " + metric)
        plt.grid(axis="y", alpha=0.2); plt.tight_layout()
        plt.savefig(output / filename, dpi=150); plt.close()
    return output


def main():
    parser = argparse.ArgumentParser(description="HW2: два изменения и две абляции")
    parser.add_argument("--config", default="configs/hw2_experiments.json")
    args = parser.parse_args()
    cfg = _load_json(args.config)
    baseline = load_config(cfg["baseline_config"])
    validate_experiment_config(cfg, baseline)
    data, outer_info = prepare_training_data(baseline, cfg)
    print("Внешний split ДЗ1:", outer_info)
    runs = run_experiments(data, baseline, cfg)
    reference = check_original_baseline(runs)
    print("Проверка контрольной модели:", reference)
    summary = summarize_experiments(runs)
    abl_fold, abl_summary = build_ablations(runs)
    output = save_outputs(runs, summary, abl_fold, abl_summary, cfg, baseline, args.config, outer_info, reference)
    print("\nСводка:\n", summary.to_string(index=False))
    print("\nАбляции:\n", abl_summary.to_string(index=False))
    print("\nСохранено в:", output.resolve())


if __name__ == "__main__":
    main()
