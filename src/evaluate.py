"""Локальная оценка зафиксированного baseline на поздних транзакциях.

Запуск: python -m src.evaluate --config configs/baseline.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .utils import load_checked_data, load_config, prepare_holdout


def main():
    parser = argparse.ArgumentParser(description="Оценка ROC-AUC/AP/Precision@1%")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    saved_dir = Path(cfg["artifact_dir"])
    with (saved_dir / "baseline_metadata.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    if info["config"] != cfg:
        raise ValueError("Конфигурация отличается от использованной при обучении.")
    # Загрузка joblib допустима только для самостоятельно созданного доверенного файла.
    model = joblib.load(saved_dir / "baseline_logreg.joblib")
    data = load_checked_data(cfg)
    train, val, cutoff = prepare_holdout(data, cfg)
    if not np.isclose(cutoff, float(info["split_cutoff"])):
        raise ValueError("Граница времени изменилась по сравнению с обучением.")
    if len(train) != info["train_rows"] or len(val) != info["validation_rows"]:
        raise ValueError("Размеры частей отличаются от сохранённых при обучении.")
    features = cfg["numeric_features"] + cfg["categorical_features"]
    y_val = val[cfg["target_column"]].astype(int).to_numpy()
    if set(y_val) != {0, 1}:
        raise ValueError("На validation должны присутствовать оба класса.")

    pred = model.predict_proba(val[features])[:, 1]
    if not np.isfinite(pred).all():
        raise ValueError("Обнаружены некорректные значения предсказаний.")
    const = np.full(len(val), info["train_prior"], dtype=float)
    records = [
        {"model": "DummyClassifier (prior)", "ROC-AUC": roc_auc_score(y_val, const),
         "AP": average_precision_score(y_val, const)},
        {"model": "LogisticRegression (baseline)", "ROC-AUC": roc_auc_score(y_val, pred),
         "AP": average_precision_score(y_val, pred)},
    ]

    top = float(cfg["top_fraction"])
    if not 0 < top <= 1:
        raise ValueError("top_fraction должен лежать в (0, 1].")
    k = max(1, int(np.ceil(top * len(val))))
    order = np.argsort(-pred, kind="mergesort")
    flagged = np.zeros(len(val), dtype=bool)
    flagged[order[:k]] = True
    tn, fp, fn, tp = confusion_matrix(y_val, flagged.astype(int), labels=[0, 1]).ravel()

    results = Path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(results / "baseline_metrics.csv", index=False)
    summary = {
        "source": "local chronological holdout",
        "validation_rows": len(val),
        "validation_fraud": int(y_val.sum()),
        "roc_auc": float(records[1]["ROC-AUC"]),
        "average_precision": float(records[1]["AP"]),
        "dummy_roc_auc": float(records[0]["ROC-AUC"]),
        "dummy_ap": float(records[0]["AP"]),
        "top_fraction": top,
        "k": k,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision_at_top": float(tp / k),
        "recall_at_top": float(tp / int(y_val.sum())),
    }
    with (results / "baseline_script_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Top {top:.1%}: TP={tp}, FP={fp}, FN={fn}, TN={tn}; "
          f"precision={tp/k:.4f}, recall={tp/int(y_val.sum()):.4f}")
    print(f"Результаты: {results / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
