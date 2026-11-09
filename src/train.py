"""Обучение baseline на ранних транзакциях хронологической выборки.

Запуск: python -m src.train --config configs/baseline.json
"""

import argparse
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .utils import load_checked_data, load_config, make_model, prepare_holdout


def main():
    parser = argparse.ArgumentParser(description="Обучение LogisticRegression baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_checked_data(cfg)
    training, validation, cutoff = prepare_holdout(data, cfg)
    features = cfg["numeric_features"] + cfg["categorical_features"]
    target = cfg["target_column"]
    y_train = training[target].astype(int).to_numpy()
    if set(y_train) != {0, 1}:
        raise ValueError("Обучающая часть должна содержать оба класса.")

    model = make_model(cfg)
    model.fit(training[features], y_train)

    output_dir = Path(cfg["artifact_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_logreg.joblib"
    metadata_path = output_dir / "baseline_metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "archive": cfg["archive"],
        "method": "chronological_holdout",
        "split_cutoff": cutoff,
        "train_time_max": float(training[cfg["time_column"]].max()),
        "validation_time_min": float(validation[cfg["time_column"]].min()),
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_positive_count": int(y_train.sum()),
        "train_prior": float(y_train.mean()),
        "features": features,
        "config": cfg,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print(f"Обучено на {len(training):,} строках, validation: {len(validation):,}")
    print(f"Граница времени: {cutoff:.6f}; доля fraud на train: {y_train.mean():.5f}")
    print(f"Модель: {model_path}; метаданные: {metadata_path}")


if __name__ == "__main__":
    main()
