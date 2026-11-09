"""Загрузка данных и построение модели для локального baseline IEEE-CIS."""

from pathlib import Path, PurePosixPath
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    features = config["numeric_features"] + config["categorical_features"]
    if len(set(features)) != len(features):
        raise ValueError("В конфигурации обнаружены повторяющиеся признаки.")
    if not 0 < float(config["train_fraction"]) < 1:
        raise ValueError("train_fraction должен принадлежать интервалу (0, 1).")
    return config


def load_transaction_columns(archive_path, columns):
    """Загружает необходимые колонки CSV из ZIP без распаковки на диск."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Не найден архив: {archive_path.resolve()}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        candidates = [
            item.filename for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).name == "train_transaction.csv"
        ]
        if len(candidates) != 1:
            raise ValueError("Ожидается ровно один train_transaction.csv внутри ZIP.")
        with archive.open(candidates[0]) as stream:
            data = pd.read_csv(stream, usecols=columns, low_memory=False)
    if data.empty:
        raise ValueError("Таблица транзакций пуста.")
    return data


def prepare_holdout(data, config):
    """Определяет единую границу по времени, не используя целевую переменную."""
    time_field = config["time_column"]
    if data[time_field].isna().any():
        raise ValueError("В временной координате есть пропущенные значения.")
    cutoff = float(data[time_field].quantile(float(config["train_fraction"])))
    training = data.loc[data[time_field] < cutoff].copy()
    validation = data.loc[data[time_field] >= cutoff].copy()
    if training.empty or validation.empty:
        raise ValueError("Одна из частей хронологического разделения пуста.")
    if training[time_field].max() >= validation[time_field].min():
        raise ValueError("Нарушен хронологический порядок частей.")
    return training, validation, cutoff


def make_model(config):
    """В точности повторяет преобразования исходного ноутбука baseline_v1."""
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", one_hot),
    ])
    prep = ColumnTransformer(transformers=[
        ("numeric", numeric, config["numeric_features"]),
        ("categorical", categorical, config["categorical_features"]),
    ], remainder="drop")
    clf = LogisticRegression(**config["model"])
    return Pipeline(steps=[("preprocessing", prep), ("classifier", clf)])


def required_columns(config):
    return list(dict.fromkeys([
        "TransactionID", config["time_column"], config["target_column"],
        *config["numeric_features"], *config["categorical_features"]
    ]))


def load_checked_data(config):
    data = load_transaction_columns(config["archive"], required_columns(config))
    if data["TransactionID"].isna().any() or not data["TransactionID"].is_unique:
        raise ValueError("TransactionID должен быть уникальным и непустым.")
    y = config["target_column"]
    if data[y].isna().any() or not set(data[y].unique()).issubset({0, 1}):
        raise ValueError("Ожидается бинарный isFraud без пропусков.")
    xnum = config["numeric_features"]
    if not np.isfinite(data[xnum].to_numpy(dtype="float64")).all():
        # Пропуски допускаются при медианной импутации, бесконечности нет.
        if np.isinf(data[xnum].to_numpy(dtype="float64")).any():
            raise ValueError("Числовые признаки содержат бесконечности.")
    for f in xnum:
        if (data[f].dropna() < 0).any():
            raise ValueError(f"Для log1p недопустимы отрицательные значения: {f}")
    return data
