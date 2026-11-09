"""Контролируемая проверка вклада признаков identity на временных фолдах.

Используется исключительно train исходного 80/20 разделения. Основная
отложенная validation не участвует в оценке или выборе набора признаков.

Запуск: python -m src.identity_ablation --config configs/identity_ablation.json
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .utils import load_checked_data, load_config, make_model, prepare_holdout


def read_identity_selected(archive_path, identity_fields):
    """Читает ограниченный набор столбцов identity из локального ZIP."""
    with zipfile.ZipFile(archive_path, 'r') as archive:
        names = [
            item.filename for item in archive.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename).name == 'train_identity.csv'
        ]
        if len(names) != 1:
            raise ValueError('Необходимо ровно одно вхождение train_identity.csv в архиве.')
        with archive.open(names[0]) as stream:
            identity = pd.read_csv(stream, usecols=['TransactionID', *identity_fields], low_memory=False)
    if identity['TransactionID'].isna().any() or not identity['TransactionID'].is_unique:
        raise ValueError('Идентификатор TransactionID в identity должен быть уникальным и непустым.')
    return identity


def prepare_identity_train(baseline_config, experiment_config):
    """Загружает train исходного протокола без обращения к целевым меткам holdout."""
    transactions = load_checked_data(baseline_config)
    train, _, _ = prepare_holdout(transactions, baseline_config)
    identity = read_identity_selected(
        baseline_config['archive'], experiment_config['identity_features']
    )
    joined = train.merge(
        identity, on='TransactionID', how='left', validate='one_to_one',
        indicator=True, sort=False
    )
    if len(joined) != len(train):
        raise AssertionError('Соединение identity изменило количество транзакций.')
    joined['has_identity'] = joined.pop('_merge').eq('both')
    return joined


def run_ablation(train_df, baseline_config, experiment_config):
    """Возвращает метрики на трёх общих последовательных внутренних фолдах."""
    extra = list(experiment_config['identity_features'])
    categorical_base = list(baseline_config['categorical_features'])
    numeric_base = list(baseline_config['numeric_features'])
    target = baseline_config['target_column']
    time_col = baseline_config['time_column']
    required = set([target, time_col, 'has_identity', *numeric_base, *categorical_base, *extra])
    missing = required.difference(train_df.columns)
    if missing:
        raise KeyError(f'Отсутствуют столбцы для эксперимента: {sorted(missing)}')

    q = np.asarray(experiment_config['fold_quantiles'], dtype=float)
    if len(q) != 3 or not np.array_equal(q, np.array([0.4, 0.6, 0.8])):
        raise ValueError('Эксперимент зафиксирован для квантилей [0.4, 0.6, 0.8].')
    t = train_df[time_col]
    cuts = t.quantile(q).to_numpy(dtype=float)
    if not np.all(np.diff(cuts) > 0):
        raise ValueError('Временные квантильные границы совпадают.')

    data = train_df[list(required)].copy()
    data['identity_status'] = np.where(data['has_identity'].astype(bool), 'present', 'absent')
    # Поля identity интерпретируются как категории, даже когда содержат цифры.
    for col in extra:
        s = data[col].astype('object')
        data[col] = s.where(s.isna(), s.astype(str))

    variants = {
        'A_transaction_only': categorical_base,
        'B_plus_identity_status': categorical_base + ['identity_status'],
        'C_plus_identity_fields': categorical_base + ['identity_status', *extra],
    }
    rows = []
    intervals = [(cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], np.inf)]
    for fold, (train_end, valid_end) in enumerate(intervals, start=1):
        fit = data.loc[t < train_end]
        check = data.loc[(t >= train_end) & (t < valid_end)]
        assert not fit.empty and not check.empty
        assert fit[time_col].max() < check[time_col].min()
        y_fit = fit[target].astype(int).to_numpy()
        y_check = check[target].astype(int).to_numpy()
        if np.unique(y_fit).size != 2 or np.unique(y_check).size != 2:
            raise ValueError(f'Fold {fold}: требуются оба класса для ROC-AUC.')

        for variant, categorical in variants.items():
            cfg = {
                **baseline_config,
                'numeric_features': numeric_base,
                'categorical_features': categorical,
            }
            features = numeric_base + categorical
            model = make_model(cfg)  # Внутри: те же preprocessing и параметры LogisticRegression.
            model.fit(fit[features], y_fit)
            p = model.predict_proba(check[features])[:, 1]
            if not np.isfinite(p).all():
                raise ValueError(f'Fold {fold}, {variant}: предсказания некорректны.')
            k = max(1, int(np.ceil(float(experiment_config['top_fraction']) * len(check))))
            top = np.argsort(-p, kind='mergesort')[:k]
            tp = int(y_check[top].sum())
            rows.append({
                'fold': fold,
                'variant': variant,
                'train_rows': len(fit),
                'validation_rows': len(check),
                'validation_fraud_count': int(y_check.sum()),
                'validation_fraud_rate': float(y_check.mean()),
                'roc_auc': float(roc_auc_score(y_check, p)),
                'average_precision': float(average_precision_score(y_check, p)),
                'precision_at_1pct': tp / k,
                'recall_at_1pct': tp / int(y_check.sum()),
            })
            print(f'Fold {fold}, {variant}: ROC-AUC={rows[-1]["roc_auc"]:.5f}, '
                  f'AP={rows[-1]["average_precision"]:.5f}, '
                  f'Precision@1%={rows[-1]["precision_at_1pct"]:.4f}')

    folds = pd.DataFrame(rows)
    summaries = []
    for variant, grp in folds.groupby('variant', sort=False):
        summaries.append({
            'variant': variant,
            'roc_auc_mean': float(grp['roc_auc'].mean()),
            'roc_auc_std': float(grp['roc_auc'].std(ddof=1)),
            'ap_mean': float(grp['average_precision'].mean()),
            'ap_std': float(grp['average_precision'].std(ddof=1)),
            'precision_at_1pct_mean': float(grp['precision_at_1pct'].mean()),
            'recall_at_1pct_mean': float(grp['recall_at_1pct'].mean()),
        })
    summary = pd.DataFrame(summaries)
    reference = summary.loc[summary['variant'] == 'A_transaction_only'].iloc[0]
    summary['delta_roc_auc_vs_A'] = summary['roc_auc_mean'] - reference['roc_auc_mean']
    summary['delta_ap_vs_A'] = summary['ap_mean'] - reference['ap_mean']
    return folds, summary


def save_results(folds, summary, experiment_config):
    output = Path(experiment_config['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / 'identity_ablation_folds.csv', index=False)
    summary.to_csv(output / 'identity_ablation_summary.csv', index=False)
    import matplotlib.pyplot as plt
    for metric, filename, ylabel in [
        ('roc_auc', 'identity_ablation_roc_auc.png', 'ROC-AUC'),
        ('average_precision', 'identity_ablation_ap.png', 'Average Precision'),
    ]:
        plt.figure(figsize=(8,4.5))
        for variant, group in folds.groupby('variant', sort=False):
            plt.plot(group['fold'], group[metric], marker='o', label=variant)
        plt.xticks([1,2,3]); plt.xlabel('Номер временного фолда'); plt.ylabel(ylabel)
        plt.title(f'Identity ablation: {ylabel} по временным фолдам')
        plt.grid(alpha=0.25); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(output/filename, dpi=150, bbox_inches='tight')
        plt.close()
    with (output / 'identity_ablation_protocol.json').open('w', encoding='utf-8') as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser(description='Временная ablation-проверка identity')
    parser.add_argument('--config', default='configs/identity_ablation.json')
    args = parser.parse_args()
    with Path(args.config).open(encoding='utf-8') as f:
        exp = json.load(f)
    baseline = load_config(exp['baseline_config'])
    training = prepare_identity_train(baseline, exp)
    folds, summary = run_ablation(training, baseline, exp)
    print('\nСводка:')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    out = save_results(folds, summary, exp)
    print(f'Результаты сохранены: {out.resolve()}')


if __name__ == '__main__':
    main()
