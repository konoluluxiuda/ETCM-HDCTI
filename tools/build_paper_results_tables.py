#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.validate_hctx_ablation_configs import parse_config
from tools.validate_cold_start_external_baselines import (
    validate_manifest as validate_cold_external_manifest,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / 'configs' / 'paper_results_manifest.json'
DEFAULT_OUTPUT = REPOSITORY_ROOT / 'docs' / 'FINAL_RESULTS_TABLES.md'
METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
NUMBER_PATTERN = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    r'(?:\((?:±|\+?-)([+-]?(?:\d+(?:\.\d*)?|\.\d+))\))?\s*$'
)


def repository_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def parse_summary(value):
    match = NUMBER_PATTERN.match(value or '')
    if not match:
        raise ValueError('Invalid metric summary: %r.' % value)
    return float(match.group(1)), (
        float(match.group(2)) if match.group(2) is not None else None
    )


def format_summary(mean, std):
    if std is None:
        return '%.6f' % mean
    return '%.6f (±%.6f)' % (mean, std)


def format_signed_summary(mean, std):
    if std is None:
        return '%+.6f' % mean
    return '%+.6f (±%.6f)' % (mean, std)


def canonical_dataset_name(name):
    aliases = {
        'ETCM2.0-mention10': 'ETCM2.0 mention10',
    }
    return aliases.get(name, name)


def check_file(path, expected_hash):
    path = repository_path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing frozen result source: %s' % path)
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(
            'Frozen result hash mismatch for %s: %s != %s'
            % (path, actual, expected_hash)
        )
    return path


def unique_row(rows, dataset, variant):
    matches = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('variant') == variant
        and row.get('status', 'OK') == 'OK'
    ]
    if len(matches) != 1:
        raise ValueError(
            'Expected one successful %s row for %s, found %d.'
            % (variant, dataset, len(matches))
        )
    return matches[0]


def unique_method_row(rows, dataset, method):
    matches = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('method') == method
        and row.get('status', 'OK') == 'OK'
    ]
    if len(matches) != 1:
        raise ValueError(
            'Expected one successful %s row for %s, found %d.'
            % (method, dataset, len(matches))
        )
    return matches[0]


def unique_dataset_row(rows, dataset):
    matches = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('status', 'OK') == 'OK'
    ]
    if len(matches) != 1:
        raise ValueError(
            'Expected one successful row for %s, found %d.'
            % (dataset, len(matches))
        )
    return matches[0]


def validate_config(row, strategy, variant):
    config_path = repository_path(row['config'])
    if not config_path.is_file():
        raise FileNotFoundError('Missing result config: %s' % config_path)
    if sha256_file(config_path) != row['config_sha256']:
        raise ValueError('Config hash mismatch for %s.' % config_path)
    config = parse_config(config_path)
    required = {
        'experiment.protocol': 'strict',
        'split.strategy': strategy,
        'split.reuse': 'True',
        'evaluation.setup': '-cv 5',
        'evaluation.outer.test': 'True',
        'early.stopping': 'True',
        'pair.decoder': 'dot',
        'attention.max.nodes': '0',
        'random.seed': '2026',
        'split.seed': '2026',
        'validation.seed': '102026',
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                '%s requires %s=%s.' % (config_path, key, expected)
            )
    if strategy == 'pair_stratified':
        expected_switches = {
            'NoContext': ('False', 'False', 'False'),
            'Hctx-P': ('True', 'True', 'False'),
            'Hctx-P+CHCR': ('True', 'True', 'True'),
        }
        actual = (
            config.get('context.interaction'),
            config.get('context.herb_protein'),
            config.get('counterfactual.context'),
        )
        if actual != expected_switches[variant]:
            raise ValueError('Unexpected random-edge switches in %s.' % config_path)
    else:
        expected_switches = {
            'NoContext': ('False', 'False', 'False'),
            'HerbOnly': ('True', 'True', 'False'),
            'SDIS': ('True', 'True', 'True'),
        }
        actual = (
            config.get('context.interaction'),
            config.get('context.herb_protein'),
            config.get('inductive.context'),
        )
        if actual != expected_switches[variant]:
            raise ValueError('Unexpected SDIS switch in %s.' % config_path)
    return config_path


def validate_external_config(row, required):
    config_path = repository_path(row['config'])
    if not config_path.is_file():
        raise FileNotFoundError('Missing result config: %s' % config_path)
    if sha256_file(config_path) != row['config_sha256']:
        raise ValueError('Config hash mismatch for %s.' % config_path)
    config = parse_config(config_path)
    common = {
        'experiment.protocol': 'strict',
        'split.strategy': 'pair_stratified',
        'split.reuse': 'True',
        'evaluation.setup': '-cv 5',
        'evaluation.outer.test': 'True',
        'early.stopping': 'True',
        'pair.decoder': 'dot',
        'random.seed': '2026',
        'split.seed': '2026',
        'validation.seed': '102026',
        'num.factors': '64',
        'num.max.epoch': '50',
        'batch_size': '2000',
    }
    for key, expected in dict(common, **required).items():
        if config.get(key) != expected:
            raise ValueError(
                '%s requires %s=%s.' % (config_path, key, expected)
            )
    if 'evaluation.fold.limit' in config:
        raise ValueError(
            'External full config must not limit folds: %s.' % config_path
        )
    return config_path


def metric_record(row):
    record = {}
    for metric in METRICS:
        record[metric], record[metric + '_std'] = parse_summary(row[metric])
    return record


def collect_random(manifest):
    collected = []
    datasets = manifest['datasets']
    for source in manifest['random_edge']['sources']:
        path = check_file(source['results'], source['sha256'])
        rows = read_tsv(path)
        for dataset in datasets:
            row = unique_row(rows, dataset, source['variant'])
            config_path = validate_config(
                row, 'pair_stratified', source['variant']
            )
            collected.append(dict(
                protocol='random_edge', dataset=dataset,
                method=source['method'], variant=source['variant'],
                source=str(path), config=str(config_path),
                **metric_record(row)
            ))
    return collected


def collect_external(manifest):
    section = manifest['external_same_input']
    source_cache = {}
    collected = []
    for method in section['methods']:
        results_path = method.get('results', section['results'])
        results_hash = method.get('sha256', section['sha256'])
        source_key = (results_path, results_hash)
        if source_key not in source_cache:
            path = check_file(results_path, results_hash)
            source_cache[source_key] = (path, read_tsv(path))
        path, rows = source_cache[source_key]
        for dataset in manifest['datasets']:
            if rows and 'method' in rows[0]:
                row = unique_method_row(rows, dataset, method['method'])
            else:
                row = unique_dataset_row(rows, dataset)
            config_path = validate_external_config(
                row,
                method['required'],
            )
            collected.append(dict(
                protocol='external_same_input',
                dataset=dataset,
                method=method['method'],
                source=str(path),
                config=str(config_path),
                **metric_record(row)
            ))
    return collected


def collect_cold_start(manifest):
    section = manifest['compound_cold_start']
    fixed_path = check_file(section['fixed_results'], section['fixed_sha256'])
    no_context_path = check_file(
        section['no_context_results'],
        section['no_context_sha256'],
    )
    calibrated_path = check_file(
        section['calibrated_results'], section['calibrated_sha256']
    )
    fixed_rows = read_tsv(fixed_path)
    no_context_rows = read_tsv(no_context_path)
    calibrated_rows = read_tsv(calibrated_path)
    fixed = []
    calibrated = []
    for method in section['methods']:
        method_path = (
            no_context_path
            if method['variant'] == 'NoContext'
            else fixed_path
        )
        method_rows = (
            no_context_rows
            if method['variant'] == 'NoContext'
            else fixed_rows
        )
        for dataset in manifest['datasets']:
            row = unique_row(method_rows, dataset, method['variant'])
            config_path = validate_config(
                row, 'compound_cold_start', method['variant']
            )
            fixed_record = dict(
                protocol='compound_cold_start', dataset=dataset,
                method=method['method'], variant=method['variant'],
                source=str(method_path), config=str(config_path),
                **metric_record(row)
            )
            fixed.append(fixed_record)

            if not method.get('calibrated', True):
                continue
            calibration = unique_row(
                calibrated_rows, dataset, method['variant']
            )
            if repository_path(calibration['config']) != config_path:
                raise ValueError('Calibration config does not match training row.')
            for metric in ('AUC', 'AUPR'):
                calibrated_value = float(calibration['fixed_' + metric.lower()])
                if abs(calibrated_value - fixed_record[metric]) > 1e-6:
                    raise ValueError(
                        '%s changed during calibration for %s %s.'
                        % (metric, dataset, method['variant'])
                    )
            calibrated.append({
                'protocol': 'compound_cold_start_calibrated',
                'dataset': dataset,
                'method': method['method'],
                'variant': method['variant'],
                'source': str(calibrated_path),
                'config': str(config_path),
                'AUC': float(calibration['calibrated_auc']),
                'AUC_std': float(calibration['calibrated_auc_std']),
                'AUPR': float(calibration['calibrated_aupr']),
                'AUPR_std': float(calibration['calibrated_aupr_std']),
                'Recall': float(calibration['calibrated_recall']),
                'Recall_std': float(calibration['calibrated_recall_std']),
                'Precision': float(calibration['calibrated_precision']),
                'Precision_std': float(
                    calibration['calibrated_precision_std']
                ),
                'F1-score': float(calibration['calibrated_f1_score']),
                'F1-score_std': float(
                    calibration['calibrated_f1_score_std']
                ),
                'threshold': float(calibration['threshold_mean']),
                'threshold_std': float(calibration['threshold_std']),
            })
    return fixed, calibrated


def collect_support_state_five_unit(manifest):
    section = manifest.get('support_state_five_unit')
    if not section:
        return None
    path = check_file(section['summary'], section['sha256'])
    summary = json.loads(path.read_text(encoding='utf-8'))
    if summary.get('protocol') != (
            'frozen_base_hctx_router_five_unit_descriptive_v1'):
        raise ValueError('Unexpected support-state summary protocol.')
    if summary.get('analysis_role') != section['role']:
        raise ValueError('Support-state summary role changed.')
    if summary.get('dataset_count') != 4 or summary.get('unit_count') != 20:
        raise ValueError('Support-state summary must contain 4 x 5 units.')
    if summary.get('positive_macro_units') != 20:
        raise ValueError('Support-state summary no longer has 20/20 positives.')
    for row in summary['datasets']:
        if row['units'] != 5 or row['positive_macro_units'] != 5:
            raise ValueError('Support-state dataset is incomplete.')
        if row['delta']['warm_cold']['mean'] != 0.0:
            raise ValueError('Support-state WC preservation changed.')
        if row['delta']['cold_cold']['mean'] != 0.0:
            raise ValueError('Support-state CC preservation changed.')
    summary['_source'] = str(path)
    return summary


def collect_schpt_full(manifest):
    section = manifest.get('schpt_full')
    if not section:
        return None
    path = check_file(section['summary'], section['sha256'])
    summary = json.loads(path.read_text(encoding='utf-8'))
    if summary.get('protocol') != section['protocol']:
        raise ValueError('Unexpected SCHPT summary protocol.')
    if summary.get('gate') != 'PASS':
        raise ValueError('SCHPT full confirmation Gate is not PASS.')
    if len(summary.get('rows') or []) != 4:
        raise ValueError('SCHPT summary must contain four datasets.')
    if summary.get('fold_count') != 20:
        raise ValueError('SCHPT summary must contain twenty paired folds.')
    if not all(summary.get('checks', {}).values()):
        raise ValueError('At least one SCHPT frozen Gate check failed.')
    summary['_source'] = str(path)
    return summary


def collect_cold_external(manifest):
    section = manifest.get('compound_cold_start_external')
    if not section:
        return [], None
    experiment_manifest_path = check_file(
        section['experiment_manifest'], section['experiment_manifest_sha256']
    )
    experiment_manifest, validated_jobs = validate_cold_external_manifest(
        experiment_manifest_path
    )
    results_path = check_file(section['results'], section['sha256'])
    rows = read_tsv(results_path)
    result_dataset_names = section.get('dataset_names', {})
    job_index = {
        (job['dataset'], job['method']): job for job in validated_jobs
    }
    collected = []
    for dataset in manifest['datasets']:
        result_dataset = result_dataset_names.get(dataset, dataset)
        for method in experiment_manifest['methods']:
            row = unique_method_row(rows, result_dataset, method['name'])
            frozen_job = job_index[(result_dataset, method['name'])]
            if row['config'] != frozen_job['config']:
                raise ValueError(
                    'Cold-start result config differs from the frozen job.'
                )
            if row['config_sha256'] != frozen_job['sha256']:
                raise ValueError(
                    'Cold-start result config hash differs from the manifest.'
                )
            collected.append(dict(
                protocol='compound_cold_start_external',
                dataset=dataset,
                method=method['name'],
                source=str(results_path),
                config=str(repository_path(row['config'])),
                **metric_record(row)
            ))
    metadata = {
        'methods': [method['name'] for method in experiment_manifest['methods']],
        'reference_method': section['reference_method'],
        'experiment_manifest': str(experiment_manifest_path),
    }
    if metadata['reference_method'] not in metadata['methods']:
        raise ValueError('Cold-start reference method is not in the manifest.')
    return collected, metadata


def markdown_metric(row, metric):
    return format_summary(row[metric], row[metric + '_std'])


def metric_table(title, records, methods, datasets, calibrated=False):
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | 方法 | AUC | AUPR | Recall | Precision | F1-score%s |'
        % (' | 阈值' if calibrated else ''),
        '|---|---|---:|---:|---:|---:|---:|%s'
        % ('---:|' if calibrated else ''),
    ]
    index = {(row['dataset'], row['method']): row for row in records}
    for dataset in datasets:
        for method in methods:
            row = index[(dataset, method)]
            values = [markdown_metric(row, metric) for metric in METRICS]
            if calibrated:
                values.append(format_summary(
                    row['threshold'], row['threshold_std']
                ))
            lines.append('| %s | %s | %s |' % (
                dataset, method, ' | '.join(values)
            ))
    return lines


def delta_table(title, records, baseline, candidate, datasets):
    index = {(row['dataset'], row['method']): row for row in records}
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | AUC delta | AUPR delta | Recall delta | '
        'Precision delta | F1 delta |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    macro = defaultdict(float)
    for dataset in datasets:
        base = index[(dataset, baseline)]
        current = index[(dataset, candidate)]
        deltas = {metric: current[metric] - base[metric] for metric in METRICS}
        for metric, value in deltas.items():
            macro[metric] += value / len(datasets)
        lines.append('| %s | %s |' % (
            dataset,
            ' | '.join('%+.6f' % deltas[metric] for metric in METRICS),
        ))
    lines.append('| **Macro** | %s |' % ' | '.join(
        '**%+.6f**' % macro[metric] for metric in METRICS
    ))
    return lines


def support_state_table(title, summary):
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | Units | NoContext Macro-AUPR | V3 Macro-AUPR | '
        'Delta | Positive | WW Delta | CW Delta | WC Delta | CC Delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['datasets']:
        lines.append(
            '| %s | %d | %s | %s | %s | %d/%d | %+.6f | %+.6f | '
            '%+.6f | %+.6f |' % (
                row['display_name'], row['units'],
                format_summary(
                    row['baseline_macro_aupr']['mean'],
                    row['baseline_macro_aupr']['std'],
                ),
                format_summary(
                    row['candidate_macro_aupr']['mean'],
                    row['candidate_macro_aupr']['std'],
                ),
                format_signed_summary(
                    row['delta']['macro']['mean'],
                    row['delta']['macro']['std'],
                ),
                row['positive_macro_units'], row['units'],
                row['delta']['warm_warm']['mean'],
                row['delta']['cold_warm']['mean'],
                row['delta']['warm_cold']['mean'],
                row['delta']['cold_cold']['mean'],
            )
        )
    lines.extend([
        '',
        '20 单元总体 Macro-AUPR 增量为 `%s`。该表包含历史 `c0p0`，'
        '仅作描述性汇总；预注册确认 Gate 只使用 `c1p1-c4p4` 的 16 个新单元。'
        % format_signed_summary(
            summary['overall_macro_aupr_delta']['mean'],
            summary['overall_macro_aupr_delta']['std'],
        ),
    ])
    return lines


def schpt_table(title, summary):
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | Baseline AUC | Ours-full AUC | Baseline AUPR | '
        'Ours-full AUPR | AUPR 增量 | 正向 folds |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['rows']:
        lines.append(
            '| %s | %.6f | %.6f | %.6f | %.6f | %+.6f | %d/5 |'
            % (
                row['dataset'],
                row['baseline_metrics']['AUC'],
                row['candidate_metrics']['AUC'],
                row['baseline_metrics']['AUPR'],
                row['candidate_metrics']['AUPR'],
                row['outer_aupr_delta'],
                row['positive_fold_count'],
            )
        )
    lines.extend([
        '| **Macro delta** | - | - | - | - | **%+.6f** | **%d/%d** |'
        % (
            summary['mean_outer_aupr_delta'],
            summary['positive_fold_count'],
            summary['fold_count'],
        ),
        '',
        'Ours-full 使用 `Hctx-P + SDIS + SCHPT`。Baseline 使用相同 seed、'
        'compound cold-start folds、Hctx-P、SDIS 和 inner-validation AUPR '
        '早停；唯一方法差异是 SCHPT 以支持度校准 LOCO 药材靶点原型替换 '
        'compound C-P PageRank。预注册 Gate 全部通过。TCM-Suite 仅 `2/5` '
        'folds 为正，因此该结果支持跨库总体增益，不支持逐 fold 单调提升主张。',
    ])
    return lines


def cold_external_aupr_table(
        title, external_rows, external_metadata, schpt_summary, datasets):
    methods = external_metadata['methods']
    external_index = {
        (row['dataset'], row['method']): row for row in external_rows
    }
    ours_index = {
        canonical_dataset_name(row['dataset']): row
        for row in schpt_summary['rows']
    }
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | %s | Ours-full | 排名 |' % ' | '.join(methods),
        '|---|%s---:|---:|' % ('---:|' * len(methods)),
    ]
    method_macro = defaultdict(float)
    ours_macro = 0.0
    for dataset in datasets:
        method_values = [
            external_index[(dataset, method)]['AUPR'] for method in methods
        ]
        ours_row = ours_index[dataset]
        ours = ours_row['candidate_metrics']['AUPR']
        ours_std = statistics.pstdev(ours_row['candidate_fold_aupr'])
        all_values = method_values + [ours]
        best = max(all_values)
        rank = 1 + sum(value > ours for value in method_values)
        cells = []
        for method, value in zip(methods, method_values):
            row = external_index[(dataset, method)]
            rendered = format_summary(value, row['AUPR_std'])
            if value == best:
                rendered = '**%s**' % rendered
            cells.append(rendered)
            method_macro[method] += value / len(datasets)
        ours_rendered = format_summary(ours, ours_std)
        if ours == best:
            ours_rendered = '**%s**' % ours_rendered
        ours_macro += ours / len(datasets)
        lines.append('| %s | %s | %s | %d/%d |' % (
            dataset, ' | '.join(cells), ours_rendered, rank,
            len(methods) + 1,
        ))
    macro_values = [method_macro[method] for method in methods]
    macro_best = max(macro_values + [ours_macro])
    macro_cells = [
        ('**%.6f**' % value) if value == macro_best else '%.6f' % value
        for value in macro_values
    ]
    ours_macro_rendered = (
        '**%.6f**' % ours_macro
        if ours_macro == macro_best else '%.6f' % ours_macro
    )
    macro_rank = 1 + sum(value > ours_macro for value in macro_values)
    lines.append('| **Macro** | %s | %s | **%d/%d** |' % (
        ' | '.join(macro_cells), ours_macro_rendered, macro_rank,
        len(methods) + 1,
    ))
    return lines


def cold_external_delta_table(
        title, external_rows, external_metadata, schpt_summary, datasets):
    methods = external_metadata['methods']
    reference = external_metadata['reference_method']
    external_index = {
        (row['dataset'], row['method']): row for row in external_rows
    }
    ours_index = {
        canonical_dataset_name(row['dataset']): row
        for row in schpt_summary['rows']
    }
    lines = [
        '## %s' % title,
        '',
        '| 数据集 | Ours-full AUPR | %s AUPR | 相对统一基线 | '
        '每库最佳外部基线 | 相对每库最佳 |' % reference,
        '|---|---:|---:|---:|---:|---:|',
    ]
    macro_ours = 0.0
    macro_reference = 0.0
    macro_best = 0.0
    for dataset in datasets:
        ours = ours_index[dataset]['candidate_metrics']['AUPR']
        reference_value = external_index[(dataset, reference)]['AUPR']
        best_method = max(
            methods, key=lambda method: external_index[(dataset, method)]['AUPR']
        )
        best_value = external_index[(dataset, best_method)]['AUPR']
        macro_ours += ours / len(datasets)
        macro_reference += reference_value / len(datasets)
        macro_best += best_value / len(datasets)
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %s (%.6f) | %+.6f |'
            % (
                dataset, ours, reference_value, ours - reference_value,
                best_method, best_value, ours - best_value,
            )
        )
    lines.extend([
        '| **Macro** | **%.6f** | **%.6f** | **%+.6f** | **%.6f** | '
        '**%+.6f** |' % (
            macro_ours, macro_reference, macro_ours - macro_reference,
            macro_best, macro_ours - macro_best,
        ),
        '',
        'Ours-full 的四库 macro AUPR 高于统一的最强单一外部基线 `%s`，'
        '但逐库只在 TCMSP 和 ETCM2.0-mention10 排名第一；TCM-Suite 与 '
        'SymMap2.0 分别落后各自最佳外部基线。每库最佳列仅作描述性上界，'
        '不能当作一个预先固定的单一比较方法。' % reference,
    ])
    return lines


def build_markdown(
        manifest, random_rows, cold_rows, calibrated_rows,
        external_rows=None, support_state_summary=None, schpt_summary=None,
        cold_external_rows=None, cold_external_metadata=None):
    datasets = manifest['datasets']
    random_methods = [item['method'] for item in manifest['random_edge']['sources']]
    cold_methods = [
        item['method'] for item in manifest['compound_cold_start']['methods']
    ]
    calibrated_methods = [
        item['method']
        for item in manifest['compound_cold_start']['methods']
        if item.get('calibrated', True)
    ]
    external_rows = external_rows or []
    cold_external_rows = cold_external_rows or []
    external_methods = [
        item['method']
        for item in manifest.get('external_same_input', {}).get('methods', [])
    ]
    external_method_count = len(external_methods)
    external_reference = manifest.get('external_same_input', {}).get(
        'reference_method',
        external_methods[-1] if external_methods else None,
    )
    lines = [
        '# 最终统一实验结果表',
        '',
        '本文件由 `tools/build_paper_results_tables.py` 从冻结的机器可读结果生成。',
        '所有来源文件、配置文件和协议开关均在生成前校验。',
        '',
        '> 注意：五折标准差表示 fold 差异，不等同于多随机初始化标准差。',
        '',
    ]
    lines.extend(metric_table(
        '1. 普通 Strict 随机边五折', random_rows,
        random_methods, datasets,
    ))
    section = 2
    if external_rows:
        final_method = random_methods[-1]
        final_rows = [
            row for row in random_rows if row['method'] == final_method
        ]
        comparison_rows = external_rows + final_rows
        comparison_methods = external_methods + [final_method]
        lines.extend([''])
        lines.extend(metric_table(
            '%d. 普通随机边同输入方法比较' % section,
            comparison_rows,
            comparison_methods,
            datasets,
        ))
        section += 1
        lines.extend([''])
        lines.extend(delta_table(
            '%d. 最终随机边模型相对 %s' % (
                section,
                external_reference,
            ),
            comparison_rows,
            external_reference,
            final_method,
            datasets,
        ))
        section += 1
    lines.extend([''])
    lines.extend(delta_table(
        '%d. 随机边 Hctx-P 直接消融' % section, random_rows,
        random_methods[0], random_methods[1], datasets,
    ))
    section += 1
    lines.extend([''])
    lines.extend(delta_table(
        '%d. 随机边 CHCR 增量' % section, random_rows,
        random_methods[1], random_methods[2], datasets,
    ))
    section += 1
    lines.extend([''])
    lines.extend(metric_table(
        '%d. Compound cold-start 五折（固定阈值 0.5）' % section,
        cold_rows,
        cold_methods, datasets,
    ))
    section += 1
    lines.extend([''])
    lines.extend(delta_table(
        '%d. Compound cold-start Hctx-P 直接消融' % section,
        cold_rows,
        cold_methods[0], cold_methods[1], datasets,
    ))
    section += 1
    lines.extend([''])
    lines.extend(delta_table(
        '%d. Compound cold-start SDIS 增量（固定阈值 0.5）' % section,
        cold_rows,
        cold_methods[1], cold_methods[2], datasets,
    ))
    section += 1
    lines.extend([''])
    lines.extend(metric_table(
        '%d. Compound cold-start（inner-validation 阈值）' % section,
        calibrated_rows, calibrated_methods, datasets, calibrated=True,
    ))
    section += 1
    lines.extend([''])
    lines.extend(delta_table(
        '%d. Compound cold-start SDIS 校准指标增量' % section,
        calibrated_rows,
        calibrated_methods[0], calibrated_methods[1], datasets,
    ))
    section += 1
    if support_state_summary is not None:
        lines.extend([''])
        lines.extend(support_state_table(
            '%d. 支持状态五单元描述性结果' % section,
            support_state_summary,
        ))
        section += 1
    if schpt_summary is not None:
        lines.extend([''])
        lines.extend(schpt_table(
            '%d. 最终 Ours-full compound cold-start 五折确认' % section,
            schpt_summary,
        ))
        section += 1
    if cold_external_rows and schpt_summary is not None:
        lines.extend([''])
        lines.extend(cold_external_aupr_table(
            '%d. Compound cold-start 外部基线 AUPR 比较' % section,
            cold_external_rows, cold_external_metadata,
            schpt_summary, datasets,
        ))
        section += 1
        lines.extend([''])
        lines.extend(cold_external_delta_table(
            '%d. Ours-full 相对冷启动外部基线' % section,
            cold_external_rows, cold_external_metadata,
            schpt_summary, datasets,
        ))
        section += 1
    lines.extend([
        '',
        '## %d. 解释边界' % section,
        '',
        '- 随机边主配置为 `Hctx-P + CHCR`；CHCR 不进入 cold-start 主配置。',
        '- %d 种外部方法是共享匿名拓扑输入和 BCE 监督的适配基线，不是'
        '原论文属性模型的原样复现。' % external_method_count,
        '- HGT-CTI 采用四库统一的每关系/目标节点 64 入邻居上限；其 ETCM'
        ' 结果不能解释为无采样完整 HGT 的性能。',
        '- 最终随机边模型相对 R-GCN-CTI 在 SymMap2.0 和 ETCM2.0 mention10'
        ' 取得更高 AUPR，在 TCMSP 基本持平，在 TCM-Suite 略低；不能声称'
        '四库全部最优。',
        '- Cold-start 最终主配置为 `Hctx-P + SDIS + SCHPT`；其中 SCHPT'
        ' 替换 compound C-P PageRank，AUC/AUPR 与阈值无关。',
        '- Cold-start 固定 `0.5` 阈值与 inner-validation 阈值必须同时报告。',
        '- Compound cold-start 下 Hctx-P 相对 NoContext 的四库 AUPR 增量'
        '均为正，macro 增量为 `+0.437826`；该结果只适用于具有 H-C 侧信息'
        '的 compound cold-start。',
        '- NoContext 未执行事后阈值校准；其固定 `0.5` 分类指标仅用于完整披露，'
        '不与已校准的 Hctx-P/SDIS 分类指标混合比较。',
        '- 支持状态五单元表中的历史 `c0p0` 只用于描述性汇总；V3 的确认性'
        '结论来自预注册的 16 个新 outer units，不能混写为 20 单元预注册 Gate。',
        '- SCHPT 四库平均 AUPR 增量和 17/20 正向 folds 通过预注册 Gate；'
        'TCM-Suite 的 fold 异质性必须在讨论中披露。',
        '- 冷启动外部比较中，Ours-full 的 macro AUPR 排名第一，但只在 '
        'TCMSP 和 ETCM2.0 mention10 逐库排名第一；不得声称四库全部最优。',
        '',
        '## %d. 冻结来源' % (section + 1),
        '',
    ])
    seen = []
    for row in (
            random_rows + external_rows + cold_rows + calibrated_rows
            + cold_external_rows):
        if row['source'] not in seen:
            seen.append(row['source'])
    if (
            support_state_summary is not None
            and support_state_summary.get('_source')):
        seen.append(support_state_summary['_source'])
    if schpt_summary is not None and schpt_summary.get('_source'):
        seen.append(schpt_summary['_source'])
    lines.extend('- `%s`' % path for path in seen)
    lines.append('')
    return '\n'.join(lines)


def generate(manifest_path, output_path):
    manifest_path = repository_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2:
        raise ValueError('Unsupported paper results manifest schema.')
    if len(manifest.get('datasets') or []) != 4:
        raise ValueError('Paper results manifest must freeze four datasets.')
    random_rows = collect_random(manifest)
    external_rows = collect_external(manifest)
    cold_rows, calibrated_rows = collect_cold_start(manifest)
    support_state_summary = collect_support_state_five_unit(manifest)
    schpt_summary = collect_schpt_full(manifest)
    cold_external_rows, cold_external_metadata = collect_cold_external(manifest)
    output_path = repository_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_markdown(
            manifest,
            random_rows,
            cold_rows,
            calibrated_rows,
            external_rows=external_rows,
            support_state_summary=support_state_summary,
            schpt_summary=schpt_summary,
            cold_external_rows=cold_external_rows,
            cold_external_metadata=cold_external_metadata,
        ),
        encoding='utf-8',
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Build publication tables from frozen HDCTI results.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = generate(args.manifest, args.output)
    print('Final paper results written to: %s' % output)


if __name__ == '__main__':
    main()
