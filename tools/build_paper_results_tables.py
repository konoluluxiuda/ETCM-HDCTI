#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.validate_hctx_ablation_configs import parse_config


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
        expected_inductive = {'HerbOnly': 'False', 'SDIS': 'True'}
        if config.get('context.interaction') != 'True':
            raise ValueError('Cold-start rows must enable context interaction.')
        if config.get('context.herb_protein') != 'True':
            raise ValueError('Cold-start rows must enable Hctx-P.')
        if config.get('inductive.context') != expected_inductive[variant]:
            raise ValueError('Unexpected SDIS switch in %s.' % config_path)
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


def collect_cold_start(manifest):
    section = manifest['compound_cold_start']
    fixed_path = check_file(section['fixed_results'], section['fixed_sha256'])
    calibrated_path = check_file(
        section['calibrated_results'], section['calibrated_sha256']
    )
    fixed_rows = read_tsv(fixed_path)
    calibrated_rows = read_tsv(calibrated_path)
    fixed = []
    calibrated = []
    for method in section['methods']:
        for dataset in manifest['datasets']:
            row = unique_row(fixed_rows, dataset, method['variant'])
            config_path = validate_config(
                row, 'compound_cold_start', method['variant']
            )
            fixed_record = dict(
                protocol='compound_cold_start', dataset=dataset,
                method=method['method'], variant=method['variant'],
                source=str(fixed_path), config=str(config_path),
                **metric_record(row)
            )
            fixed.append(fixed_record)

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


def build_markdown(manifest, random_rows, cold_rows, calibrated_rows):
    datasets = manifest['datasets']
    random_methods = [item['method'] for item in manifest['random_edge']['sources']]
    cold_methods = [
        item['method'] for item in manifest['compound_cold_start']['methods']
    ]
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
    lines.extend([''])
    lines.extend(delta_table(
        '2. 随机边 Hctx-P 直接消融', random_rows,
        random_methods[0], random_methods[1], datasets,
    ))
    lines.extend([''])
    lines.extend(delta_table(
        '3. 随机边 CHCR 增量', random_rows,
        random_methods[1], random_methods[2], datasets,
    ))
    lines.extend([''])
    lines.extend(metric_table(
        '4. Compound cold-start 五折（固定阈值 0.5）', cold_rows,
        cold_methods, datasets,
    ))
    lines.extend([''])
    lines.extend(delta_table(
        '5. Compound cold-start SDIS 增量（固定阈值 0.5）', cold_rows,
        cold_methods[0], cold_methods[1], datasets,
    ))
    lines.extend([''])
    lines.extend(metric_table(
        '6. Compound cold-start（inner-validation 阈值）',
        calibrated_rows, cold_methods, datasets, calibrated=True,
    ))
    lines.extend([''])
    lines.extend(delta_table(
        '7. Compound cold-start SDIS 校准指标增量', calibrated_rows,
        cold_methods[0], cold_methods[1], datasets,
    ))
    lines.extend([
        '',
        '## 8. 解释边界',
        '',
        '- 随机边主配置为 `Hctx-P + CHCR`；CHCR 不进入 cold-start 主配置。',
        '- Cold-start 主配置为 `Hctx-P + SDIS`；AUC/AUPR 与阈值无关。',
        '- Cold-start 固定 `0.5` 阈值与 inner-validation 阈值必须同时报告。',
        '- 四库统一无稠密注意力的 cold-start NoContext 完整五折尚不存在，'
        '因此本表不使用旧 attention 口径或单折 Pilot 填充该行。',
        '- TCM-Suite 上 Hctx-P 相对 Strict-HDCTI AUPR 轻微下降，不能声称'
        ' Hctx-P 在四库全部提高。',
        '',
        '## 9. 冻结来源',
        '',
    ])
    seen = []
    for row in random_rows + cold_rows + calibrated_rows:
        if row['source'] not in seen:
            seen.append(row['source'])
    lines.extend('- `%s`' % path for path in seen)
    lines.append('')
    return '\n'.join(lines)


def generate(manifest_path, output_path):
    manifest_path = repository_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported paper results manifest schema.')
    if len(manifest.get('datasets') or []) != 4:
        raise ValueError('Paper results manifest must freeze four datasets.')
    random_rows = collect_random(manifest)
    cold_rows, calibrated_rows = collect_cold_start(manifest)
    output_path = repository_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_markdown(manifest, random_rows, cold_rows, calibrated_rows),
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
