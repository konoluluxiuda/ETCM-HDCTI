#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_PATTERN = re.compile(
    r'^Validation-AUPR:\s*([0-9]+(?:\.[0-9]+)?)', re.MULTILINE
)
METADATA_PATTERN = re.compile(
    r'^Herb prototype metadata:\s*(.+)$', re.MULTILINE
)


def parse_validation_aupr(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    values = VALIDATION_PATTERN.findall(text)
    if not values:
        raise ValueError('No Validation-AUPR found in %s.' % path)
    return float(values[-1]), text


def summarize(baseline_log, candidate_log):
    baseline_aupr, _ = parse_validation_aupr(baseline_log)
    candidate_aupr, candidate_text = parse_validation_aupr(candidate_log)
    metadata_matches = METADATA_PATTERN.findall(candidate_text)
    if not metadata_matches:
        raise ValueError('No herb prototype metadata path found in candidate log.')
    metadata_path = Path(metadata_matches[-1].strip())
    if not metadata_path.is_absolute():
        metadata_path = Path.cwd() / metadata_path
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    delta = candidate_aupr - baseline_aupr
    coverage = float(metadata['validation']['evidence_coverage'])
    learned_scale = float(metadata['learned_scale'])
    checks = {
        'validation_aupr_delta_at_least_0.003': delta >= 0.003,
        'evidence_coverage_at_least_0.30': coverage >= 0.30,
        'learned_scale_nonzero': abs(learned_scale) > 1e-6,
    }
    return {
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'protocol': 'schpt_seed52026_inner_validation_gate_v1',
        'baseline_log': str(Path(baseline_log).resolve()),
        'candidate_log': str(Path(candidate_log).resolve()),
        'metadata_path': str(metadata_path.resolve()),
        'baseline_validation_aupr': baseline_aupr,
        'candidate_validation_aupr': candidate_aupr,
        'validation_aupr_delta': delta,
        'evidence_coverage': coverage,
        'mean_abs_residual': float(
            metadata['validation']['mean_abs_residual']
        ),
        'learned_scale': learned_scale,
        'checks': checks,
        'gate': 'PASS' if all(checks.values()) else 'NO-GO',
    }


def markdown(report):
    checks = report['checks']
    return '\n'.join([
        '# SCHPT Paired Pilot',
        '',
        '- Protocol: fresh seed 52026, fold-1 inner validation only',
        '- Outer-test parameter selection: disabled',
        '- Gate: `%s`' % report['gate'],
        '',
        '| Metric | Baseline | SCHPT | Delta / Value |',
        '|---|---:|---:|---:|',
        '| Validation AUPR | %.6f | %.6f | %+.6f |' % (
            report['baseline_validation_aupr'],
            report['candidate_validation_aupr'],
            report['validation_aupr_delta'],
        ),
        '| Prototype evidence coverage | - | - | %.4f |' % (
            report['evidence_coverage']
        ),
        '| Mean absolute residual | - | - | %.6f |' % (
            report['mean_abs_residual']
        ),
        '| Learned residual scale | - | - | %.6f |' % (
            report['learned_scale']
        ),
        '',
        '## Gate Checks',
        '',
        *[
            '- `%s`: `%s`' % (key, value)
            for key, value in checks.items()
        ],
        '',
        'A PASS permits a frozen four-dataset confirmation. A NO-GO stops '
        'SCHPT without prior, seed, or threshold search.',
        '',
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument('--candidate-log', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(args.baseline_log, args.candidate_log)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'summary.md').write_text(
        markdown(report), encoding='utf-8'
    )
    print('SCHPT Gate: %s' % report['gate'])
    print('Summary: %s' % (output_dir / 'summary.md').resolve())
    return 0 if report['gate'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
