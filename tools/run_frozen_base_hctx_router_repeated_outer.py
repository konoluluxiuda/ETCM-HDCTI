#!/usr/bin/env python3
"""Run the preregistered repeated frozen-router confirmation pipeline."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_four_state_checkpoint import (  # noqa: E402
    normalize_checkpoint,
    sha256_file,
)


DEFAULT_PREPARED_MANIFEST = (
    REPOSITORY_ROOT
    / 'configs'
    / 'frozen_base_hctx_router_repeated_units_manifest.json'
)
CHECKPOINT_PATTERN = re.compile(
    r'模型权重保存成功:\s*(.+?/hdcti_model\.ckpt)\s*$'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run base training, frozen Hctx-P head training, and pure outer '
            'inference for all preregistered repeated units.'
        )
    )
    parser.add_argument(
        '--prepared-manifest', default=str(DEFAULT_PREPARED_MANIFEST)
    )
    parser.add_argument('--run-dir')
    parser.add_argument(
        '--stage', choices=('all', 'base', 'head', 'outer'), default='all'
    )
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + '\n',
        encoding='utf-8',
    )


def write_frozen_json(path, value):
    serialized = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + '\n'
    )
    if path.exists():
        if path.read_text(encoding='utf-8') != serialized:
            raise ValueError('Frozen manifest differs: %s' % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding='utf-8')


def verify_file(path, expected_hash, label):
    if not path.is_file():
        raise FileNotFoundError('%s not found: %s' % (label, path))
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(
            '%s hash mismatch: expected=%s actual=%s'
            % (label, expected_hash, actual)
        )
    return actual


def checkpoint_hashes(prefix):
    paths = sorted(prefix.parent.glob(prefix.name + '.*'))
    if not paths:
        raise FileNotFoundError('Checkpoint files not found: %s' % prefix)
    return {str(path): sha256_file(path) for path in paths}


def load_prepared_manifest(path):
    prepared = load_json(path)
    if prepared.get('protocol') != (
            'frozen_base_hctx_router_repeated_units_prepared_v1'):
        raise ValueError('Unexpected prepared-unit manifest protocol.')
    plan_path = resolve_path(prepared['plan'])
    verify_file(plan_path, prepared['plan_sha256'], 'Preregistered plan')
    plan = load_json(plan_path)
    jobs = prepared.get('jobs', [])
    if len(jobs) != int(plan['confirmatory_gate']['new_outer_unit_count']):
        raise ValueError('Prepared manifest does not contain 16 units.')
    seen = set()
    for job in jobs:
        if job['job_key'] in seen:
            raise ValueError('Duplicate prepared job: %s' % job['job_key'])
        seen.add(job['job_key'])
        verify_file(
            resolve_path(job['config']),
            job['config_sha256'],
            '%s config' % job['job_key'],
        )
        verify_file(
            resolve_path(job['artifact_manifest']),
            job['artifact_manifest_sha256'],
            '%s support manifest' % job['job_key'],
        )
    return prepared, plan, plan_path


def default_run_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return (
        REPOSITORY_ROOT / 'results' / 'batch_runs'
        / ('frozen_base_hctx_router_repeated_outer_' + timestamp)
    )


def command_environment(device):
    environment = os.environ.copy()
    environment.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    if device == 'cpu':
        environment['HDCTI_FORCE_CPU'] = '1'
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        environment.pop('HDCTI_FORCE_CPU', None)
        if environment.get('CUDA_VISIBLE_DEVICES') == '-1':
            environment.pop('CUDA_VISIBLE_DEVICES')
    return environment


def run_command(command, log_path=None, environment=None):
    printable = ' '.join(str(part) for part in command)
    print('\n$ %s' % printable, flush=True)
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w', encoding='utf-8')
    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            lines.append(line)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
        status = process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()
    if status != 0:
        raise RuntimeError(
            'Command failed with status %d: %s' % (status, printable)
        )
    return ''.join(lines)


def checkpoint_from_training_output(output):
    matches = []
    for line in output.splitlines():
        match = CHECKPOINT_PATTERN.search(line.strip())
        if match:
            matches.append(match.group(1))
    if not matches:
        raise ValueError('Training output did not report a saved checkpoint.')
    checkpoint = Path(matches[-1]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPOSITORY_ROOT / checkpoint
    return normalize_checkpoint(checkpoint)


def base_manifest_template(prepared_path, prepared, plan_path, plan):
    return {
        'protocol': 'frozen_base_hctx_router_repeated_base_runs_v1',
        'prepared_units_manifest': str(prepared_path),
        'prepared_units_manifest_sha256': sha256_file(prepared_path),
        'plan': str(plan_path),
        'plan_sha256': sha256_file(plan_path),
        'device': None,
        'jobs': {},
    }


def validate_base_entry(job, entry):
    config_path = resolve_path(job['config'])
    verify_file(config_path, job['config_sha256'], 'Frozen base config')
    checkpoint = normalize_checkpoint(entry['checkpoint'])
    index_path = Path(str(checkpoint) + '.index')
    verify_file(
        index_path, entry['checkpoint_index_sha256'], 'Base checkpoint index'
    )
    report_path = resolve_path(entry['baseline_report'])
    verify_file(
        report_path, entry['baseline_report_sha256'], 'Baseline report'
    )
    report = load_json(report_path)
    if report['config']['sha256'] != job['config_sha256']:
        raise ValueError('Baseline/config mismatch for %s.' % job['job_key'])
    if report['support_unit']['assignments_sha256'] != (
            job['assignments_sha256']):
        raise ValueError(
            'Baseline assignment mismatch for %s.' % job['job_key']
        )
    if report['checkpoint']['prefix'] != str(checkpoint):
        raise ValueError(
            'Baseline checkpoint mismatch for %s.' % job['job_key']
        )
    return checkpoint, report_path


def run_base_stage(
        jobs, prepared_path, prepared, plan_path, plan, run_dir,
        environment, device):
    manifest_path = run_dir / 'base_manifest.json'
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        expected = base_manifest_template(
            prepared_path, prepared, plan_path, plan
        )
        for key in expected:
            if key == 'jobs':
                continue
            if key == 'device':
                if manifest.get(key) != device:
                    raise ValueError('Base-stage device differs on resume.')
            elif manifest.get(key) != expected[key]:
                raise ValueError('Base manifest provenance differs: %s' % key)
    else:
        manifest = base_manifest_template(
            prepared_path, prepared, plan_path, plan
        )
        manifest['device'] = device

    for position, job in enumerate(jobs, 1):
        job_key = job['job_key']
        existing = manifest['jobs'].get(job_key)
        if existing is not None:
            validate_base_entry(job, existing)
            print('[base %d/%d] Reusing %s' % (
                position, len(jobs), job_key
            ))
            continue
        print('[base %d/%d] Training %s' % (
            position, len(jobs), job_key
        ))
        job_dir = run_dir / 'base' / job_key
        output = run_command(
            [REPOSITORY_ROOT / 'run_hdcti.sh', resolve_path(job['config'])],
            log_path=job_dir / 'train.log',
            environment=environment,
        )
        checkpoint = checkpoint_from_training_output(output)
        baseline_dir = job_dir / 'baseline'
        run_command([
            sys.executable,
            REPOSITORY_ROOT / 'tools' / 'evaluate_four_state_checkpoint.py',
            '--config', resolve_path(job['config']),
            '--checkpoint', checkpoint,
            '--output-dir', baseline_dir,
        ], log_path=job_dir / 'baseline.log', environment=environment)
        baseline_report = baseline_dir / 'report.json'
        report = load_json(baseline_report)
        if report['support_unit']['assignments_sha256'] != (
                job['assignments_sha256']):
            raise ValueError('Base support assignment changed: %s' % job_key)
        manifest['jobs'][job_key] = {
            'config': str(resolve_path(job['config'])),
            'config_sha256': job['config_sha256'],
            'checkpoint': str(checkpoint),
            'checkpoint_index_sha256': sha256_file(
                Path(str(checkpoint) + '.index')
            ),
            'checkpoint_files': checkpoint_hashes(checkpoint),
            'baseline_report': str(baseline_report),
            'baseline_report_sha256': sha256_file(baseline_report),
            'assignments_sha256': job['assignments_sha256'],
        }
        write_json(manifest_path, manifest)
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def build_head_manifest(prepared_path, plan, jobs, base_manifest):
    datasets = {}
    for job in jobs:
        entry = base_manifest['jobs'][job['job_key']]
        checkpoint, baseline_report = validate_base_entry(job, entry)
        datasets[job['job_key']] = {
            'display_name': '%s c%dp%d' % (
                job['display_name'],
                job['compound_group'],
                job['protein_group'],
            ),
            'dataset': job['dataset'],
            'compound_group': job['compound_group'],
            'protein_group': job['protein_group'],
            'config': str(resolve_path(job['config'])),
            'baseline_report': str(baseline_report),
            'checkpoint': str(checkpoint),
            'checkpoint_index_sha256': entry[
                'checkpoint_index_sha256'
            ],
            'checkpoint_files': entry['checkpoint_files'],
            'assignments_sha256': job['assignments_sha256'],
        }
    return {
        'protocol': 'frozen_base_hctx_router_pilot_v3',
        'purpose': 'preregistered_repeated_outer_head_training',
        'prepared_units_manifest': str(prepared_path),
        'prepared_units_manifest_sha256': sha256_file(prepared_path),
        'head_training': plan['head_training'],
        'routing': plan['routing'],
        'outer_metrics_read': False,
        'datasets': datasets,
    }


def validate_head_entry(job_key, head_manifest_path, report_path):
    report = load_json(report_path)
    if report.get('protocol') != 'frozen_base_hctx_router_pilot_v3':
        raise ValueError('Unexpected head report protocol: %s' % job_key)
    if report['manifest']['sha256'] != sha256_file(head_manifest_path):
        raise ValueError('Head manifest mismatch for %s.' % job_key)
    if not all(report['preservation_checks'].values()):
        raise ValueError('Head preservation failed for %s.' % job_key)
    head_path = resolve_path(report['head_training']['artifact'])
    verify_file(
        head_path,
        report['head_training']['artifact_sha256'],
        '%s Hctx-P head' % job_key,
    )
    return report, head_path


def build_outer_manifest(
        prepared_path, head_manifest_path, jobs, run_dir):
    datasets = {}
    for job in jobs:
        report_path = run_dir / 'heads' / job['job_key'] / 'report.json'
        report, head_path = validate_head_entry(
            job['job_key'], head_manifest_path, report_path
        )
        datasets[job['job_key']] = {
            'job_key': job['job_key'],
            'dataset': job['dataset'],
            'display_name': '%s c%dp%d' % (
                job['display_name'],
                job['compound_group'],
                job['protein_group'],
            ),
            'compound_group': job['compound_group'],
            'protein_group': job['protein_group'],
            'assignments_sha256': job['assignments_sha256'],
            'training_report': str(report_path),
            'training_report_sha256': sha256_file(report_path),
            'head': str(head_path),
            'head_sha256': sha256_file(head_path),
        }
    return {
        'protocol': (
            'frozen_base_hctx_router_repeated_outer_evaluation_v1'
        ),
        'prepared_units_manifest': str(prepared_path),
        'prepared_units_manifest_sha256': sha256_file(prepared_path),
        'head_training_manifest': str(head_manifest_path),
        'head_training_manifest_sha256': sha256_file(head_manifest_path),
        'parameter_selection_on_outer': False,
        'all_heads_frozen_before_outer_evaluation': True,
        'datasets': datasets,
    }


def run_head_stage(
        jobs, prepared_path, plan, run_dir, environment):
    base_manifest_path = run_dir / 'base_manifest.json'
    if not base_manifest_path.is_file():
        raise FileNotFoundError(
            'Base stage is incomplete: %s' % base_manifest_path
        )
    base_manifest = load_json(base_manifest_path)
    missing = [
        job['job_key'] for job in jobs
        if job['job_key'] not in base_manifest.get('jobs', {})
    ]
    if missing:
        raise ValueError('Base jobs are missing: %s' % missing)
    head_manifest_path = run_dir / 'head_training_manifest.json'
    head_manifest = build_head_manifest(
        prepared_path, plan, jobs, base_manifest
    )
    write_frozen_json(head_manifest_path, head_manifest)

    reports = []
    for position, job in enumerate(jobs, 1):
        job_key = job['job_key']
        output_dir = run_dir / 'heads' / job_key
        report_path = output_dir / 'report.json'
        if report_path.is_file():
            validate_head_entry(job_key, head_manifest_path, report_path)
            print('[head %d/%d] Reusing %s' % (
                position, len(jobs), job_key
            ))
        else:
            print('[head %d/%d] Training %s' % (
                position, len(jobs), job_key
            ))
            run_command([
                sys.executable,
                REPOSITORY_ROOT / 'tools'
                / 'train_frozen_base_hctx_router.py',
                '--manifest', head_manifest_path,
                '--dataset', job_key,
                '--output-dir', output_dir,
            ], log_path=run_dir / 'heads' / (job_key + '.log'),
               environment=environment)
            validate_head_entry(job_key, head_manifest_path, report_path)
        reports.append(report_path)

    summary_command = [
        sys.executable,
        REPOSITORY_ROOT / 'tools'
        / 'summarize_frozen_base_hctx_router.py',
        '--output-dir', run_dir / 'head_diagnostics',
    ]
    for report_path in reports:
        summary_command.extend(['--report', report_path])
    run_command(
        summary_command,
        log_path=run_dir / 'head_diagnostics.log',
        environment=environment,
    )

    outer_manifest_path = run_dir / 'outer_evaluation_manifest.json'
    outer_manifest = build_outer_manifest(
        prepared_path, head_manifest_path, jobs, run_dir
    )
    write_frozen_json(outer_manifest_path, outer_manifest)
    print('All 16 heads frozen before outer evaluation: %s' % (
        outer_manifest_path
    ))
    return outer_manifest_path


def validate_outer_report(job, manifest_path, report_path):
    report = load_json(report_path)
    if report.get('protocol') != (
            'frozen_base_hctx_router_repeated_outer_evaluation_v1'):
        raise ValueError('Unexpected repeated outer report protocol.')
    if report['job']['key'] != job['job_key']:
        raise ValueError('Outer report job mismatch: %s' % job['job_key'])
    if report['manifest']['sha256'] != sha256_file(manifest_path):
        raise ValueError('Outer manifest mismatch: %s' % job['job_key'])
    if report['training_optimizer_steps'] != 0:
        raise ValueError('Outer evaluation trained parameters.')
    if report['parameter_selection_on_outer']:
        raise ValueError('Outer evaluation selected parameters.')
    if not all(report['preservation_checks'].values()):
        raise ValueError('Outer artifact preservation failed.')
    return report


def run_outer_stage(
        jobs, prepared_path, plan_path, run_dir, environment):
    manifest_path = run_dir / 'outer_evaluation_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            'Frozen outer manifest is absent. Complete the head stage first.'
        )
    reports = []
    for position, job in enumerate(jobs, 1):
        job_key = job['job_key']
        output_dir = run_dir / 'outer' / job_key
        report_path = output_dir / 'report.json'
        if report_path.is_file():
            validate_outer_report(job, manifest_path, report_path)
            print('[outer %d/%d] Reusing %s' % (
                position, len(jobs), job_key
            ))
        else:
            print('[outer %d/%d] Evaluating %s' % (
                position, len(jobs), job_key
            ))
            run_command([
                sys.executable,
                REPOSITORY_ROOT / 'tools'
                / 'evaluate_frozen_base_hctx_router_outer.py',
                '--manifest', manifest_path,
                '--dataset', job_key,
                '--output-dir', output_dir,
            ], log_path=run_dir / 'outer' / (job_key + '.log'),
               environment=environment)
            validate_outer_report(job, manifest_path, report_path)
        reports.append(report_path)

    summary_command = [
        sys.executable,
        REPOSITORY_ROOT / 'tools'
        / 'summarize_repeated_frozen_base_hctx_router_outer.py',
        '--plan', plan_path,
        '--prepared-manifest', prepared_path,
        '--output-dir', run_dir,
        '--require-pass',
    ]
    for report_path in reports:
        summary_command.extend(['--report', report_path])
    run_command(
        summary_command,
        log_path=run_dir / 'summary.log',
        environment=environment,
    )
    return run_dir / 'summary.md'


def dry_run_report(prepared_path, plan_path, jobs, stage, device, run_dir):
    print('Repeated frozen-router confirmation dry run')
    print('  prepared manifest: %s' % prepared_path)
    print('  plan: %s' % plan_path)
    print('  stage: %s' % stage)
    print('  device: %s' % device)
    print('  run directory: %s' % run_dir)
    print('  new outer units: %d' % len(jobs))
    for job in jobs:
        print(
            '  - %s: %s c%dp%d' % (
                job['job_key'], job['display_name'],
                job['compound_group'], job['protein_group'],
            )
        )


def main():
    args = parse_args()
    prepared_path = resolve_path(args.prepared_manifest)
    prepared, plan, plan_path = load_prepared_manifest(prepared_path)
    jobs = list(prepared['jobs'])
    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
    elif args.stage in ('head', 'outer') and not args.dry_run:
        raise ValueError('--run-dir is required when resuming head/outer.')
    else:
        run_dir = default_run_dir()

    if args.dry_run:
        dry_run_report(
            prepared_path, plan_path, jobs, args.stage, args.device, run_dir
        )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    environment = command_environment(args.device)
    if args.stage in ('all', 'base'):
        run_base_stage(
            jobs, prepared_path, prepared, plan_path, plan, run_dir,
            environment, args.device,
        )
    if args.stage in ('all', 'head'):
        run_head_stage(jobs, prepared_path, plan, run_dir, environment)
    if args.stage in ('all', 'outer'):
        summary_path = run_outer_stage(
            jobs, prepared_path, plan_path, run_dir, environment
        )
        print('\nRepeated outer Gate summary: %s' % summary_path)
    else:
        print('\nCompleted stage %s: %s' % (args.stage, run_dir))


if __name__ == '__main__':
    main()
