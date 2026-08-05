#!/usr/bin/env python3
"""Run the frozen 16-unit V3 versus jointly trained SDIS comparison."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_four_state_checkpoint import sha256_file  # noqa: E402
from tools.run_frozen_base_hctx_router_repeated_outer import (  # noqa: E402
    checkpoint_from_training_output,
    checkpoint_hashes,
    command_environment,
    run_command,
    write_json,
    write_frozen_json,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / 'configs' / (
    'frozen_base_hctx_router_vs_sdis_units_manifest.json'
)


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_file(path, expected, label):
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError('%s is absent or has a hash mismatch.' % label)


def default_run_dir():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return REPOSITORY_ROOT / 'results' / 'batch_runs' / (
        'frozen_base_router_vs_sdis_' + stamp
    )


def load_manifest(path):
    manifest = load_json(path)
    if manifest.get('protocol') != 'frozen_base_hctx_router_vs_sdis_units_v1':
        raise ValueError('Unexpected V3/SDIS unit manifest.')
    plan_path = resolve_path(manifest['plan'])
    verify_file(plan_path, manifest['plan_sha256'], 'Comparison plan')
    plan = load_json(plan_path)
    verify_file(
        resolve_path(manifest['prepared_units_manifest']),
        manifest['prepared_units_manifest_sha256'], 'Prepared units',
    )
    verify_file(
        resolve_path(manifest['v3_repeated_summary']),
        manifest['v3_repeated_summary_sha256'], 'V3 repeated summary',
    )
    if len(manifest.get('jobs', [])) != 16:
        raise ValueError('Comparison manifest must contain 16 units.')
    for job in manifest['jobs']:
        verify_file(resolve_path(job['config']), job['config_sha256'], job['job_key'])
    return manifest, plan, plan_path


def validate_training_entry(job, entry):
    if entry['job_key'] != job['job_key']:
        raise ValueError('SDIS training job mismatch.')
    verify_file(resolve_path(entry['config']), entry['config_sha256'], 'SDIS config')
    prefix = resolve_path(entry['checkpoint'])
    if checkpoint_hashes(prefix) != entry['checkpoint_files']:
        raise ValueError('Frozen SDIS checkpoint changed: %s' % job['job_key'])


def training_progress_template(manifest_path, device):
    return {
        'protocol': 'frozen_base_hctx_router_vs_sdis_training_progress_v1',
        'unit_manifest': str(manifest_path),
        'unit_manifest_sha256': sha256_file(manifest_path),
        'device': device,
        'complete': False,
        'jobs': {},
    }


def load_training_progress(path, manifest_path, device):
    expected = training_progress_template(manifest_path, device)
    if not path.is_file():
        return expected
    progress = load_json(path)
    for key in ('protocol', 'unit_manifest', 'unit_manifest_sha256', 'device'):
        if progress.get(key) != expected[key]:
            raise ValueError('Training progress provenance differs: %s' % key)
    if not isinstance(progress.get('jobs'), dict):
        raise ValueError('Training progress jobs must be a mapping.')
    return progress


def train_stage(manifest_path, manifest, jobs, run_dir, environment, device):
    frozen_path = run_dir / 'sdis_training_manifest.json'
    progress_path = run_dir / 'sdis_training_progress.json'
    if frozen_path.is_file():
        frozen = load_json(frozen_path)
        if frozen.get('unit_manifest_sha256') != sha256_file(manifest_path):
            raise ValueError('Frozen SDIS manifest provenance differs.')
        if frozen.get('device') != device:
            raise ValueError('Frozen SDIS training device differs on resume.')
        entries = {row['job_key']: row for row in frozen['jobs']}
        for job in jobs:
            validate_training_entry(job, entries[job['job_key']])
        print('Reusing all %d frozen SDIS checkpoints.' % len(jobs))
        return frozen_path

    progress = load_training_progress(progress_path, manifest_path, device)
    entries = progress['jobs']
    for index, job in enumerate(jobs, 1):
        if job['job_key'] in entries:
            validate_training_entry(job, entries[job['job_key']])
            print('[train %d/%d] Reusing %s' % (index, len(jobs), job['job_key']))
            continue
        print('[train %d/%d] Training %s' % (index, len(jobs), job['job_key']))
        output = run_command([
            sys.executable, REPOSITORY_ROOT / 'main.py',
            '--config', resolve_path(job['config']),
        ], log_path=run_dir / 'train' / (job['job_key'] + '.log'),
           environment=environment)
        checkpoint = checkpoint_from_training_output(output)
        entries[job['job_key']] = {
            'job_key': job['job_key'],
            'dataset': job['dataset'],
            'config': job['config'],
            'config_sha256': job['config_sha256'],
            'checkpoint': str(checkpoint),
            'checkpoint_files': checkpoint_hashes(checkpoint),
        }
        write_json(progress_path, progress)
    frozen = {
        'protocol': 'frozen_base_hctx_router_vs_sdis_training_v1',
        'unit_manifest': str(manifest_path),
        'unit_manifest_sha256': sha256_file(manifest_path),
        'device': device,
        'all_checkpoints_frozen_before_outer': True,
        'outer_metrics_read': False,
        'jobs': [entries[job['job_key']] for job in jobs],
    }
    write_frozen_json(frozen_path, frozen)
    progress['complete'] = True
    progress['frozen_manifest'] = str(frozen_path)
    progress['frozen_manifest_sha256'] = sha256_file(frozen_path)
    write_json(progress_path, progress)
    return frozen_path


def validate_outer_report(job, entry, report_path):
    report = load_json(report_path)
    if report.get('evaluation') != 'four_state_checkpoint_pure_inference':
        raise ValueError('Unexpected SDIS evaluation: %s' % job['job_key'])
    if report.get('records') != 'outer':
        raise ValueError('SDIS report is not an outer evaluation.')
    if report.get('training_optimizer_steps') != 0:
        raise ValueError('SDIS outer evaluation performed training.')
    if report.get('parameter_selection_on_records'):
        raise ValueError('SDIS outer evaluation selected parameters.')
    if report['config']['sha256'] != job['config_sha256']:
        raise ValueError('SDIS outer config changed: %s' % job['job_key'])
    if report['checkpoint']['prefix'] != str(resolve_path(entry['checkpoint'])):
        raise ValueError('SDIS outer checkpoint mismatch: %s' % job['job_key'])
    if report['support_unit']['assignments_sha256'] != job[
            'assignments_sha256']:
        raise ValueError('SDIS outer assignment mismatch: %s' % job['job_key'])
    return report


def outer_stage(manifest_path, plan_path, jobs, run_dir, environment):
    training_path = run_dir / 'sdis_training_manifest.json'
    if not training_path.is_file():
        raise FileNotFoundError('Complete the train stage before outer evaluation.')
    training = load_json(training_path)
    entries = {row['job_key']: row for row in training['jobs']}
    reports = []
    for index, job in enumerate(jobs, 1):
        validate_training_entry(job, entries[job['job_key']])
        output_dir = run_dir / 'outer' / job['job_key']
        report_path = output_dir / 'report.json'
        if report_path.is_file():
            validate_outer_report(job, entries[job['job_key']], report_path)
            print('[outer %d/%d] Reusing %s' % (index, len(jobs), job['job_key']))
        else:
            print('[outer %d/%d] Evaluating %s' % (index, len(jobs), job['job_key']))
            run_command([
                sys.executable,
                REPOSITORY_ROOT / 'tools' / 'evaluate_four_state_checkpoint.py',
                '--config', resolve_path(job['config']),
                '--checkpoint', entries[job['job_key']]['checkpoint'],
                '--records', 'outer', '--output-dir', output_dir,
            ], log_path=run_dir / 'outer' / (job['job_key'] + '.log'),
               environment=environment)
            validate_outer_report(job, entries[job['job_key']], report_path)
        reports.append(report_path)

    command = [
        sys.executable,
        REPOSITORY_ROOT / 'tools' / 'summarize_frozen_base_hctx_router_vs_sdis.py',
        '--plan', plan_path, '--manifest', manifest_path,
        '--output-dir', run_dir,
    ]
    for report in reports:
        command.extend(['--sdis-report', report])
    run_command(command, log_path=run_dir / 'summary.log', environment=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--run-dir')
    parser.add_argument('--stage', choices=('all', 'train', 'outer'), default='all')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.stage == 'outer' and not args.run_dir:
        parser.error('--stage outer requires --run-dir from the train stage.')
    manifest_path = resolve_path(args.manifest)
    manifest, _, plan_path = load_manifest(manifest_path)
    jobs = manifest['jobs']
    run_dir = resolve_path(args.run_dir) if args.run_dir else default_run_dir()
    if args.dry_run:
        print('Frozen V3 versus SDIS comparison')
        print('  units: %d' % len(jobs))
        print('  stage: %s' % args.stage)
        print('  device: %s' % args.device)
        print('  run directory: %s' % run_dir)
        for job in jobs:
            print('  - %s: %s' % (job['job_key'], job['config']))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = command_environment(args.device)
    if args.stage in ('all', 'train'):
        train_stage(
            manifest_path, manifest, jobs, run_dir, environment, args.device
        )
    if args.stage in ('all', 'outer'):
        outer_stage(manifest_path, plan_path, jobs, run_dir, environment)
    print('Results directory: %s' % run_dir)


if __name__ == '__main__':
    main()
