#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from tools.build_paper_results_tables import repository_path, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / 'configs' / 'paper_result_bundle_sources.json'


def portable_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def export_bundle(plan_path=DEFAULT_PLAN):
    plan_path = repository_path(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema_version') != 1:
        raise ValueError('Unsupported paper result bundle schema.')
    bundle_dir = repository_path(plan['bundle'])
    bundle_dir.mkdir(parents=True, exist_ok=True)
    records = []
    destinations = set()
    for item in plan.get('sources', []):
        source = repository_path(item['source'])
        destination = bundle_dir / item['destination']
        if item['destination'] in destinations:
            raise ValueError('Duplicate bundle destination: %s' % destination)
        destinations.add(item['destination'])
        if not source.is_file():
            raise FileNotFoundError('Missing paper result source: %s' % source)
        actual = sha256_file(source)
        if actual != item['sha256']:
            raise ValueError('Source hash mismatch for %s.' % source)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != item['sha256']:
            raise ValueError('Copied bundle file changed: %s' % destination)
        records.append({
            'name': item['name'],
            'path': portable_path(destination),
            'sha256': item['sha256'],
            'original_source': item['source'],
            'size_bytes': destination.stat().st_size,
        })
    manifest = {
        'schema_version': 1,
        'source_plan': portable_path(plan_path),
        'source_plan_sha256': sha256_file(plan_path),
        'files': records,
    }
    manifest_path = bundle_dir / 'MANIFEST.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return manifest_path, records


def verify_bundle(plan_path=DEFAULT_PLAN):
    plan_path = repository_path(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema_version') != 1:
        raise ValueError('Unsupported paper result bundle schema.')
    bundle_dir = repository_path(plan['bundle'])
    manifest_path = bundle_dir / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported paper result manifest schema.')
    if manifest.get('source_plan_sha256') != sha256_file(plan_path):
        raise ValueError('Paper result source plan hash mismatch.')

    expected = {
        item['destination']: (item['name'], item['sha256'])
        for item in plan.get('sources', [])
    }
    records = manifest.get('files', [])
    if len(records) != len(expected):
        raise ValueError('Paper result bundle file count mismatch.')
    for record in records:
        destination = Path(record['path']).name
        if destination not in expected:
            raise ValueError('Unexpected paper result file: %s' % destination)
        expected_name, expected_hash = expected[destination]
        if record['name'] != expected_name or record['sha256'] != expected_hash:
            raise ValueError('Paper result manifest entry mismatch: %s' % destination)
        artifact = repository_path(record['path'])
        if not artifact.is_file():
            raise FileNotFoundError('Missing paper result artifact: %s' % artifact)
        if sha256_file(artifact) != expected_hash:
            raise ValueError('Paper result artifact hash mismatch: %s' % artifact)
        if artifact.stat().st_size != int(record['size_bytes']):
            raise ValueError('Paper result artifact size mismatch: %s' % artifact)
    return manifest_path, records


def main():
    parser = argparse.ArgumentParser(
        description='Export the compact, hash-verified paper result bundle.'
    )
    parser.add_argument('--plan', default=str(DEFAULT_PLAN))
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='verify the published bundle without requiring raw result sources',
    )
    args = parser.parse_args()
    if args.verify_only:
        manifest_path, records = verify_bundle(args.plan)
        print('Verified %d frozen result files.' % len(records))
    else:
        manifest_path, records = export_bundle(args.plan)
        print('Exported %d frozen result files.' % len(records))
    print('Bundle manifest: %s' % manifest_path)


if __name__ == '__main__':
    main()
