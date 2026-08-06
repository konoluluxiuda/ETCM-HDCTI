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


def main():
    parser = argparse.ArgumentParser(
        description='Export the compact, hash-verified paper result bundle.'
    )
    parser.add_argument('--plan', default=str(DEFAULT_PLAN))
    args = parser.parse_args()
    manifest_path, records = export_bundle(args.plan)
    print('Exported %d frozen result files.' % len(records))
    print('Bundle manifest: %s' % manifest_path)


if __name__ == '__main__':
    main()
