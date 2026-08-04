#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.validate_hctx_ablation_configs import (
    config_differences,
    parse_config,
    repository_path,
    sha256_file,
)


DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / 'configs' / 'cold_start_hctx_ablation_manifest.json'
)


def load_rows(path):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def require_values(config, expected, label):
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('%s requires %s=%s.' % (label, key, value))


def successful_reference(rows, dataset, variant, config_hash):
    matching = [
        row for row in rows
        if row.get('dataset') == dataset
        and row.get('variant') == variant
        and row.get('status') == 'OK'
    ]
    if len(matching) != 1:
        raise ValueError(
            'Expected one successful %s row for %s.' % (variant, dataset)
        )
    if matching[0].get('config_sha256') != config_hash:
        raise ValueError(
            'Frozen %s result hash mismatch for %s.' % (variant, dataset)
        )
    return matching[0]


def validate_manifest(manifest_path=DEFAULT_MANIFEST):
    manifest_path = repository_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported cold-start ablation manifest schema.')
    allowed = set(
        manifest.get('allowed_no_context_hctx_differences') or []
    )
    if allowed != {
            'model.variant', 'context.interaction', 'context.herb_protein'}:
        raise ValueError('Unexpected allowed config differences.')
    datasets = manifest.get('datasets') or []
    if len(datasets) != 4:
        raise ValueError('Cold-start ablation requires four datasets.')

    reference_path = repository_path(manifest['reference_results'])
    if not reference_path.is_file():
        raise FileNotFoundError('Missing frozen results: %s' % reference_path)
    reference_rows = load_rows(reference_path)
    shared = {
        'evaluation.setup': '-cv 5',
        'evaluation.outer.test': 'True',
        'experiment.protocol': 'strict',
        'random.seed': '2026',
        'split.strategy': 'compound_cold_start',
        'split.seed': '2026',
        'split.reuse': 'True',
        'early.stopping': 'True',
        'validation.seed': '102026',
        'pair.decoder': 'dot',
        'counterfactual.context': 'False',
        'context.mask.training': 'False',
        'support.router': 'False',
        'hyperedge.attention': 'False',
        'global.token.attention': 'False',
        'attention.max.nodes': '0',
    }
    validated = []
    for dataset in datasets:
        paths = {}
        configs = {}
        for role in ('no_context', 'hctx', 'sdis'):
            path = repository_path(dataset[role + '_config'])
            expected_hash = dataset[role + '_sha256']
            if not path.is_file():
                raise FileNotFoundError('Missing config: %s' % path)
            if sha256_file(path) != expected_hash:
                raise ValueError('Config hash mismatch for %s.' % path)
            paths[role] = path
            configs[role] = parse_config(path)

        differences = config_differences(
            configs['no_context'], configs['hctx']
        )
        if set(differences) != allowed:
            raise ValueError(
                '%s NoContext/Hctx-P differences are %s.'
                % (dataset['name'], sorted(differences))
            )
        require_values(configs['no_context'], shared, 'NoContext')
        require_values(configs['hctx'], shared, 'Hctx-P')
        require_values(configs['sdis'], shared, 'SDIS')
        require_values(configs['no_context'], {
            'context.interaction': 'False',
            'context.herb_protein': 'False',
            'inductive.context': 'False',
        }, 'NoContext')
        require_values(configs['hctx'], {
            'context.interaction': 'True',
            'context.herb_protein': 'True',
            'inductive.context': 'False',
        }, 'Hctx-P')
        require_values(configs['sdis'], {
            'context.interaction': 'True',
            'context.herb_protein': 'True',
            'inductive.context': 'True',
            'inductive.context.suppress.base.zero.support': 'True',
            'inductive.context.self.excluded': 'False',
        }, 'SDIS')
        successful_reference(
            reference_rows, dataset['name'], 'HerbOnly',
            dataset['hctx_sha256'],
        )
        successful_reference(
            reference_rows, dataset['name'], 'SDIS',
            dataset['sdis_sha256'],
        )
        validated.append({
            'dataset': dataset['name'],
            'paths': paths,
            'differences': differences,
        })
    return manifest, validated


def main():
    parser = argparse.ArgumentParser(
        description='Validate frozen compound cold-start ablation configs.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    manifest, validated = validate_manifest(args.manifest)
    print('Protocol: %s' % manifest['protocol'])
    for item in validated:
        print('%s: OK (%s)' % (
            item['dataset'], ', '.join(sorted(item['differences']))
        ))
    print('Frozen NoContext, Hctx-P, and SDIS chain is valid.')


if __name__ == '__main__':
    main()
