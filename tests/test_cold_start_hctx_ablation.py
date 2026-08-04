import json
import unittest
from pathlib import Path


from tools.validate_hctx_ablation_configs import (
    config_differences,
    parse_config,
    sha256_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY_ROOT / 'configs' / 'cold_start_hctx_ablation_manifest.json'
)


class ColdStartHctxAblationConfigTest(unittest.TestCase):
    def test_four_frozen_config_chains(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['datasets']), 4)
        allowed = set(manifest['allowed_no_context_hctx_differences'])
        self.assertEqual(allowed, {
            'model.variant',
            'context.interaction',
            'context.herb_protein',
        })
        for dataset in manifest['datasets']:
            configs = {}
            for role in ('no_context', 'hctx', 'sdis'):
                path = REPOSITORY_ROOT / dataset[role + '_config']
                self.assertEqual(
                    sha256_file(path), dataset[role + '_sha256']
                )
                configs[role] = parse_config(path)
            self.assertEqual(
                set(config_differences(
                    configs['no_context'], configs['hctx']
                )),
                allowed,
            )
            for role in configs:
                with self.subTest(dataset=dataset['name'], role=role):
                    self.assertEqual(
                        configs[role]['split.strategy'],
                        'compound_cold_start',
                    )
                    self.assertEqual(
                        configs[role]['attention.max.nodes'], '0'
                    )
                    self.assertEqual(configs[role]['pair.decoder'], 'dot')
            self.assertEqual(
                configs['no_context']['context.interaction'], 'False'
            )
            self.assertEqual(
                configs['no_context']['context.herb_protein'], 'False'
            )
            self.assertEqual(
                configs['hctx']['context.herb_protein'], 'True'
            )
            self.assertEqual(
                configs['sdis']['inductive.context'], 'True'
            )


if __name__ == '__main__':
    unittest.main()
