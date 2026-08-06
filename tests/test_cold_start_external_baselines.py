import json
import unittest
from pathlib import Path

from tools.validate_cold_start_external_baselines import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs' / 'cold_start_external_baselines_manifest.json'


class ColdStartExternalBaselineConfigTest(unittest.TestCase):
    def test_frozen_manifest_and_all_configs_are_valid(self):
        manifest, jobs = validate_manifest(MANIFEST)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(manifest['random_seed'], 52026)
        self.assertFalse(manifest['dataset_specific_tuning'])

    def test_manifest_is_complete_cartesian_product(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        expected = {
            (dataset['key'], method['key'])
            for dataset in manifest['datasets']
            for method in manifest['methods']
        }
        actual = {
            (job['dataset'], job['method']) for job in manifest['jobs']
        }
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
