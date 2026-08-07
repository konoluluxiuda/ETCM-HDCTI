import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.export_paper_result_bundle import export_bundle, verify_bundle


ROOT = Path(__file__).resolve().parents[1]


class PaperResultBundleTest(unittest.TestCase):
    def test_published_bundle_verifies_without_raw_results(self):
        manifest_path, records = verify_bundle()
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(len(records), 10)

    def test_exported_bundle_matches_all_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir)
            plan = json.loads((
                ROOT / 'configs' / 'paper_result_bundle_sources.json'
            ).read_text(encoding='utf-8'))
            plan['bundle'] = str(temporary / 'bundle')
            plan_path = temporary / 'plan.json'
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False), encoding='utf-8'
            )
            with mock.patch(
                    'tools.export_paper_result_bundle.ROOT', ROOT):
                manifest_path, records = export_bundle(plan_path)
            self.assertEqual(len(records), 10)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['files']), 10)


if __name__ == '__main__':
    unittest.main()
