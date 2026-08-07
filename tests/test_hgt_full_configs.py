import subprocess
import unittest
from pathlib import Path

from util.config import ModelConf


class HGTFullConfigsTest(unittest.TestCase):
    DATASET_SLUGS = (
        'tcmsuite',
        'tcmsp',
        'symmap',
        'etcm_mention10',
    )

    def test_full_configs_only_enable_complete_outer_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        for dataset_slug in self.DATASET_SLUGS:
            with self.subTest(dataset=dataset_slug):
                full = ModelConf(str(
                    root / 'configs' /
                    f'HGTCTI_{dataset_slug}_pair_stratified_full.conf'
                ))
                self.assertEqual(full['evaluation.setup'], '-cv 5')
                self.assertEqual(full['experiment.protocol'], 'strict')
                self.assertEqual(full['split.strategy'], 'pair_stratified')
                self.assertEqual(full['split.seed'], '2026')
                self.assertEqual(full['evaluation.outer.test'], 'True')
                self.assertFalse(full.contains('evaluation.fold.limit'))
                self.assertTrue(full['model.variant'].endswith('_full_v1'))

    def test_full_batch_dry_run_lists_four_jobs(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            ['bash', str(root / 'run_hgt_cti_full_batch.sh'), '--dry-run'],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count('.conf'), 4)
        for dataset_slug in self.DATASET_SLUGS:
            self.assertIn(
                f'HGTCTI_{dataset_slug}_pair_stratified_full.conf',
                completed.stdout,
            )


if __name__ == '__main__':
    unittest.main()
