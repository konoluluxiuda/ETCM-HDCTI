import subprocess
import unittest
from pathlib import Path


class ExternalBaselineFullBatchTest(unittest.TestCase):
    def test_dry_run_lists_all_frozen_jobs(self):
        root = Path(__file__).resolve().parents[1]
        script = root / 'run_external_baselines_full_batch.sh'
        completed = subprocess.run(
            ['bash', str(script), '--dry-run'],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count('.conf'), 12)
        for method in ('Dual-HGNN-CTI', 'LightGCN-CTI', 'R-GCN-CTI'):
            self.assertEqual(completed.stdout.count(method), 4)
        for dataset in (
            'tcmsuite',
            'tcmsp',
            'symmap',
            'etcm_mention10',
        ):
            for model_prefix in ('DualHGNN', 'LightGCNCTI', 'RGCNCTI'):
                config_name = (
                    f'{model_prefix}_{dataset}_pair_stratified_full.conf'
                )
                self.assertIn(config_name, completed.stdout)


if __name__ == '__main__':
    unittest.main()
