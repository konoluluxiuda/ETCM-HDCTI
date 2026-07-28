import subprocess
import unittest
from pathlib import Path

from util.config import ModelConf


class HGTConfigsTest(unittest.TestCase):
    CONFIGS = (
        'HGTCTI_tcmsuite_pair_stratified_pilot.conf',
        'HGTCTI_tcmsp_pair_stratified_pilot.conf',
        'HGTCTI_symmap_pair_stratified_pilot.conf',
        'HGTCTI_etcm_mention10_pair_stratified_pilot.conf',
    )

    def test_four_dataset_pilots_use_one_frozen_protocol(self):
        root = Path(__file__).resolve().parents[1]
        for name in self.CONFIGS:
            config = ModelConf(str(root / 'configs' / name))
            with self.subTest(config=name):
                self.assertEqual(config['model.name'], 'HGTCTI')
                self.assertEqual(config['experiment.protocol'], 'strict')
                self.assertEqual(config['split.strategy'], 'pair_stratified')
                self.assertEqual(config['split.seed'], '2026')
                self.assertEqual(config['evaluation.fold.limit'], '1')
                self.assertEqual(config['evaluation.outer.test'], 'False')
                self.assertEqual(config['pair.decoder'], 'dot')
                self.assertEqual(config['hgt.layers'], '2')
                self.assertEqual(config['hgt.heads'], '2')
                self.assertEqual(config['hgt.activation'], 'gelu')
                self.assertEqual(config['hgt.objective'], 'bce')
                self.assertEqual(config['hgt.max.neighbors'], '64')
                self.assertEqual(config['hgt.sampling.seed'], '2026')
                self.assertFalse(config.contains('context.interaction'))
                self.assertFalse(config.contains('attention.max.nodes'))

    def test_batch_dry_run_lists_the_four_pilots(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            ['bash', str(root / 'run_hgt_cti_pilot_batch.sh'), '--dry-run'],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count('.conf'), 4)
        for name in self.CONFIGS:
            self.assertIn(name, completed.stdout)


if __name__ == '__main__':
    unittest.main()
