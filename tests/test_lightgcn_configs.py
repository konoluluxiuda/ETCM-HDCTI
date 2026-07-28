import unittest
from pathlib import Path

from util.config import ModelConf


class LightGCNConfigsTest(unittest.TestCase):
    CONFIGS = (
        'LightGCNCTI_tcmsuite_pair_stratified_pilot.conf',
        'LightGCNCTI_tcmsp_pair_stratified_pilot.conf',
        'LightGCNCTI_symmap_pair_stratified_pilot.conf',
        'LightGCNCTI_etcm_mention10_pair_stratified_pilot.conf',
    )

    def test_four_dataset_pilots_use_one_frozen_protocol(self):
        root = Path(__file__).resolve().parents[1]
        for name in self.CONFIGS:
            config = ModelConf(str(root / 'configs' / name))
            with self.subTest(config=name):
                self.assertEqual(config['model.name'], 'LightGCNCTI')
                self.assertEqual(config['experiment.protocol'], 'strict')
                self.assertEqual(config['split.strategy'], 'pair_stratified')
                self.assertEqual(config['split.seed'], '2026')
                self.assertEqual(config['evaluation.fold.limit'], '1')
                self.assertEqual(config['evaluation.outer.test'], 'False')
                self.assertEqual(config['pair.decoder'], 'dot')
                self.assertEqual(config['lightgcn.layers'], '3')
                self.assertEqual(config['lightgcn.objective'], 'bce')
                self.assertFalse(config.contains('context.interaction'))
                self.assertFalse(config.contains('attention.max.nodes'))


if __name__ == '__main__':
    unittest.main()
