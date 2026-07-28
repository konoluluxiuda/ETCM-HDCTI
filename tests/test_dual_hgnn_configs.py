import unittest
from pathlib import Path

from util.config import ModelConf
from util.model_components import resolve_encoder_profile


class DualHGNNConfigsTest(unittest.TestCase):
    CONFIGS = (
        'DualHGNN_tcmsuite_pair_stratified_pilot.conf',
        'DualHGNN_tcmsp_pair_stratified_pilot.conf',
        'DualHGNN_symmap_pair_stratified_pilot.conf',
        'DualHGNN_etcm_mention10_pair_stratified_pilot.conf',
    )

    def test_four_dataset_pilots_use_one_frozen_protocol(self):
        root = Path(__file__).resolve().parents[1]
        for name in self.CONFIGS:
            config = ModelConf(str(root / 'configs' / name))
            profile = resolve_encoder_profile(config)
            with self.subTest(config=name):
                self.assertEqual(profile['name'], 'dual_hgnn_cti')
                self.assertEqual(config['experiment.protocol'], 'strict')
                self.assertEqual(config['split.strategy'], 'pair_stratified')
                self.assertEqual(config['split.seed'], '2026')
                self.assertEqual(config['evaluation.fold.limit'], '1')
                self.assertEqual(config['evaluation.outer.test'], 'False')
                self.assertEqual(config['pair.decoder'], 'dot')
                self.assertEqual(config['context.interaction'], 'False')
                self.assertEqual(config['hyperedge.attention'], 'False')
                self.assertEqual(config['global.token.attention'], 'False')
                self.assertEqual(config['attention.max.nodes'], '0')


if __name__ == '__main__':
    unittest.main()
