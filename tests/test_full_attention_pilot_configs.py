import unittest
from pathlib import Path

from util.config import ModelConf


class FullAttentionPilotConfigTest(unittest.TestCase):
    DATASETS = ('tcmsuite', 'tcmsp')

    def test_pilots_only_enable_dense_attention(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                no_dense = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_chcr_no_dense_pilot.conf' %
                     dataset)
                ))
                full_attention = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_chcr_full_attention_pilot.conf' %
                     dataset)
                ))

                no_dense_values = dict(no_dense.config)
                full_attention_values = dict(full_attention.config)
                no_dense_values.pop('model.variant')
                full_attention_values.pop('model.variant')
                no_dense_values.pop('attention.max.nodes')

                self.assertEqual(full_attention_values, no_dense_values)
                self.assertFalse(full_attention.contains('attention.max.nodes'))
                self.assertEqual(no_dense['attention.max.nodes'], '0')
                self.assertEqual(full_attention['evaluation.fold.limit'], '1')
                self.assertEqual(full_attention['evaluation.outer.test'], 'False')
                self.assertEqual(full_attention['counterfactual.context'], 'True')
                self.assertEqual(full_attention['hyperedge.attention'], 'False')


if __name__ == '__main__':
    unittest.main()
