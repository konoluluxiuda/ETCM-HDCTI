import unittest
from pathlib import Path

from util.config import ModelConf
from util.model_components import resolve_hyperedge_attention


class HyperedgeAttentionConfigTest(unittest.TestCase):
    DATASETS = (
        'tcmsuite',
        'tcmsp',
        'symmap',
        'etcm_mention10',
    )
    NEW_KEYS = {
        'evaluation.outer.test',
        'hyperedge.attention',
        'hyperedge.attention.mode',
        'hyperedge.attention.hc',
        'hyperedge.attention.pd',
        'hyperedge.attention.temperature',
        'hyperedge.attention.prior.scale',
    }

    def test_pilots_change_only_variant_and_hyperedge_attention(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_herb_only_pilot.conf' % dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_sp_fbha_pilot.conf' % dataset)
                ))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in self.NEW_KEYS:
                    candidate_values.pop(key)
                    baseline_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                settings = resolve_hyperedge_attention(candidate)
                self.assertTrue(settings['enabled'])
                self.assertTrue(settings['hc_enabled'])
                self.assertTrue(settings['pd_enabled'])
                self.assertEqual(settings['temperature'], 1.0)
                self.assertEqual(settings['prior_scale'], 0.1)
                self.assertEqual(candidate['evaluation.fold.limit'], '1')
                self.assertEqual(candidate['evaluation.outer.test'], 'False')
                self.assertEqual(candidate['counterfactual.context'], 'False')


if __name__ == '__main__':
    unittest.main()
