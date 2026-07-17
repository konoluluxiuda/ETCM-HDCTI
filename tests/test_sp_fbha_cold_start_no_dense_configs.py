import unittest
from pathlib import Path

from util.config import ModelConf
from util.model_components import resolve_hyperedge_attention


class ColdStartNoDenseSpFbhaConfigTest(unittest.TestCase):
    DATASETS = (
        'tcmsuite',
        'tcmsp',
        'symmap',
        'etcm_mention10',
    )
    ATTENTION_KEYS = {
        'hyperedge.attention',
        'hyperedge.attention.mode',
        'hyperedge.attention.hc',
        'hyperedge.attention.pd',
        'hyperedge.attention.temperature',
        'hyperedge.attention.prior.scale',
    }

    def test_candidates_are_paired_with_no_dense_cold_start_baselines(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_no_dense_herb_only_pilot.conf' % dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_no_dense_sp_fbha_pilot.conf' % dataset)
                ))

                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in self.ATTENTION_KEYS:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                self.assertEqual(baseline['split.strategy'], 'compound_cold_start')
                self.assertEqual(candidate['split.strategy'], 'compound_cold_start')
                self.assertEqual(baseline['evaluation.fold.limit'], '1')
                self.assertEqual(candidate['evaluation.fold.limit'], '1')
                self.assertEqual(baseline['evaluation.outer.test'], 'False')
                self.assertEqual(candidate['evaluation.outer.test'], 'False')
                self.assertEqual(baseline['attention.max.nodes'], '0')
                self.assertEqual(candidate['attention.max.nodes'], '0')
                self.assertEqual(baseline['hyperedge.attention'], 'False')

                settings = resolve_hyperedge_attention(candidate)
                self.assertTrue(settings['enabled'])
                self.assertTrue(settings['hc_enabled'])
                self.assertTrue(settings['pd_enabled'])
                self.assertEqual(settings['temperature'], 1.0)
                self.assertEqual(settings['prior_scale'], 0.1)


if __name__ == '__main__':
    unittest.main()
