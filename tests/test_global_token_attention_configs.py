import unittest
from pathlib import Path

from util.config import ModelConf
from util.model_components import resolve_global_token_attention


class GlobalTokenAttentionConfigTest(unittest.TestCase):
    DATASETS = ('tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10')
    GLOBAL_KEYS = {
        'global.token.attention',
        'global.token.attention.mode',
        'global.token.attention.hc',
        'global.token.attention.pd',
        'global.token.attention.tokens',
        'global.token.attention.heads',
        'global.token.attention.temperature',
    }

    def test_candidates_change_only_variant_and_global_attention(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_herb_only_pilot.conf' % dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_hilga_pilot.conf' % dataset)
                ))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in self.GLOBAL_KEYS:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                self.assertEqual(candidate['attention.max.nodes'], '0')
                self.assertEqual(candidate['hyperedge.attention'], 'False')
                self.assertEqual(candidate['counterfactual.context'], 'False')
                self.assertEqual(candidate['evaluation.fold.limit'], '1')
                self.assertEqual(candidate['evaluation.outer.test'], 'False')
                settings = resolve_global_token_attention(candidate)
                self.assertTrue(settings['enabled'])
                self.assertTrue(settings['hc_enabled'])
                self.assertTrue(settings['pd_enabled'])
                self.assertEqual(settings['tokens'], 32)
                self.assertEqual(settings['heads'], 2)
                self.assertEqual(settings['temperature'], 1.0)


if __name__ == '__main__':
    unittest.main()
