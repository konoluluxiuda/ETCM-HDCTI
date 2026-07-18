import unittest
from pathlib import Path

from util.config import ModelConf


class ChcrNoDenseConfigTest(unittest.TestCase):
    DATASETS = ('tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10')
    COUNTERFACTUAL_KEYS = {
        'counterfactual.context',
        'counterfactual.match',
        'counterfactual.weight',
        'counterfactual.margin',
        'counterfactual.draws',
        'counterfactual.seed',
    }

    def test_candidates_change_only_variant_and_counterfactual_settings(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_herb_only_pilot.conf' % dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_chcr_no_dense_pilot.conf' % dataset)
                ))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in self.COUNTERFACTUAL_KEYS:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                self.assertEqual(candidate['attention.max.nodes'], '0')
                self.assertEqual(candidate['hyperedge.attention'], 'False')
                self.assertEqual(candidate['counterfactual.context'], 'True')
                self.assertEqual(candidate['counterfactual.match'],
                                 'exact_hc_degree_disjoint')
                self.assertEqual(candidate['counterfactual.weight'], '0.05')
                self.assertEqual(candidate['counterfactual.margin'], '0.2')
                self.assertEqual(candidate['counterfactual.draws'], '20')
                self.assertEqual(candidate['counterfactual.seed'], '42026')
                self.assertEqual(candidate['evaluation.fold.limit'], '1')
                self.assertEqual(candidate['evaluation.outer.test'], 'False')

    def test_full_configs_are_paired_and_enable_all_outer_folds(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_herb_only_no_dense_full.conf' %
                     dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_pair_stratified_chcr_no_dense_full.conf' % dataset)
                ))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in self.COUNTERFACTUAL_KEYS:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                self.assertFalse(baseline.contains('evaluation.fold.limit'))
                self.assertFalse(candidate.contains('evaluation.fold.limit'))
                self.assertEqual(baseline['evaluation.outer.test'], 'True')
                self.assertEqual(candidate['evaluation.outer.test'], 'True')
                self.assertEqual(baseline['attention.max.nodes'], '0')
                self.assertEqual(candidate['attention.max.nodes'], '0')
                self.assertEqual(baseline['global.token.attention'], 'False')
                self.assertEqual(candidate['global.token.attention'], 'False')
                self.assertEqual(baseline['hyperedge.attention'], 'False')
                self.assertEqual(candidate['hyperedge.attention'], 'False')
                self.assertEqual(baseline['counterfactual.context'], 'False')
                self.assertEqual(candidate['counterfactual.context'], 'True')


if __name__ == '__main__':
    unittest.main()
