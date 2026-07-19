import unittest
from pathlib import Path

from util.config import ModelConf
from util.model_components import resolve_inductive_context


class InductiveContextConfigTest(unittest.TestCase):
    DATASETS = (
        'tcmsuite',
        'tcmsp',
        'symmap',
        'etcm_mention10',
    )
    PILOT_ONLY_KEYS = {
        'context.mask.training',
        'support.router',
        'global.token.attention',
        'inductive.context',
        'inductive.context.suppress.base.zero.support',
        'inductive.context.self.excluded',
    }

    def test_sdis_variants_match_no_dense_cold_start_baselines(self):
        repository_root = Path(__file__).resolve().parents[1]
        for dataset in self.DATASETS:
            baseline = ModelConf(str(
                repository_root / 'configs' /
                ('HDCTI_%s_cold_start_no_dense_herb_only_pilot.conf' % dataset)
            ))
            for suffix, self_excluded in (
                    ('sdis_pilot', False),
                    ('sdis_self_excluded_pilot', True)):
                with self.subTest(dataset=dataset, suffix=suffix):
                    candidate = ModelConf(str(
                        repository_root / 'configs' /
                        ('HDCTI_%s_cold_start_%s.conf' % (dataset, suffix))
                    ))
                    baseline_values = dict(baseline.config)
                    candidate_values = dict(candidate.config)
                    baseline_values.pop('model.variant')
                    candidate_values.pop('model.variant')
                    for key in self.PILOT_ONLY_KEYS:
                        baseline_values.pop(key, None)
                        candidate_values.pop(key, None)
                    self.assertEqual(candidate_values, baseline_values)

                    settings = resolve_inductive_context(candidate)
                    self.assertTrue(settings['enabled'])
                    self.assertTrue(settings['suppress_base_zero_support'])
                    self.assertEqual(settings['self_excluded'], self_excluded)
                    self.assertEqual(candidate['split.strategy'], 'compound_cold_start')
                    self.assertEqual(candidate['evaluation.fold.limit'], '1')
                    self.assertEqual(candidate['evaluation.outer.test'], 'False')
                    self.assertEqual(candidate['attention.max.nodes'], '0')
                    self.assertEqual(candidate['counterfactual.context'], 'False')
                    self.assertEqual(candidate['hyperedge.attention'], 'False')

    def test_full_sdis_configs_are_paired_with_full_herb_only_baselines(self):
        repository_root = Path(__file__).resolve().parents[1]
        inductive_keys = {
            'inductive.context',
            'inductive.context.suppress.base.zero.support',
            'inductive.context.self.excluded',
        }
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                baseline = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_no_dense_herb_only_full.conf' % dataset)
                ))
                candidate = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_sdis_full.conf' % dataset)
                ))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in inductive_keys:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)

                self.assertFalse(resolve_inductive_context(baseline)['enabled'])
                settings = resolve_inductive_context(candidate)
                self.assertTrue(settings['enabled'])
                self.assertTrue(settings['suppress_base_zero_support'])
                self.assertFalse(settings['self_excluded'])
                self.assertFalse(candidate.contains('evaluation.fold.limit'))
                self.assertEqual(candidate['evaluation.outer.test'], 'True')
                self.assertEqual(candidate['attention.max.nodes'], '0')

    def test_sdis_chcr_configs_only_enable_frozen_counterfactual_settings(self):
        repository_root = Path(__file__).resolve().parents[1]
        counterfactual_keys = {
            'counterfactual.context',
            'counterfactual.match',
            'counterfactual.weight',
            'counterfactual.margin',
            'counterfactual.draws',
            'counterfactual.seed',
        }
        expected = {
            'counterfactual.context': 'True',
            'counterfactual.match': 'exact_hc_degree_disjoint',
            'counterfactual.weight': '0.05',
            'counterfactual.margin': '0.2',
            'counterfactual.draws': '20',
            'counterfactual.seed': '42026',
        }
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                sdis = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_sdis_full.conf' % dataset)
                ))
                combined = ModelConf(str(
                    repository_root / 'configs' /
                    ('HDCTI_%s_cold_start_sdis_chcr_full.conf' % dataset)
                ))
                sdis_values = dict(sdis.config)
                combined_values = dict(combined.config)
                sdis_values.pop('model.variant')
                combined_values.pop('model.variant')
                for key in counterfactual_keys:
                    sdis_values.pop(key, None)
                    combined_values.pop(key, None)
                self.assertEqual(combined_values, sdis_values)
                self.assertEqual(
                    {key: combined[key] for key in counterfactual_keys},
                    expected,
                )


if __name__ == '__main__':
    unittest.main()
