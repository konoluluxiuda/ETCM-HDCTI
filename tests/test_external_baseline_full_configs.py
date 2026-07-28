import unittest
from pathlib import Path

from util.config import ModelConf


class ExternalBaselineFullConfigTest(unittest.TestCase):
    def test_full_configs_only_enable_complete_outer_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        model_prefixes = ('DualHGNN', 'LightGCNCTI', 'RGCNCTI')
        dataset_slugs = (
            'tcmsuite',
            'tcmsp',
            'symmap',
            'etcm_mention10',
        )

        for model_prefix in model_prefixes:
            for dataset_slug in dataset_slugs:
                with self.subTest(
                    model=model_prefix,
                    dataset=dataset_slug,
                ):
                    pilot_path = (
                        root / 'configs' /
                        f'{model_prefix}_{dataset_slug}_pair_stratified_pilot.conf'
                    )
                    full_path = (
                        root / 'configs' /
                        f'{model_prefix}_{dataset_slug}_pair_stratified_full.conf'
                    )
                    pilot = ModelConf(str(pilot_path))
                    full = ModelConf(str(full_path))

                    expected = dict(pilot.config)
                    expected.pop('evaluation.fold.limit')
                    expected['evaluation.outer.test'] = 'True'
                    expected['model.variant'] = expected[
                        'model.variant'
                    ].replace(
                        '_pilot_v1',
                        '_full_v1',
                    )

                    self.assertEqual(full.config, expected)
                    self.assertEqual(full['evaluation.setup'], '-cv 5')
                    self.assertEqual(full['experiment.protocol'], 'strict')
                    self.assertEqual(
                        full['split.strategy'],
                        'pair_stratified',
                    )
                    self.assertEqual(full['split.seed'], '2026')
                    self.assertEqual(full['evaluation.outer.test'], 'True')
                    self.assertFalse(full.contains('evaluation.fold.limit'))


if __name__ == '__main__':
    unittest.main()
