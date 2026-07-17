import unittest

import numpy as np

from util.model_components import (
    context_interaction_pair_scores,
    resolve_support_router,
)
from util.support_router import (
    monotonic_context_gate,
    select_pseudo_cold_compounds,
)


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class SupportRouterTest(unittest.TestCase):
    def test_configuration_defaults_off_and_validates_enabled_settings(self):
        defaults = resolve_support_router(DummyConf({}))
        self.assertFalse(defaults['enabled'])
        self.assertEqual(defaults['mode'], 'monotonic_residual')
        self.assertAlmostEqual(defaults['pseudo_cold_ratio'], 0.1)

        enabled = resolve_support_router(DummyConf({
            'support.router': 'True',
            'support.router.pseudo.cold.ratio': '0.2',
            'support.router.seed': '7',
            'support.router.initial.slope': '0.5',
        }))
        self.assertTrue(enabled['enabled'])
        self.assertEqual(enabled['seed'], 7)
        self.assertAlmostEqual(enabled['initial_slope'], 0.5)

        with self.assertRaisesRegex(ValueError, 'pseudo.cold.ratio'):
            resolve_support_router(DummyConf({
                'support.router': 'True',
                'support.router.pseudo.cold.ratio': '0',
            }))

    def test_pseudo_cold_selection_is_deterministic_and_compound_level(self):
        records = []
        for compound_index in range(10):
            records.extend([
                ['c%d' % compound_index, 'p0', 1.0],
                ['c%d' % compound_index, 'p1', 0.0],
            ])
        eligible = {'c%d' % index for index in range(10)}
        first = select_pseudo_cold_compounds(records, eligible, ratio=0.2, seed=9)
        second = select_pseudo_cold_compounds(
            list(reversed(records)), eligible, ratio=0.2, seed=9
        )

        self.assertEqual(first, second)
        self.assertEqual(first['selected_count'], 2)
        self.assertEqual(first['excluded_positive_edges'], 2)
        self.assertEqual(len(first['assignments_sha256']), 64)

    def test_monotonic_gate_is_one_at_zero_and_decreases_with_support(self):
        gates = monotonic_context_gate(
            degrees=[0, 1, 3, 7],
            context_available=[1, 1, 1, 1],
            slope=1.0,
        )
        np.testing.assert_allclose(gates, [1.0, 0.5, 0.25, 0.125])
        self.assertTrue(np.all(np.diff(gates) < 0))
        self.assertEqual(
            monotonic_context_gate([0], [0], slope=1.0)[0], 0.0
        )

    def test_pair_scores_scale_only_the_herb_protein_residual(self):
        compound = np.asarray([[1.0, 0.0]], dtype=np.float64)
        protein = np.asarray([[1.0, 1.0]], dtype=np.float64)
        compound_context = np.asarray([[0.0, 1.0]], dtype=np.float64)
        protein_context = np.zeros_like(protein)
        common = dict(
            compound_embeddings=compound,
            protein_embeddings=protein,
            compound_contexts=compound_context,
            protein_contexts=protein_context,
            compound_indices=[0],
            protein_indices=[0],
            compound_disease_weight=np.zeros(2),
            herb_protein_weight=np.ones(2),
            herb_disease_weight=np.zeros(2),
            enabled_terms={
                'compound_disease': False,
                'herb_protein': True,
                'herb_disease': False,
            },
        )
        base = context_interaction_pair_scores(
            herb_protein_scale=[0.0], **common
        )
        full = context_interaction_pair_scores(
            herb_protein_scale=[1.0], **common
        )
        half = context_interaction_pair_scores(
            herb_protein_scale=[0.5], **common
        )

        np.testing.assert_allclose(base, [1.0])
        np.testing.assert_allclose(full, [2.0])
        np.testing.assert_allclose(half, [1.5])


if __name__ == '__main__':
    unittest.main()
