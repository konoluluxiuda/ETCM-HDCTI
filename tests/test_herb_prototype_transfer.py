import unittest
import numpy as np
import scipy.sparse as sp

from util.herb_prototype_transfer import (
    build_support_calibrated_herb_prototypes,
    support_calibrated_herb_prototype_scores,
)
from util.model_components import (
    context_interaction_pair_scores,
    resolve_herb_prototype_transfer,
)


class DummyConf(object):
    def __init__(self, values):
        self.values = dict(values)

    def __getitem__(self, key):
        return self.values[key]

    def contains(self, key):
        return key in self.values


class HerbPrototypeTransferTest(unittest.TestCase):
    def setUp(self):
        # h0 contains c0, c1 and cold c2; h1 contains unsupported c4 only.
        self.hc = sp.csr_matrix(
            np.asarray([
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ], dtype=np.float32)
        )
        # c0->p0, c1->p1 and c3->p1. c2/c4 have no training C-P support.
        self.cp = sp.csr_matrix(
            np.asarray([
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
            ], dtype=np.float32)
        )
        self.prototypes = build_support_calibrated_herb_prototypes(
            self.hc, self.cp
        )
        sentinel = self.hc.shape[0]
        self.compound_herb_indices = np.asarray([
            [0], [0], [0], [sentinel], [1],
        ], dtype=np.int32)
        self.compound_herb_mask = np.asarray([
            [1], [1], [1], [0], [1],
        ], dtype=np.float32)

    def scores(self, compounds, proteins, diagnostics=False):
        return support_calibrated_herb_prototype_scores(
            self.prototypes,
            self.compound_herb_indices,
            self.compound_herb_mask,
            compounds,
            proteins,
            prior_strength=1.0,
            return_diagnostics=diagnostics,
        )

    def test_loco_removes_candidate_self_positive(self):
        score = self.scores([0], [0])[0]
        # Global p0 prevalence is 1/3. After removing c0, h0 has only c1,
        # which does not support p0, so the residual must be negative.
        self.assertAlmostEqual(score, -1.0 / 6.0, places=6)

    def test_cold_compound_receives_neighbor_only_target_evidence(self):
        score = self.scores([2], [0])[0]
        # c2 contributes no C-P edge. Its h0 neighbors provide one p0 hit
        # among two supported compounds, above the global 1/3 prevalence.
        self.assertAlmostEqual(score, 1.0 / 9.0, places=6)

    def test_no_supported_herb_neighbor_backs_off_exactly_to_zero(self):
        scores, diagnostics = self.scores([4], [0], diagnostics=True)
        self.assertEqual(float(scores[0]), 0.0)
        self.assertEqual(diagnostics['evidence_pairs'], 0)
        self.assertEqual(diagnostics['evidence_coverage'], 0.0)

    def test_training_membership_stays_sparse_for_large_entity_universe(self):
        hc = sp.csr_matrix(
            ([1.0, 1.0], ([0, 1], [0, 24999])),
            shape=(2, 25000),
            dtype=np.float32,
        )
        cp = sp.csr_matrix(
            ([1.0, 1.0], ([0, 24999], [0, 17999])),
            shape=(25000, 18000),
            dtype=np.float32,
        )
        prototypes = build_support_calibrated_herb_prototypes(hc, cp)

        self.assertNotIn('cp_membership', prototypes)
        self.assertEqual(prototypes['training_edge_keys'].size, cp.nnz)
        self.assertLess(prototypes['training_edge_keys'].nbytes, 1024)

    def test_pair_scoring_adds_only_scaled_prototype_residual(self):
        compound = np.asarray([[1.0, 0.0]], dtype=np.float32)
        protein = np.asarray([[0.0, 1.0]], dtype=np.float32)
        contexts = np.zeros_like(compound)
        base = context_interaction_pair_scores(
            compound,
            protein,
            contexts,
            contexts,
            [0],
            [0],
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            enabled_terms={
                'compound_disease': False,
                'herb_protein': False,
                'herb_disease': False,
            },
            herb_prototype_residual=[0.25],
            herb_prototype_scale=[2.0],
        )
        np.testing.assert_allclose(base, [0.5])

    def test_config_is_off_by_default_and_requires_pagerank_replacement(self):
        self.assertEqual(resolve_herb_prototype_transfer(DummyConf({})), {
            'enabled': False,
            'mode': 'support_calibrated_loco',
            'prior_strength': 1.0,
            'replace_compound_pagerank': False,
        })
        enabled = resolve_herb_prototype_transfer(DummyConf({
            'herb.prototype.transfer': 'True',
            'herb.prototype.prior': '1.0',
            'herb.prototype.replace.compound.pagerank': 'True',
        }))
        self.assertTrue(enabled['enabled'])
        self.assertTrue(enabled['replace_compound_pagerank'])
        with self.assertRaisesRegex(ValueError, 'replace compound PageRank'):
            resolve_herb_prototype_transfer(DummyConf({
                'herb.prototype.transfer': 'True',
                'herb.prototype.replace.compound.pagerank': 'False',
            }))

if __name__ == '__main__':
    unittest.main()
