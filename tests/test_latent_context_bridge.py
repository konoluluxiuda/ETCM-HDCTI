import unittest

import numpy as np

from util.latent_context_bridge import (
    degree_stratified_permutation,
    low_rank_bilinear_residual,
    pair_alignment_scores,
    select_positive_residual,
    stratified_half_split,
    train_low_rank_bilinear_probe,
)


class LatentContextBridgeTest(unittest.TestCase):
    def test_pair_alignment_uses_incident_context_sets(self):
        herbs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        diseases = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
        result = pair_alignment_scores(
            herbs,
            diseases,
            {0: np.asarray([0, 1])},
            {0: np.asarray([0, 1])},
            [0],
            [0],
            top_ks=(1, 3),
        )
        self.assertTrue(result['covered'][0])
        self.assertAlmostEqual(result['scores']['top1_mean'][0], 1.0)
        self.assertAlmostEqual(result['scores']['top3_mean'][0], 0.8)
        self.assertEqual(result['herb_degrees'][0], 2)
        self.assertEqual(result['disease_degrees'][0], 2)

    def test_nested_selection_and_permutation_are_deterministic(self):
        labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        baseline = np.zeros(8, dtype=np.float64)
        feature = np.asarray([2.0, 1.8, 1.6, 1.4, -1.4, -1.6, -1.8, -2.0])
        selection, audit = stratified_half_split(labels, seed=11)
        report = select_positive_residual(
            labels,
            baseline,
            {'top1_mean': feature},
            selection,
            audit,
            alphas=(0.0, 1.0),
        )
        self.assertEqual(report['selected']['alpha'], 1.0)
        self.assertGreater(report['audit_delta']['AUPR'], 0.0)
        permutation = degree_stratified_permutation(
            labels[audit],
            baseline[audit],
            report['audit_feature'],
            report['selected']['alpha'],
            ['same'] * len(audit),
            draws=20,
            seed=17,
        )
        self.assertEqual(permutation['draws'], 20)
        self.assertGreaterEqual(permutation['one_sided_p'], 0.0)
        self.assertLessEqual(permutation['one_sided_p'], 1.0)

    def test_low_rank_probe_learns_cross_space_signal(self):
        rng = np.random.RandomState(7)
        herb = rng.normal(size=(40, 4))
        disease = herb.copy()
        disease[20:] *= -1.0
        labels = np.asarray([1] * 20 + [0] * 20, dtype=np.int32)
        baseline = np.zeros(40, dtype=np.float64)
        selection, audit = stratified_half_split(labels, seed=19)
        probe = train_low_rank_bilinear_probe(
            herb[selection],
            disease[selection],
            labels[selection],
            baseline[selection],
            rank=2,
            steps=300,
            learning_rate=0.02,
            seed=23,
        )
        residual = low_rank_bilinear_residual(
            herb[audit], disease[audit], probe['left'], probe['right']
        )
        self.assertLess(probe['final_loss'], probe['initial_loss'])
        self.assertGreater(np.mean(residual[labels[audit] == 1]), 0.0)
        self.assertLess(np.mean(residual[labels[audit] == 0]), 0.0)


if __name__ == '__main__':
    unittest.main()
