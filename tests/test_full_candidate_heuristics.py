import unittest

from tools.evaluate_full_candidate_heuristics import evaluate_fold


class FullCandidateHeuristicsTest(unittest.TestCase):
    def test_side_information_protein_is_ranked_as_unlabeled_candidate(self):
        hc_pairs = [
            ('h1', 'cold'),
            ('h1', 'train'),
        ]
        pd_pairs = [('p3', 'd1')]
        assignments = [
            ('cold', 'p1', 1, 0),
            ('cold', 'p2', 0, 0),
            ('train', 'p1', 1, 1),
            ('train', 'p2', 0, 1),
        ]
        result = evaluate_fold(
            hc_pairs,
            pd_pairs,
            assignments,
            fold=0,
            prior_strength=1.0,
            ks=[1, 2],
        )
        self.assertEqual(result['entity_counts']['proteins'], 3)
        for method in result['methods'].values():
            metrics = method['metrics']
            self.assertEqual(metrics['candidate_pairs'], 3)
            self.assertEqual(metrics['held_out_positive_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
