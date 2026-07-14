import unittest

import numpy as np

from util.checkpoint_ranking import (
    assess_pu_evidence,
    evaluate_fixed_candidate_ranking,
)


class CheckpointRankingTest(unittest.TestCase):
    def test_fixed_candidates_filter_train_positives_and_keep_test_positives(self):
        proteins = ['p0', 'p1', 'p2', 'p3']
        outer_train = [
            ['c0', 'p0', 1.0],
            ['c0', 'p3', 0.0],
            ['c1', 'p1', 1.0],
        ]
        outer_test = [
            ['c0', 'p1', 1.0],
            ['c0', 'p2', 0.0],
            ['c1', 'p2', 1.0],
            ['c1', 'p3', 0.0],
        ]
        scores = {
            ('c0', 'p1'): 0.9,
            ('c0', 'p2'): 0.8,
            ('c0', 'p3'): 0.1,
            ('c1', 'p0'): 0.8,
            ('c1', 'p2'): 0.7,
            ('c1', 'p3'): 0.1,
        }

        def scorer(compounds, targets):
            return np.asarray([
                scores[(compound, target)]
                for compound, target in zip(compounds, targets)
            ])

        result = evaluate_fixed_candidate_ranking(
            proteins, outer_train, outer_test, scorer, ks=(1, 2), export_top=2
        )

        metrics = result['metrics']
        self.assertEqual(metrics['evaluated_compounds'], 2)
        self.assertEqual(metrics['candidate_pairs'], 6)
        self.assertAlmostEqual(metrics['MRR'], 0.75)
        self.assertAlmostEqual(metrics['Precision@1'], 0.5)
        self.assertAlmostEqual(metrics['Recall@1'], 0.5)
        self.assertAlmostEqual(metrics['Recall@2'], 1.0)
        self.assertAlmostEqual(metrics['Enrichment@1'], 1.5)
        self.assertEqual(result['protocol']['non_positive_status'], 'unlabeled')
        c0_rows = [
            row for row in result['top_candidates'] if row['compound_id'] == 'c0'
        ]
        self.assertNotIn('p0', {row['protein_id'] for row in c0_rows})
        self.assertEqual(c0_rows[0]['label_status'], 'held_out_positive')

    def test_ties_are_broken_by_protein_id(self):
        def scorer(compounds, targets):
            return np.ones(len(targets), dtype=np.float32)

        result = evaluate_fixed_candidate_ranking(
            ['p2', 'p0', 'p1'],
            [],
            [['c0', 'p1', 1.0]],
            scorer,
            ks=(1,),
            export_top=3,
        )
        ranked = [row['protein_id'] for row in result['top_candidates']]
        self.assertEqual(ranked, ['p0', 'p1', 'p2'])
        self.assertEqual(result['per_compound'][0]['first_positive_rank'], 2)

    def test_pu_assessment_does_not_claim_necessity(self):
        assessment = assess_pu_evidence(
            {
                'MRR': 0.1,
                'Recall@50': 0.3,
                'held_out_positive_prevalence': 0.001,
            },
            sampled_metrics={'AUPR': 0.98},
        )
        self.assertEqual(
            assessment['necessity_conclusion'],
            'not_identifiable_from_internal_labels_alone',
        )
        self.assertEqual(
            assessment['trial_priority'],
            'external_validation_before_pu_pilot',
        )


if __name__ == '__main__':
    unittest.main()
