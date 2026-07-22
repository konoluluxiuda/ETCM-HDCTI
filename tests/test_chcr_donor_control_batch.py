import unittest
from pathlib import Path

from tools.run_chcr_donor_control_batch import (
    report_matches_job,
    summarize_datasets,
)


def row(slug, fold, decision, drop=0.02, margin=1.5, win=0.75):
    return {
        'dataset': slug,
        'slug': slug,
        'fold': fold,
        'decision': decision,
        'degree_control_AUPR_drop': drop,
        'degree_control_positive_margin': margin,
        'degree_control_pair_win_rate': win,
        'disjoint_AUPR_drop': drop,
        'disjoint_positive_margin': margin,
        'disjoint_pair_win_rate': win,
    }


class ChcrDonorControlBatchTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            'fold_count': 5,
            'draws': 20,
            'counterfactual_seed': 42026,
            'minimum_supported_folds_per_dataset': 4,
            'minimum_dataset_mean_aupr_drop': 0.001,
            'minimum_dataset_mean_pair_win_rate': 0.60,
            'datasets': [{'name': 'Test', 'slug': 'test'}],
        }

    def test_dataset_passes_with_four_supported_folds(self):
        supported = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_confirmed'
        )
        rows = [row('test', fold, supported) for fold in range(1, 5)]
        rows.append(row(
            'test', 5, 'does_not_support_context_specificity_beyond_degree'
        ))

        summary = summarize_datasets(rows, self.manifest)[0]

        self.assertEqual(summary['supported_folds'], 4)
        self.assertEqual(summary['verdict'], 'PASS')

    def test_dataset_is_pending_until_all_folds_exist(self):
        decision = (
            'supports_context_specificity_beyond_degree_'
            'disjoint_confirmed'
        )
        rows = [row('test', fold, decision) for fold in range(1, 5)]

        summary = summarize_datasets(rows, self.manifest)[0]

        self.assertEqual(summary['verdict'], 'PENDING')

    def test_report_match_checks_frozen_inputs(self):
        job = {
            'fold': 2,
            'config_sha256': 'config-hash',
            'checkpoint': Path('/tmp/model.ckpt'),
        }
        report = {
            'metadata': {
                'evaluation_type': 'chcr_donor_control_pure_inference',
                'fold': 2,
                'draws': 20,
                'counterfactual_seed': 42026,
                'protocol': {'config_sha256': 'config-hash'},
                'checkpoint': {'prefix': '/tmp/model.ckpt'},
            }
        }

        self.assertTrue(report_matches_job(report, job, self.manifest))
        report['metadata']['draws'] = 10
        self.assertFalse(report_matches_job(report, job, self.manifest))


if __name__ == '__main__':
    unittest.main()
