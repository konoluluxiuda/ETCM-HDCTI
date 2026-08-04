import unittest

from tools.audit_four_state_cc_residual import (
    audit_routes,
    metrics_with_cc,
    parse_alphas,
)
from tools.evaluate_four_state_checkpoint import STATE_NAMES


def metrics(aupr):
    values = {
        name: {
            'AUPR': aupr,
            'AUC': aupr - 0.01,
            'records': 10,
        }
        for name in STATE_NAMES
    }
    values['macro'] = {
        'AUPR': aupr,
        'AUC': aupr - 0.01,
    }
    return values


class FourStateColdColdResidualAuditTest(unittest.TestCase):
    def test_parse_alphas_deduplicates_in_declared_order(self):
        self.assertEqual(
            parse_alphas('0,0.25,0.5,0.25,1'),
            [0.0, 0.25, 0.5, 1.0],
        )

    def test_parse_alphas_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            parse_alphas('0,-0.25')

    def test_metrics_with_cc_recomputes_macro(self):
        candidate = metrics(0.70)
        updated = metrics_with_cc(
            candidate,
            {'AUPR': 0.50, 'AUC': 0.49, 'records': 10},
        )
        self.assertAlmostEqual(updated['macro']['AUPR'], 0.65)
        self.assertAlmostEqual(updated['macro']['AUC'], 0.64)
        self.assertEqual(updated['cold_cold']['records'], 10)

    def test_audit_routes_preserves_gate_failure(self):
        baseline = metrics(0.60)
        candidate = metrics(0.70)
        rows = audit_routes(
            candidate,
            baseline,
            {'AUPR': 0.50, 'AUC': 0.49, 'records': 10},
            [
                (
                    0.0,
                    {'AUPR': 0.60, 'AUC': 0.59, 'records': 10},
                ),
            ],
        )
        self.assertEqual(rows[0]['route'], 'hctx_dctx_only')
        self.assertFalse(rows[0]['comparison']['passed'])
        self.assertEqual(rows[1]['route'], 'base_only')
        self.assertTrue(rows[1]['comparison']['passed'])


if __name__ == '__main__':
    unittest.main()
