import unittest

from tools.calibrate_checkpoint_folds import (
    METRIC_NAMES,
    build_markdown,
    summarize_fold_metrics,
)


class CheckpointFoldCalibrationTest(unittest.TestCase):
    def test_summary_uses_sample_standard_deviation(self):
        fold_results = []
        for value in (0.8, 1.0):
            metrics = {metric_name: value for metric_name in METRIC_NAMES}
            fold_results.append({'calibrated_metrics': metrics})

        summary = summarize_fold_metrics(fold_results, 'calibrated_metrics')

        self.assertAlmostEqual(summary['AUPR']['mean'], 0.9)
        self.assertAlmostEqual(summary['AUPR']['std'], 0.14142135623730948)

    def test_markdown_contains_fixed_and_calibrated_results(self):
        metrics = {metric_name: 0.75 for metric_name in METRIC_NAMES}
        summary = {
            metric_name: {'mean': 0.75, 'std': 0.0}
            for metric_name in METRIC_NAMES
        }
        payload = {
            'config_path': '/tmp/example.conf',
            'split_strategy': 'compound_cold_start',
            'fold_results': [{
                'fold': 1,
                'threshold': 0.25,
                'validation_metrics': dict(metrics),
                'fixed_metrics': dict(metrics),
                'calibrated_metrics': dict(metrics),
            }],
            'fixed_summary': summary,
            'calibrated_summary': summary,
        }

        markdown = build_markdown(payload)

        self.assertIn('inner-validation F1', markdown)
        self.assertIn('Calibrated Threshold Summary', markdown)
        self.assertIn('0.250000', markdown)


if __name__ == '__main__':
    unittest.main()
