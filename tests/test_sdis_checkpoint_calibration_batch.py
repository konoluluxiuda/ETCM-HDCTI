import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.run_sdis_checkpoint_calibration import (
    build_markdown,
    extract_checkpoint_prefixes,
    load_report,
    validate_report_against_training,
    paired_deltas,
)
from tools.run_sdis_checkpoint_calibration import CalibrationJob
from tools.analyze_context_subgroups import score_snapshot


class SdisCheckpointCalibrationBatchTest(unittest.TestCase):
    def test_extracts_five_checkpoint_prefixes_in_fold_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            lines = []
            expected = []
            for fold in range(1, 6):
                prefix = root / 'saved_model' / ('fold_%d' % fold) / 'hdcti_model.ckpt'
                prefix.parent.mkdir(parents=True)
                Path(str(prefix) + '.index').write_text('index', encoding='utf-8')
                Path(str(prefix) + '.data-00000-of-00001').write_text(
                    'data', encoding='utf-8'
                )
                expected.append(prefix.resolve())
                lines.append('模型权重保存成功: %s' % prefix)
            log_path = root / 'training.log'
            log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

            actual = extract_checkpoint_prefixes(log_path, repository_root=root)

            self.assertEqual(actual, expected)

    def test_report_rejects_threshold_dependent_aupr_change(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / 'report.json'
            metric = {'mean': 0.8, 'std': 0.0}
            payload = {
                'fold_results': [{} for _ in range(5)],
                'fixed_summary': {'AUC': metric, 'AUPR': metric},
                'calibrated_summary': {
                    'AUC': metric,
                    'AUPR': {'mean': 0.7, 'std': 0.0},
                },
            }
            report_path.write_text(json.dumps(payload), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'AUPR changed'):
                load_report(report_path)

    def test_paired_delta_and_markdown_use_calibrated_f1(self):
        baseline = {
            'dataset': 'Example', 'variant': 'HerbOnly',
            'fixed_auc': 0.7, 'fixed_aupr': 0.6,
            'calibrated_recall': 0.5, 'calibrated_precision': 0.6,
            'calibrated_f1_score': 0.55,
        }
        candidate = {
            'dataset': 'Example', 'variant': 'SDIS',
            'fixed_auc': 0.8, 'fixed_aupr': 0.7,
            'calibrated_recall': 0.6, 'calibrated_precision': 0.7,
            'calibrated_f1_score': 0.65,
        }

        delta = paired_deltas([baseline, candidate])[0]

        self.assertAlmostEqual(delta['aupr'], 0.1)
        self.assertAlmostEqual(delta['f1_score'], 0.1)
        markdown = build_markdown([], '/tmp/source')
        self.assertIn('inner-validation', markdown)
        self.assertIn('训练与优化器更新：`0`', markdown)

    def test_snapshot_scoring_applies_inductive_base_gate(self):
        snapshot = {
            'compound_map': {'c0': 0, 'c1': 1},
            'protein_map': {'p0': 0, 'p1': 1},
            'compound': np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            'protein': np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            'compound_context': np.zeros((2, 2)),
            'protein_context': np.zeros((2, 2)),
            'weights': {},
            'context_terms': {
                'compound_disease': False,
                'herb_protein': False,
                'herb_disease': False,
            },
            'pair_decoder': {'type': 'dot'},
            'inductive_base_gate': [0.0, 1.0],
            'support_context_gate': None,
        }
        records = [('c0', 'p0', 1), ('c1', 'p1', 1)]

        base_logits, total_logits = score_snapshot(
            snapshot, records, include_context=True
        )

        self.assertEqual(base_logits.tolist(), [0.0, 1.0])
        self.assertEqual(total_logits.tolist(), [0.0, 1.0])

    def test_rejects_restored_metrics_that_differ_from_training_log(self):
        job = CalibrationJob('Example', 'SDIS', 'slug', 'config', 'log')
        payload = {
            'fixed_summary': {
                'AUC': {'mean': 0.7},
                'AUPR': {'mean': 0.6},
            }
        }
        expected = {('Example', 'SDIS'): {'AUC': 0.8, 'AUPR': 0.6}}

        with self.assertRaisesRegex(ValueError, 'do not match'):
            validate_report_against_training(job, payload, expected)


if __name__ == '__main__':
    unittest.main()
