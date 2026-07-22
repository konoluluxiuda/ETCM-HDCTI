import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.summarize_hctx_ablation import (
    decision,
    metric_mean,
    parse_fold_metrics,
    summarize,
)
from tools.validate_hctx_ablation_configs import sha256_file, validate_config_pair


class HctxAblationTest(unittest.TestCase):
    def test_summarizer_can_start_as_a_direct_script(self):
        repository_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                str(repository_root / 'tools' / 'summarize_hctx_ablation.py'),
                '--help',
            ],
            cwd=str(repository_root),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--no-context-results', completed.stdout)

    def _config_pair(self):
        shared = {
            'experiment.protocol': 'strict',
            'split.strategy': 'pair_stratified',
            'split.reuse': 'True',
            'evaluation.setup': '-cv 5',
            'evaluation.outer.test': 'True',
            'early.stopping': 'True',
            'pair.decoder': 'dot',
            'counterfactual.context': 'False',
            'attention.max.nodes': '0',
            'random.seed': '2026',
            'split.seed': '2026',
            'validation.seed': '102026',
            'batch_size': '2000',
        }
        no_context = dict(shared)
        no_context.update({
            'model.variant': 'NoContext',
            'context.interaction': 'False',
            'context.herb_protein': 'False',
        })
        hctx = dict(shared)
        hctx.update({
            'model.variant': 'Hctx-P',
            'context.interaction': 'True',
            'context.herb_protein': 'True',
        })
        return no_context, hctx

    def test_config_pair_changes_only_frozen_hctx_fields(self):
        no_context, hctx = self._config_pair()

        differences = validate_config_pair(
            no_context,
            hctx,
            {'model.variant', 'context.interaction', 'context.herb_protein'},
        )

        self.assertEqual(set(differences), {
            'model.variant', 'context.interaction', 'context.herb_protein'
        })

    def test_config_pair_rejects_unexpected_training_change(self):
        no_context, hctx = self._config_pair()
        no_context['batch_size'] = '500'

        with self.assertRaisesRegex(ValueError, 'Unexpected config differences'):
            validate_config_pair(
                no_context,
                hctx,
                {'model.variant', 'context.interaction',
                 'context.herb_protein'},
            )

    def test_fold_parser_ignores_cross_validation_summary(self):
        log = '''
Predicting [1]...
AUC: 0.81
AUPR: 0.71
Predicting [2]...
AUC: 0.82
AUPR: 0.72
The result of 5-fold cross validation:
AUC:0.99(+-0.01)
AUPR:0.98(+-0.01)
'''
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / 'training.log'
            path.write_text(log, encoding='utf-8')

            folds = parse_fold_metrics(path)

        self.assertEqual(sorted(folds), [1, 2])
        self.assertEqual(folds[1]['AUPR'], 0.71)
        self.assertEqual(folds[2]['AUPR'], 0.72)

    def test_metric_mean_accepts_repository_summary_format(self):
        self.assertAlmostEqual(metric_mean('0.984146(+-0.001740)'), 0.984146)
        self.assertAlmostEqual(metric_mean('0.984146(±0.001740)'), 0.984146)

    def test_decision_uses_all_frozen_gate_criteria(self):
        manifest = {
            'decision_gate': {
                'minimum_non_decreasing_datasets': 3,
                'minimum_macro_AUPR_delta': 0.001,
                'maximum_single_dataset_AUPR_drop': 0.003,
                'minimum_datasets_with_three_positive_folds': 3,
            }
        }
        passing_rows = [
            {'AUPR_delta': value, 'AUPR_positive_folds': folds}
            for value, folds in (
                (0.002, 4), (0.003, 5), (0.001, 3), (-0.001, 2)
            )
        ]

        passing = decision(passing_rows, manifest)
        failing = decision(
            passing_rows[:-1]
            + [{'AUPR_delta': -0.004, 'AUPR_positive_folds': 1}],
            manifest,
        )

        self.assertEqual(passing['verdict'], 'PASS')
        self.assertEqual(failing['verdict'], 'NO-GO')
        self.assertFalse(failing['criteria']['maximum_single_dataset_drop'])

    def test_summarizer_writes_paired_outputs_end_to_end(self):
        fields = [
            'dataset', 'variant', 'config', 'exit_code', 'status',
            'started_at', 'finished_at', 'duration_seconds', 'AUC', 'AUPR',
            'Recall', 'Precision', 'F1-score', 'log', 'config_sha256',
        ]
        gate = {
            'minimum_non_decreasing_datasets': 3,
            'minimum_macro_AUPR_delta': 0.001,
            'maximum_single_dataset_AUPR_drop': 0.003,
            'minimum_datasets_with_three_positive_folds': 3,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            no_rows = []
            hctx_rows = []
            datasets = []
            for index in range(4):
                name = 'Dataset-%d' % (index + 1)
                no_config = root / ('no_%d.conf' % index)
                hctx_config = root / ('hctx_%d.conf' % index)
                no_config.write_text('variant=no\n', encoding='utf-8')
                hctx_config.write_text('variant=hctx\n', encoding='utf-8')
                no_log = root / ('no_%d.log' % index)
                hctx_log = root / ('hctx_%d.log' % index)
                no_lines = []
                hctx_lines = []
                for fold in range(1, 6):
                    no_lines.extend([
                        'Predicting [%d]...' % fold,
                        'AUC: 0.800000',
                        'AUPR: 0.800000',
                        'Recall: 0.700000',
                        'Precision: 0.700000',
                        'F1-score: 0.700000',
                    ])
                    hctx_lines.extend([
                        'Predicting [%d]...' % fold,
                        'AUC: 0.802000',
                        'AUPR: 0.802000',
                        'Recall: 0.702000',
                        'Precision: 0.702000',
                        'F1-score: 0.702000',
                    ])
                no_log.write_text('\n'.join(no_lines) + '\n', encoding='utf-8')
                hctx_log.write_text(
                    '\n'.join(hctx_lines) + '\n', encoding='utf-8'
                )
                no_hash = sha256_file(no_config)
                hctx_hash = sha256_file(hctx_config)
                common = {
                    'dataset': name,
                    'exit_code': '0',
                    'status': 'OK',
                    'started_at': '',
                    'finished_at': '',
                    'duration_seconds': '1',
                    'Recall': '0.700000(+-0.000000)',
                    'Precision': '0.700000(+-0.000000)',
                    'F1-score': '0.700000(+-0.000000)',
                }
                no_rows.append(dict(common, **{
                    'variant': 'NoContext',
                    'config': str(no_config),
                    'AUC': '0.800000(+-0.000000)',
                    'AUPR': '0.800000(+-0.000000)',
                    'log': str(no_log),
                    'config_sha256': no_hash,
                }))
                hctx_rows.append(dict(common, **{
                    'variant': 'Hctx-P',
                    'config': str(hctx_config),
                    'AUC': '0.802000(+-0.000000)',
                    'AUPR': '0.802000(+-0.000000)',
                    'Recall': '0.702000(+-0.000000)',
                    'Precision': '0.702000(+-0.000000)',
                    'F1-score': '0.702000(+-0.000000)',
                    'log': str(hctx_log),
                    'config_sha256': hctx_hash,
                }))
                datasets.append({
                    'name': name,
                    'no_context_config': str(no_config),
                    'no_context_sha256': no_hash,
                    'hctx_config': str(hctx_config),
                    'hctx_sha256': hctx_hash,
                })

            no_results = root / 'no_results.tsv'
            hctx_results = root / 'hctx_results.tsv'
            for path, rows in ((no_results, no_rows), (hctx_results, hctx_rows)):
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=fields, delimiter='\t'
                    )
                    writer.writeheader()
                    writer.writerows(rows)
            manifest = {
                'reference_results': str(hctx_results),
                'decision_gate': gate,
                'datasets': datasets,
            }
            output_dir = root / 'paired'
            with patch(
                    'tools.summarize_hctx_ablation.validate_manifest',
                    return_value=(manifest, [])):
                verdict = summarize(
                    root / 'manifest.json', no_results, output_dir
                )

            self.assertEqual(verdict['verdict'], 'PASS')
            self.assertTrue((output_dir / 'paired_results.tsv').is_file())
            self.assertTrue((output_dir / 'paired_folds.tsv').is_file())
            self.assertEqual(
                json.loads((output_dir / 'decision.json').read_text(
                    encoding='utf-8'
                ))['verdict'],
                'PASS',
            )
            summary = (output_dir / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('Macro AUPR delta：`+0.002000`', summary)
            self.assertIn('Verdict: **PASS**', summary)


if __name__ == '__main__':
    unittest.main()
