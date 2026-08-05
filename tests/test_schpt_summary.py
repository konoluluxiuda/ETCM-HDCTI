import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.summarize_schpt_pilot import summarize
from tools.summarize_schpt_gate1 import summarize as summarize_gate1
from tools.summarize_schpt_full import summarize as summarize_full
from util.config import ModelConf


class SchptSummaryTest(unittest.TestCase):
    def test_gate_summary_supports_direct_script_execution(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(repository_root / 'tools/summarize_schpt_gate1.py'),
                '--help',
            ],
            cwd=str(repository_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gate_uses_frozen_aupr_coverage_and_scale_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({
                'learned_scale': 0.25,
                'validation': {
                    'evidence_coverage': 0.4,
                    'mean_abs_residual': 0.1,
                },
            }), encoding='utf-8')
            baseline = root / 'baseline.log'
            baseline.write_text(
                'Validation-AUPR:0.900000(+-0.000000)\n', encoding='utf-8'
            )
            candidate = root / 'candidate.log'
            candidate.write_text(
                'Herb prototype metadata: %s\n'
                'Validation-AUPR:0.904000(+-0.000000)\n' % metadata,
                encoding='utf-8',
            )
            report = summarize(baseline, candidate)
        self.assertEqual(report['gate'], 'PASS')
        self.assertAlmostEqual(report['validation_aupr_delta'], 0.004)

    def test_four_dataset_gate_requires_cross_dataset_consistency(self):
        repository_root = Path(__file__).resolve().parents[1]
        manifest_path = repository_root / 'configs/schpt_gate1_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for index, dataset in enumerate(manifest['datasets']):
                metadata = run_dir / ('%s_metadata.json' % dataset['slug'])
                metadata.write_text(json.dumps({
                    'learned_scale': 0.2,
                    'positive_mean_residual': 0.1,
                    'negative_mean_residual': -0.01,
                    'validation': {
                        'evidence_coverage': 0.8,
                        'mean_abs_residual': 0.05,
                    },
                }), encoding='utf-8')
                (run_dir / ('%02d_%s_baseline.log' % (
                    index * 2 + 1, dataset['slug']
                ))).write_text(
                    'Validation-AUPR:0.900000(+-0.000000)\n',
                    encoding='utf-8',
                )
                (run_dir / ('%02d_%s_candidate.log' % (
                    index * 2 + 2, dataset['slug']
                ))).write_text(
                    'Herb prototype metadata: %s\n'
                    'Validation-AUPR:0.904000(+-0.000000)\n' % metadata,
                    encoding='utf-8',
                )
            report = summarize_gate1(run_dir, manifest_path)
        self.assertEqual(report['gate'], 'PASS')
        self.assertEqual(report['positive_dataset_count'], 4)
        self.assertAlmostEqual(report['mean_validation_aupr_delta'], 0.004)

    def test_four_dataset_configs_are_hash_frozen_and_pairwise_isolated(self):
        repository_root = Path(__file__).resolve().parents[1]
        manifest = json.loads((
            repository_root / 'configs/schpt_gate1_manifest.json'
        ).read_text(encoding='utf-8'))
        prototype_keys = {
            'herb.prototype.transfer',
            'herb.prototype.mode',
            'herb.prototype.prior',
            'herb.prototype.replace.compound.pagerank',
        }
        for dataset in manifest['datasets']:
            with self.subTest(dataset=dataset['name']):
                baseline_path = repository_root / dataset['baseline_config']
                candidate_path = repository_root / dataset['candidate_config']
                self.assertEqual(
                    hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    dataset['baseline_sha256'],
                )
                self.assertEqual(
                    hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    dataset['candidate_sha256'],
                )
                baseline = ModelConf(str(baseline_path))
                candidate = ModelConf(str(candidate_path))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in prototype_keys:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)
                self.assertEqual(candidate['split.seed'], '52026')
                self.assertEqual(candidate['evaluation.fold.limit'], '1')
                self.assertEqual(candidate['evaluation.outer.test'], 'False')

    def test_five_fold_confirmation_gate_parses_paired_outer_results(self):
        repository_root = Path(__file__).resolve().parents[1]
        manifest_path = repository_root / 'configs/schpt_full_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        def log_text(aupr, metadata_paths=()):
            lines = []
            for fold in range(5):
                lines.extend([
                    'AUC: %.6f' % (aupr + 0.01),
                    'AUPR: %.6f' % (aupr + fold * 0.0001),
                    'Recall: 0.800000',
                    'Precision: 0.810000',
                    'F1-score: 0.805000',
                ])
                if metadata_paths:
                    lines.append(
                        'Herb prototype metadata: %s' % metadata_paths[fold]
                    )
            lines.extend([
                'The result of 5-fold cross validation:',
                'AUC:%.6f(+-0.001000)' % (aupr + 0.01),
                'AUPR:%.6f(+-0.001000)' % aupr,
                'Recall:0.800000(+-0.001000)',
                'Precision:0.810000(+-0.001000)',
                'F1-score:0.805000(+-0.001000)',
            ])
            return '\n'.join(lines) + '\n'

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for index, dataset in enumerate(manifest['datasets']):
                metadata_paths = []
                for fold in range(5):
                    metadata_path = run_dir / (
                        '%s_fold%d.json' % (dataset['slug'], fold + 1)
                    )
                    metadata_path.write_text(json.dumps({
                        'learned_scale': 0.2,
                        'validation': {
                            'evidence_coverage': 0.8,
                            'mean_abs_residual': 0.05,
                        },
                    }), encoding='utf-8')
                    metadata_paths.append(metadata_path)
                (run_dir / ('%02d_%s_baseline.log' % (
                    index * 2 + 1, dataset['slug']
                ))).write_text(log_text(0.900), encoding='utf-8')
                (run_dir / ('%02d_%s_candidate.log' % (
                    index * 2 + 2, dataset['slug']
                ))).write_text(
                    log_text(0.904, metadata_paths), encoding='utf-8'
                )
            report = summarize_full(run_dir, manifest_path)
        self.assertEqual(report['gate'], 'PASS')
        self.assertEqual(report['positive_fold_count'], 20)
        self.assertAlmostEqual(report['mean_outer_aupr_delta'], 0.004)

    def test_five_fold_configs_are_frozen_and_pairwise_isolated(self):
        repository_root = Path(__file__).resolve().parents[1]
        manifest = json.loads((
            repository_root / 'configs/schpt_full_manifest.json'
        ).read_text(encoding='utf-8'))
        prototype_keys = {
            'herb.prototype.transfer',
            'herb.prototype.mode',
            'herb.prototype.prior',
            'herb.prototype.replace.compound.pagerank',
        }
        for dataset in manifest['datasets']:
            with self.subTest(dataset=dataset['name']):
                baseline_path = repository_root / dataset['baseline_config']
                candidate_path = repository_root / dataset['candidate_config']
                self.assertEqual(
                    hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    dataset['baseline_sha256'],
                )
                self.assertEqual(
                    hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                    dataset['candidate_sha256'],
                )
                baseline = ModelConf(str(baseline_path))
                candidate = ModelConf(str(candidate_path))
                baseline_values = dict(baseline.config)
                candidate_values = dict(candidate.config)
                baseline_values.pop('model.variant')
                candidate_values.pop('model.variant')
                for key in prototype_keys:
                    baseline_values.pop(key, None)
                    candidate_values.pop(key, None)
                self.assertEqual(candidate_values, baseline_values)
                self.assertFalse(candidate.contains('evaluation.fold.limit'))
                self.assertEqual(candidate['evaluation.outer.test'], 'True')
                self.assertEqual(candidate['split.seed'], '52026')

    def test_full_summary_supports_direct_script_execution(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(repository_root / 'tools/summarize_schpt_full.py'),
                '--help',
            ],
            cwd=str(repository_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
