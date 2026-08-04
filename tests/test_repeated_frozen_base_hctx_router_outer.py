import json
import tempfile
import unittest
from pathlib import Path

from tools.run_frozen_base_hctx_router_repeated_outer import (
    checkpoint_from_training_output,
    command_environment,
    load_prepared_manifest,
)
from tools.summarize_frozen_base_hctx_router_five_units import (
    HISTORICAL_PROTOCOL,
    summarize as summarize_five_units,
)
from tools.summarize_repeated_frozen_base_hctx_router_outer import (
    EXPECTED_PROTOCOL,
    STATE_NAMES,
    summarize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARED_MANIFEST = (
    REPOSITORY_ROOT
    / 'configs'
    / 'frozen_base_hctx_router_repeated_units_manifest.json'
)


def synthetic_plan():
    return {
        'confirmatory_units': [
            {'compound_group': group, 'protein_group': group}
            for group in range(1, 5)
        ],
        'confirmatory_gate': {
            'new_outer_unit_count': 16,
            'overall_mean_macro_aupr_delta_minimum': 0.005,
            'minimum_positive_units': 12,
            'maximum_dataset_mean_state_aupr_drop': 0.02,
        },
        'datasets': {
            key: {'display_name': display}
            for key, display in (
                ('tcmsuite', 'TCM-Suite'),
                ('tcmsp', 'TCMSP'),
                ('symmap', 'SymMap2.0'),
                ('etcm_mention10', 'ETCM2.0-mention10'),
            )
        },
    }


def synthetic_prepared(plan):
    jobs = []
    for dataset, spec in plan['datasets'].items():
        for group in range(1, 5):
            jobs.append({
                'job_key': '%s_c%dp%d' % (dataset, group, group),
                'dataset': dataset,
                'display_name': spec['display_name'],
                'compound_group': group,
                'protein_group': group,
                'assignments_sha256': str(group) * 64,
            })
    return {'jobs': jobs}


def synthetic_report(job, macro_delta=0.01, cold_cold_delta=0.0):
    deltas = {
        'warm_warm': {'AUPR': 0.01, 'AUC': 0.01},
        'cold_warm': {'AUPR': 0.03, 'AUC': 0.03},
        'warm_cold': {'AUPR': 0.0, 'AUC': 0.0},
        'cold_cold': {
            'AUPR': cold_cold_delta,
            'AUC': cold_cold_delta,
        },
        'macro': {'AUPR': macro_delta, 'AUC': macro_delta},
    }
    return {
        'evaluation': 'frozen_base_hctx_router_outer_pure_inference',
        'protocol': EXPECTED_PROTOCOL,
        'job': {'key': job['job_key']},
        'support_unit': {
            'assignments_sha256': job['assignments_sha256'],
        },
        'training_optimizer_steps': 0,
        'parameter_selection_on_outer': False,
        'preservation_checks': {
            'warm_cold_exact': True,
            'cold_cold_exact': True,
            'checkpoint_hashes_unchanged': True,
            'head_hash_unchanged': True,
        },
        'baseline_metrics': {
            'macro': {'AUPR': 0.6, 'AUC': 0.6},
        },
        'metrics': {
            'macro': {
                'AUPR': 0.6 + macro_delta,
                'AUC': 0.6 + macro_delta,
            },
        },
        'comparison': {'deltas': deltas},
    }


class RepeatedFrozenRouterOuterTest(unittest.TestCase):
    def write_reports(self, directory, prepared, **overrides):
        paths = []
        for job in prepared['jobs']:
            values = overrides.get(job['job_key'], {})
            report = synthetic_report(job, **values)
            path = directory / (job['job_key'] + '.json')
            path.write_text(json.dumps(report), encoding='utf-8')
            paths.append(path)
        return paths

    def test_checked_prepared_manifest_has_sixteen_units(self):
        prepared, plan, _ = load_prepared_manifest(PREPARED_MANIFEST)
        self.assertEqual(len(prepared['jobs']), 16)
        self.assertEqual(
            plan['confirmatory_gate']['new_outer_unit_count'], 16
        )

    def test_checkpoint_path_is_parsed_from_training_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / 'hdcti_model.ckpt'
            Path(str(prefix) + '.index').write_bytes(b'index')
            output = '模型权重保存成功: %s\n' % prefix
            self.assertEqual(checkpoint_from_training_output(output), prefix)

    def test_relative_checkpoint_is_resolved_from_repository(self):
        checkpoint_dir = REPOSITORY_ROOT / 'saved_model' / '_parser_test'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = checkpoint_dir / 'hdcti_model.ckpt'
        index_path = Path(str(prefix) + '.index')
        index_path.write_bytes(b'index')
        try:
            output = (
                '模型权重保存成功: '
                './saved_model/_parser_test/hdcti_model.ckpt\n'
            )
            self.assertEqual(
                checkpoint_from_training_output(output), prefix.resolve()
            )
        finally:
            index_path.unlink()
            checkpoint_dir.rmdir()

    def test_cpu_environment_disables_cuda(self):
        environment = command_environment('cpu')
        self.assertEqual(environment['HDCTI_FORCE_CPU'], '1')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '-1')

    def test_preregistered_gate_passes_consistent_positive_units(self):
        plan = synthetic_plan()
        prepared = synthetic_prepared(plan)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.write_reports(Path(temporary), prepared)
            summary = summarize(plan, prepared, paths)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['positive_macro_units'], 16)
        self.assertAlmostEqual(
            summary['overall_mean_macro_aupr_delta'], 0.01
        )

    def test_preregistered_gate_rejects_nonpreserved_cold_cold(self):
        plan = synthetic_plan()
        prepared = synthetic_prepared(plan)
        first = prepared['jobs'][0]['job_key']
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.write_reports(
                Path(temporary), prepared,
                **{first: {'cold_cold_delta': -0.001}},
            )
            summary = summarize(plan, prepared, paths)
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['gate_checks'][
            'warm_cold_and_cold_cold_exact_preservation'
        ])

    def test_duplicate_outer_unit_fails_closed(self):
        plan = synthetic_plan()
        prepared = synthetic_prepared(plan)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.write_reports(Path(temporary), prepared)
            paths[-1] = paths[0]
            with self.assertRaises(ValueError):
                summarize(plan, prepared, paths)

    def test_five_unit_summary_is_descriptive_and_complete(self):
        plan = synthetic_plan()
        prepared = synthetic_prepared(plan)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repeated_paths = self.write_reports(root, prepared)
            repeated = summarize(plan, prepared, repeated_paths)
            historical_rows = []
            for dataset, spec in plan['datasets'].items():
                job = {
                    'job_key': dataset + '_c0p0',
                    'assignments_sha256': '0' * 64,
                }
                report = synthetic_report(job, macro_delta=0.02)
                report_path = root / (dataset + '_c0p0.json')
                report_path.write_text(
                    json.dumps(report), encoding='utf-8'
                )
                historical_rows.append({
                    'dataset': spec['display_name'],
                    'report': str(report_path),
                    'baseline_macro_aupr': 0.6,
                    'candidate_macro_aupr': 0.62,
                    'deltas': {
                        name: report['comparison']['deltas'][name]['AUPR']
                        for name in STATE_NAMES + ('macro',)
                    },
                })
            historical = {
                'protocol': HISTORICAL_PROTOCOL,
                'all_passed': True,
                'datasets': historical_rows,
            }
            historical_path = root / 'historical.json'
            repeated_path = root / 'repeated.json'
            historical_path.write_text(
                json.dumps(historical), encoding='utf-8'
            )
            repeated_path.write_text(
                json.dumps(repeated), encoding='utf-8'
            )
            combined = summarize_five_units(
                historical, repeated, historical_path, repeated_path
            )
        self.assertEqual(combined['unit_count'], 20)
        self.assertEqual(combined['positive_macro_units'], 20)
        self.assertAlmostEqual(
            combined['overall_macro_aupr_delta']['mean'], 0.012
        )
        for row in combined['datasets']:
            self.assertEqual(row['units'], 5)
            self.assertEqual(row['positive_macro_units'], 5)

    def test_state_names_match_four_state_protocol(self):
        self.assertEqual(STATE_NAMES, (
            'warm_warm', 'cold_warm', 'warm_cold', 'cold_cold'
        ))


if __name__ == '__main__':
    unittest.main()
