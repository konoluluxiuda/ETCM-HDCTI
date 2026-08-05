import json
import tempfile
import unittest
from pathlib import Path

from tools.run_frozen_base_hctx_router_vs_sdis import load_manifest
from tools.summarize_frozen_base_hctx_router_vs_sdis import summarize


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY_ROOT
    / 'configs'
    / 'frozen_base_hctx_router_vs_sdis_units_manifest.json'
)
STATE_NAMES = ('warm_warm', 'cold_warm', 'warm_cold', 'cold_cold')


def synthetic_plan():
    return {
        'protocol': 'frozen_base_hctx_router_vs_sdis_preregistration_v1',
        'decision_gate': {
            'unit_count': 16,
            'noninferiority': {
                'overall_mean_macro_aupr_delta_minimum': -0.005,
                'each_dataset_mean_macro_aupr_delta_minimum': -0.01,
                'maximum_dataset_mean_state_aupr_drop': 0.02,
            },
            'superiority': {
                'overall_mean_macro_aupr_delta_minimum': 0.005,
                'minimum_positive_units': 12,
                'each_dataset_mean_macro_aupr_delta_minimum': 0.0,
            },
        },
        'decision_rule': {
            'superiority_pass': 'superiority',
            'noninferiority_only': 'noninferiority',
            'noninferiority_fail': 'failure',
        },
    }


def synthetic_jobs():
    jobs = []
    for dataset, display in (
        ('tcmsuite', 'TCM-Suite'),
        ('tcmsp', 'TCMSP'),
        ('symmap', 'SymMap2.0'),
        ('etcm_mention10', 'ETCM2.0-mention10'),
    ):
        for group in range(1, 5):
            jobs.append({
                'job_key': '%s_c%dp%d' % (dataset, group, group),
                'dataset': dataset,
                'display_name': display,
                'compound_group': group,
                'protein_group': group,
                'assignments_sha256': ('%02d' % group) * 32,
            })
    return jobs


def state_metrics(value):
    metrics = {
        name: {'AUPR': value, 'AUC': value, 'records': 100}
        for name in STATE_NAMES
    }
    metrics['macro'] = {'AUPR': value, 'AUC': value}
    return metrics


class FrozenBaseRouterVersusSdisTest(unittest.TestCase):
    def build_reports(self, root, delta):
        jobs = synthetic_jobs()
        v3_units = []
        sdis_paths = []
        for job in jobs:
            v3_dir = root / 'v3' / job['job_key']
            sdis_dir = root / 'sdis' / job['job_key']
            v3_dir.mkdir(parents=True)
            sdis_dir.mkdir(parents=True)
            v3_report = {
                'protocol': (
                    'frozen_base_hctx_router_repeated_outer_evaluation_v1'
                ),
                'training_optimizer_steps': 0,
                'parameter_selection_on_outer': False,
                'support_unit': {
                    'assignments_sha256': job['assignments_sha256'],
                },
                'metrics': state_metrics(0.7 + delta),
            }
            sdis_report = {
                'evaluation': 'four_state_checkpoint_pure_inference',
                'records': 'outer',
                'training_optimizer_steps': 0,
                'parameter_selection_on_records': False,
                'support_unit': {
                    'assignments_sha256': job['assignments_sha256'],
                },
                'metrics': state_metrics(0.7),
            }
            v3_path = v3_dir / 'report.json'
            sdis_path = sdis_dir / 'report.json'
            v3_path.write_text(json.dumps(v3_report), encoding='utf-8')
            sdis_path.write_text(json.dumps(sdis_report), encoding='utf-8')
            v3_units.append({
                'job_key': job['job_key'],
                'report': str(v3_path),
            })
            sdis_paths.append(sdis_path)
        v3_summary = {
            'protocol': 'frozen_base_hctx_router_repeated_outer_gate_v1',
            'passed': True,
            'outer_unit_count': 16,
            'units': v3_units,
        }
        v3_summary_path = root / 'v3_summary.json'
        v3_summary_path.write_text(
            json.dumps(v3_summary), encoding='utf-8'
        )
        manifest = {
            'protocol': 'frozen_base_hctx_router_vs_sdis_units_v1',
            'jobs': jobs,
        }
        return manifest, v3_summary_path, sdis_paths

    def test_frozen_manifest_contains_sixteen_matched_sdis_units(self):
        manifest, _, _ = load_manifest(MANIFEST)
        self.assertEqual(len(manifest['jobs']), 16)
        for job in manifest['jobs']:
            text = (REPOSITORY_ROOT / job['config']).read_text(
                encoding='utf-8'
            )
            self.assertIn('context.herb_protein=True\n', text)
            self.assertIn('inductive.context=True\n', text)
            self.assertIn(
                'inductive.context.suppress.base.zero.support=True\n', text
            )
            self.assertIn('attention.max.nodes=0\n', text)

    def test_superiority_gate_passes_consistent_positive_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, v3_path, reports = self.build_reports(
                Path(temporary), 0.01
            )
            result = summarize(
                synthetic_plan(), manifest, v3_path, reports
            )
        self.assertTrue(result['noninferiority_passed'])
        self.assertTrue(result['superiority_passed'])
        self.assertEqual(result['decision'], 'superiority_pass')
        self.assertEqual(result['positive_units'], 16)

    def test_zero_delta_is_noninferior_but_not_superior(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, v3_path, reports = self.build_reports(
                Path(temporary), 0.0
            )
            result = summarize(
                synthetic_plan(), manifest, v3_path, reports
            )
        self.assertTrue(result['noninferiority_passed'])
        self.assertFalse(result['superiority_passed'])
        self.assertEqual(result['decision'], 'noninferiority_only')

    def test_material_drop_fails_noninferiority(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, v3_path, reports = self.build_reports(
                Path(temporary), -0.03
            )
            result = summarize(
                synthetic_plan(), manifest, v3_path, reports
            )
        self.assertFalse(result['noninferiority_passed'])
        self.assertEqual(result['decision'], 'noninferiority_fail')

    def test_sdis_outer_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, v3_path, reports = self.build_reports(root, 0.01)
            report = json.loads(reports[0].read_text(encoding='utf-8'))
            report['training_optimizer_steps'] = 1
            reports[0].write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                summarize(synthetic_plan(), manifest, v3_path, reports)


if __name__ == '__main__':
    unittest.main()
