import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_four_state_routing_gate import (
    STATE_NAMES,
    markdown_summary,
    summarize_reports,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PAIRS = (
    (
        'TCMSP',
        'configs/HDCTI_tcmsp_four_state_no_context_unit_pilot.conf',
        'configs/HDCTI_tcmsp_four_state_isolated_routing_unit_pilot.conf',
    ),
    (
        'SymMap2.0',
        'configs/HDCTI_symmap_four_state_no_context_unit_pilot.conf',
        'configs/HDCTI_symmap_four_state_isolated_routing_unit_pilot.conf',
    ),
    (
        'ETCM2.0-mention10',
        'configs/HDCTI_etcm_mention10_four_state_no_context_unit_pilot.conf',
        'configs/HDCTI_etcm_mention10_four_state_isolated_routing_unit_pilot.conf',
    ),
)
FROZEN_KEYS = (
    'datapath',
    'evaluation.setup',
    'evaluation.outer.test',
    'experiment.protocol',
    'support.four.state.manifest',
    'random.seed',
    'validation.ratio',
    'validation.seed',
    'validation.metric',
    'validation.interval',
    'validation.patience',
    'validation.min.delta',
    'negative.strategy',
    'pair.decoder',
    'weight.reg',
    'num.factors',
    'num.max.epoch',
    'batch_size',
    'attention.max.nodes',
    'learnRate',
    'reg.lambda',
)


def read_config(relative_path):
    values = {}
    path = REPOSITORY_ROOT / relative_path
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith(('#', ';')):
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip()
    return values


def write_candidate_report(path, macro_delta, passed=True):
    candidate_macro = 0.70
    state_deltas = {
        'warm_warm': 0.01,
        'cold_warm': 0.02,
        'warm_cold': 0.00,
        'cold_cold': 0.03,
    }
    report = {
        'evaluation': 'four_state_checkpoint_pure_inference',
        'support_unit': {'assignments_sha256': 'abc123'},
        'metrics': {
            name: {'AUPR': 0.65, 'AUC': 0.64, 'records': 20}
            for name in STATE_NAMES
        },
        'comparison': {
            'deltas': {
                name: {'AUPR': state_deltas[name], 'AUC': 0.01}
                for name in STATE_NAMES
            },
            'gate_checks': {
                'macro_aupr_delta_at_least_0.005': passed,
                'cold_cold_aupr_not_lower': True,
                'no_state_aupr_drop_over_0.020': True,
            },
            'passed': passed,
        },
    }
    report['metrics']['macro'] = {
        'AUPR': candidate_macro,
        'AUC': 0.64,
    }
    report['comparison']['deltas']['macro'] = {
        'AUPR': macro_delta,
        'AUC': 0.01,
    }
    path.write_text(json.dumps(report), encoding='utf-8')


class FourStateRoutingGateConfigTest(unittest.TestCase):
    def test_cross_dataset_configs_match_frozen_tcmsuite_protocol(self):
        excluded = {
            'datapath',
            'model.variant',
            'support.four.state.manifest',
        }
        reference_baseline = read_config(
            'configs/HDCTI_tcmsuite_four_state_no_context_unit_pilot.conf'
        )
        reference_candidate = read_config(
            'configs/'
            'HDCTI_tcmsuite_four_state_isolated_routing_unit_pilot.conf'
        )
        expected_baseline = {
            key: value for key, value in reference_baseline.items()
            if key not in excluded
        }
        expected_candidate = {
            key: value for key, value in reference_candidate.items()
            if key not in excluded
        }
        for dataset, baseline_path, candidate_path in CONFIG_PAIRS:
            baseline = read_config(baseline_path)
            candidate = read_config(candidate_path)
            self.assertEqual(
                {
                    key: value for key, value in baseline.items()
                    if key not in excluded
                },
                expected_baseline,
                msg='%s baseline drifted from TCM-Suite.' % dataset,
            )
            self.assertEqual(
                {
                    key: value for key, value in candidate.items()
                    if key not in excluded
                },
                expected_candidate,
                msg='%s V2 drifted from TCM-Suite.' % dataset,
            )

    def test_pairs_only_differ_by_frozen_routing_treatment(self):
        variants = set()
        for dataset, baseline_path, candidate_path in CONFIG_PAIRS:
            baseline = read_config(baseline_path)
            candidate = read_config(candidate_path)
            for key in FROZEN_KEYS:
                self.assertEqual(
                    baseline[key],
                    candidate[key],
                    msg='%s differs on frozen key %s' % (dataset, key),
                )
            variants.update((
                baseline['model.variant'],
                candidate['model.variant'],
            ))

            self.assertEqual(baseline['context.interaction'], 'False')
            self.assertEqual(baseline['context.herb_protein'], 'False')
            self.assertEqual(baseline['context.herb_disease'], 'False')
            self.assertEqual(baseline['support.state.routing'], 'False')

            self.assertEqual(candidate['context.interaction'], 'True')
            self.assertEqual(candidate['context.herb_protein'], 'True')
            self.assertEqual(candidate['context.herb_disease'], 'True')
            self.assertEqual(candidate['support.state.routing'], 'True')
            self.assertEqual(
                candidate['support.state.routing.training'],
                'isolated_heads',
            )
            self.assertEqual(
                candidate[
                    'support.state.routing.detach.cold_cold.features'
                ],
                'True',
            )
            for key in (
                'counterfactual.context',
                'context.mask.training',
                'support.router',
                'support.experts',
                'hyperedge.attention',
                'global.token.attention',
                'hplga.enabled',
                'inductive.context',
            ):
                self.assertEqual(baseline[key], 'False')
                self.assertEqual(candidate[key], 'False')
        self.assertEqual(len(variants), len(CONFIG_PAIRS) * 2)

    def test_frozen_manifests_exist_and_have_balanced_states(self):
        for _, baseline_path, _ in CONFIG_PAIRS:
            config = read_config(baseline_path)
            manifest_path = (
                REPOSITORY_ROOT / config['support.four.state.manifest']
            ).resolve()
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(
                manifest_path.read_text(encoding='utf-8')
            )
            self.assertEqual(
                manifest['protocol'],
                'support_complete_four_state',
            )
            for state_name in STATE_NAMES:
                state = manifest['metadata']['states'][state_name]
                self.assertGreater(state['positive_count'], 0)
                self.assertEqual(
                    state['positive_count'],
                    state['negative_count'],
                )

    def test_summary_reconstructs_baseline_and_all_pass_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / 'first.json'
            second = Path(temp_dir) / 'second.json'
            write_candidate_report(first, 0.10, passed=True)
            write_candidate_report(second, 0.05, passed=True)
            summary = summarize_reports([
                'First=%s' % first,
                'Second=%s' % second,
            ])
        self.assertTrue(summary['all_passed'])
        self.assertAlmostEqual(summary['mean_macro_aupr_delta'], 0.075)
        self.assertAlmostEqual(
            summary['datasets'][0]['baseline_macro_aupr'],
            0.60,
        )
        self.assertIn('First', markdown_summary(summary))

    def test_summary_fails_closed_on_a_dataset_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / 'failed.json'
            write_candidate_report(report_path, -0.01, passed=False)
            summary = summarize_reports(['Failed=%s' % report_path])
        self.assertFalse(summary['all_passed'])


if __name__ == '__main__':
    unittest.main()
