import json
import unittest
from pathlib import Path

import numpy as np

from tools.train_frozen_base_hctx_router import (
    evaluate_router,
    train_head,
    verify_baseline_metrics,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT / 'configs/frozen_base_hctx_router_pilot.json'
)
FOUR_DATASET_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / 'configs/frozen_base_hctx_router_four_dataset_gate.json'
)
OUTER_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / 'configs/frozen_base_hctx_router_outer_evaluation.json'
)


def state_arrays(base_logits, features, labels):
    return {
        'base_logits': np.asarray(base_logits, dtype=np.float64),
        'hctx_p_features': np.asarray(features, dtype=np.float64),
        'labels': np.asarray(labels, dtype=np.float64),
    }


class FrozenBaseHctxRouterTest(unittest.TestCase):
    def test_wc_and_cc_are_preserved_exactly(self):
        arrays = {
            name: state_arrays(
                [1.0, -1.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0, 0.0],
            )
            for name in (
                'warm_warm', 'cold_warm', 'warm_cold', 'cold_cold'
            )
        }
        preserved = {
            'warm_cold': {'AUPR': 0.61, 'AUC': 0.62, 'records': 2},
            'cold_cold': {'AUPR': 0.51, 'AUC': 0.52, 'records': 2},
        }
        metrics = evaluate_router(
            arrays,
            np.asarray([0.5, -0.5]),
            preserved_metrics=preserved,
        )
        self.assertEqual(metrics['warm_cold'], preserved['warm_cold'])
        self.assertEqual(metrics['cold_cold'], preserved['cold_cold'])
        self.assertNotEqual(metrics['warm_warm'], preserved['warm_cold'])

    def test_baseline_verification_fails_closed(self):
        metrics = {
            name: {'AUPR': 0.6, 'AUC': 0.6}
            for name in (
                'warm_warm', 'cold_warm', 'warm_cold',
                'cold_cold', 'macro',
            )
        }
        reported = json.loads(json.dumps(metrics))
        reported['cold_cold']['AUPR'] = 0.5
        with self.assertRaises(ValueError):
            verify_baseline_metrics(metrics, reported)

    def test_head_training_is_deterministic(self):
        features = np.asarray([
            [2.0, 0.0],
            [1.5, 0.0],
            [-2.0, 0.0],
            [-1.5, 0.0],
        ])
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        validation = {
            name: state_arrays(
                [0.5, 0.5, -0.5, -0.5],
                features,
                labels,
            )
            for name in (
                'warm_warm', 'cold_warm', 'warm_cold', 'cold_cold'
            )
        }
        preserved = {
            'warm_cold': {'AUPR': 1.0, 'AUC': 1.0, 'records': 4},
            'cold_cold': {'AUPR': 1.0, 'AUC': 1.0, 'records': 4},
        }
        settings = {
            'seed': 7,
            'max_epochs': 4,
            'batch_size': 2,
            'learning_rate': 0.01,
            'l2': 0.01,
            'validation_interval': 1,
            'patience': 2,
            'min_delta': 0.0001,
        }
        first = train_head(
            features, labels, validation, preserved, settings
        )
        second = train_head(
            features, labels, validation, preserved, settings
        )
        np.testing.assert_allclose(first['head'], second['head'])
        self.assertEqual(first['best_epoch'], second['best_epoch'])

    def test_frozen_manifest_references_verified_artifacts(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        self.assertEqual(
            manifest['protocol'],
            'frozen_base_hctx_router_pilot_v3',
        )
        self.assertEqual(set(manifest['datasets']), {
            'tcmsp', 'etcm_mention10',
        })
        for spec in manifest['datasets'].values():
            self.assertTrue(
                (REPOSITORY_ROOT / spec['config']).is_file()
            )
            self.assertTrue(spec['baseline_report'].endswith('report.json'))
            self.assertEqual(len(spec['checkpoint_index_sha256']), 64)
            self.assertEqual(len(spec['assignments_sha256']), 64)
            checkpoint_index = REPOSITORY_ROOT / (
                spec['checkpoint'] + '.index'
            )
            if checkpoint_index.is_file():
                from tools.evaluate_four_state_checkpoint import sha256_file
                self.assertEqual(
                    sha256_file(checkpoint_index),
                    spec['checkpoint_index_sha256'],
                )

    def test_four_dataset_gate_preserves_pilot_settings(self):
        pilot = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        gate = json.loads(
            FOUR_DATASET_MANIFEST_PATH.read_text(encoding='utf-8')
        )
        self.assertEqual(gate['protocol'], pilot['protocol'])
        self.assertEqual(gate['head_training'], pilot['head_training'])
        self.assertEqual(
            gate['parent_pilot_manifest_sha256'],
            '620a99851cd1f9be43b2f049b9cec2e21d75cde5344dfd575f6c212e32327026',
        )
        self.assertEqual(set(gate['datasets']), {
            'tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10',
        })
        for dataset in ('tcmsp', 'etcm_mention10'):
            self.assertEqual(
                gate['datasets'][dataset],
                pilot['datasets'][dataset],
            )

    def test_outer_manifest_freezes_all_selected_heads(self):
        outer = json.loads(
            OUTER_MANIFEST_PATH.read_text(encoding='utf-8')
        )
        self.assertEqual(
            outer['protocol'],
            'frozen_base_hctx_router_outer_evaluation_v3',
        )
        self.assertEqual(
            outer['parent_gate_manifest_sha256'],
            '3afdd445f809b7b009ddea83058e55989dbcf000799b89e34cd9df689741c2a6',
        )
        self.assertEqual(set(outer['datasets']), {
            'tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10',
        })
        for spec in outer['datasets'].values():
            self.assertEqual(len(spec['training_report_sha256']), 64)
            self.assertEqual(len(spec['head_sha256']), 64)
            report_path = REPOSITORY_ROOT / spec['training_report']
            head_path = REPOSITORY_ROOT / spec['head']
            if report_path.is_file():
                from tools.evaluate_four_state_checkpoint import sha256_file
                self.assertEqual(
                    sha256_file(report_path),
                    spec['training_report_sha256'],
                )
            if head_path.is_file():
                from tools.evaluate_four_state_checkpoint import sha256_file
                self.assertEqual(
                    sha256_file(head_path),
                    spec['head_sha256'],
                )


if __name__ == '__main__':
    unittest.main()
