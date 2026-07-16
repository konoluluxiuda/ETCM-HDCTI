import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from rating import Rating
from util.dataSplit import DataSplit
from util.graph import bipartite_pagerank


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class StrictProtocolTest(unittest.TestCase):
    def test_inner_validation_split_is_deterministic_balanced_and_disjoint(self):
        records = []
        for index in range(10):
            records.append(['c%d' % index, 'p0', 1.0])
            records.append(['c%d' % index, 'p1', 0.0])

        first_train, first_validation, first_info = DataSplit.innerValidationSplit(
            records, ratio=0.2, seed=41
        )
        second_train, second_validation, second_info = DataSplit.innerValidationSplit(
            list(reversed(records)), ratio=0.2, seed=41
        )

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(first_info, second_info)
        self.assertEqual(len(first_train), 16)
        self.assertEqual(len(first_validation), 4)
        self.assertEqual(sum(row[2] > 0 for row in first_validation), 2)
        self.assertEqual(sum(row[2] == 0 for row in first_validation), 2)
        self.assertFalse(
            {(row[0], row[1]) for row in first_train}
            & {(row[0], row[1]) for row in first_validation}
        )

    def test_compound_inner_validation_is_deterministic_and_compound_disjoint(self):
        records = []
        for index in range(10):
            records.append(['c%d' % index, 'p0', 1.0])
            records.append(['c%d' % index, 'p1', 0.0])

        first_train, first_validation, first_info = (
            DataSplit.innerCompoundValidationSplit(records, ratio=0.2, seed=41)
        )
        second_train, second_validation, second_info = (
            DataSplit.innerCompoundValidationSplit(
                list(reversed(records)), ratio=0.2, seed=41
            )
        )

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info['strategy'], 'compound_cold_start')
        self.assertEqual(first_info['inner_train_compounds'], 8)
        self.assertEqual(first_info['validation_compounds'], 2)
        self.assertEqual(len(first_train), 16)
        self.assertEqual(len(first_validation), 4)
        self.assertFalse(
            {row[0] for row in first_train} & {row[0] for row in first_validation}
        )
        self.assertEqual(first_info['class_counts']['validation']['0'], 2)
        self.assertEqual(first_info['class_counts']['validation']['1'], 2)

    def test_mixed_training_negatives_are_deterministic_and_reserved_pair_safe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc2\nh1\tc3\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td0\np2\td1\np3\td1\n', encoding='utf-8'
            )
            records = [
                ['c0', 'p0', 1.0], ['c1', 'p1', 1.0],
                ['c2', 'p2', 1.0], ['c3', 'p3', 1.0],
                ['c0', 'p3', 0.0], ['c1', 'p2', 0.0],
                ['c2', 'p1', 0.0], ['c3', 'p0', 0.0],
            ]
            settings = {'strategy': 'mixed', 'hard_ratio': 0.5}
            reserved_pairs = {('c0', 'p1')}

            first, first_info = DataSplit.applyTrainingNegativeStrategy(
                records,
                settings,
                dataset_dir,
                reserved_pairs=reserved_pairs,
                seed=91,
                fold_index=0,
                manifest_dir=dataset_dir / 'manifests',
            )
            second, second_info = DataSplit.applyTrainingNegativeStrategy(
                list(reversed(records)),
                settings,
                dataset_dir,
                reserved_pairs=reserved_pairs,
                seed=91,
                fold_index=0,
                manifest_dir=dataset_dir / 'manifests_second',
            )

            self.assertEqual(first, second)
            self.assertEqual(first_info['assignments_sha256'], second_info['assignments_sha256'])
            self.assertEqual(first_info['hard_negative_count'], 2)
            self.assertEqual(first_info['random_negative_count'], 2)
            self.assertAlmostEqual(first_info['hard_ratio_actual'], 0.5)
            self.assertNotIn(('c0', 'p1'), {(row[0], row[1]) for row in first})
            self.assertEqual(sum(row[2] > 0 for row in first), 4)
            self.assertEqual(sum(row[2] == 0 for row in first), 4)
            self.assertTrue(Path(first_info['assignments_path']).exists())
            self.assertTrue(Path(first_info['manifest_path']).exists())
            assignment_text = Path(first_info['assignments_path']).read_text(encoding='utf-8')
            self.assertIn('hard_', assignment_text)

    def test_random_training_negative_strategy_preserves_records(self):
        records = [['c0', 'p0', 1.0], ['c0', 'p1', 0.0]]
        transformed, info = DataSplit.applyTrainingNegativeStrategy(
            records,
            {'strategy': 'random', 'hard_ratio': 0.25},
            '.',
            seed=7,
        )
        self.assertEqual(transformed, records)
        self.assertEqual(info['strategy'], 'random')

    def test_strict_split_is_reused_and_balanced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            positives = [
                'c0\tp0\t1', 'c0\tp1\t1', 'c1\tp0\t1',
                'c1\tp2\t1', 'c2\tp1\t1', 'c2\tp2\t1',
            ]
            negatives = [
                'c0\tp2\t0', 'c1\tp1\t0', 'c2\tp0\t0',
                'c3\tp0\t0', 'c3\tp1\t0', 'c3\tp2\t0',
            ]
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('\n'.join(positives) + '\n', encoding='utf-8')
            (dataset_dir / 'ZERO_indices.txt').write_text(
                '\n'.join(negatives) + '\n', encoding='utf-8'
            )
            split_dir = dataset_dir / 'strict_split'
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'random.seed': '17',
                'split.dir': str(split_dir),
                'split.reuse': 'True',
            })

            first_folds, first_manifest = DataSplit.prepareStrictFolds(conf, str(datapath), 3)
            first_assignments = (split_dir / 'fold_assignments.tsv').read_bytes()
            second_folds, second_manifest = DataSplit.prepareStrictFolds(conf, str(datapath), 3)

            self.assertEqual(first_assignments, (split_dir / 'fold_assignments.tsv').read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_folds, second_folds)
            self.assertEqual(first_manifest['positive_count'], 6)
            self.assertEqual(first_manifest['negative_count'], 6)
            for train, test in first_folds:
                self.assertEqual(sum(row[2] > 0 for row in test), 2)
                self.assertEqual(sum(row[2] == 0 for row in test), 2)
                self.assertFalse(
                    {(row[0], row[1]) for row in train}
                    & {(row[0], row[1]) for row in test}
                )

            stored_manifest = json.loads((split_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(stored_manifest['strict_guarantees']['training_graph_must_use_fold_training_positives'])
            self.assertFalse(stored_manifest['strict_guarantees']['fixed_hd_side_information'])

            conf.config['split.reuse'] = 'False'
            rebuilt_folds, rebuilt_manifest = DataSplit.prepareStrictFolds(conf, str(datapath), 3)
            self.assertEqual(first_assignments, (split_dir / 'fold_assignments.tsv').read_bytes())
            self.assertEqual(first_folds, rebuilt_folds)
            self.assertEqual(first_manifest['assignments_sha256'], rebuilt_manifest['assignments_sha256'])

    def test_compound_cold_start_split_is_matched_reused_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            positive_rows = []
            negative_rows = []
            for compound_index in range(6):
                compound_id = 'c%d' % compound_index
                first = compound_index % 4
                second = (compound_index + 1) % 4
                positive_rows.extend([
                    '%s\tp%d\t1' % (compound_id, first),
                    '%s\tp%d\t1' % (compound_id, second),
                ])
                negative_rows.append(
                    '%s\tp%d\t0' % (compound_id, (compound_index + 2) % 4)
                )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('\n'.join(positive_rows) + '\n', encoding='utf-8')
            (dataset_dir / 'ZERO_indices.txt').write_text(
                '\n'.join(negative_rows) + '\n', encoding='utf-8'
            )
            split_dir = dataset_dir / 'cold_split'
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'split.strategy': 'compound_cold_start',
                'split.seed': '17',
                'split.dir': str(split_dir),
                'split.reuse': 'True',
            })

            first_folds, first_manifest = DataSplit.prepareStrictFolds(
                conf, str(datapath), 3
            )
            first_assignments = (split_dir / 'fold_assignments.tsv').read_bytes()
            second_folds, second_manifest = DataSplit.prepareStrictFolds(
                conf, str(datapath), 3
            )

            self.assertEqual(first_folds, second_folds)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                first_assignments,
                (split_dir / 'fold_assignments.tsv').read_bytes(),
            )
            self.assertEqual(first_manifest['split_strategy'], 'compound_cold_start')
            self.assertEqual(
                first_manifest['split_algorithm'],
                'compound_group_greedy_balance_v1',
            )
            self.assertTrue(
                first_manifest['strict_guarantees']['compound_disjoint_train_test']
            )
            self.assertEqual(first_manifest['negative_fallback_compounds'], 6)
            self.assertEqual(first_manifest['negative_fallback_records'], 6)
            for train, test in first_folds:
                self.assertFalse(
                    {row[0] for row in train} & {row[0] for row in test}
                )
                self.assertEqual(sum(row[2] > 0 for row in test), 4)
                self.assertEqual(sum(row[2] == 0 for row in test), 4)
                for compound_id in {row[0] for row in test}:
                    compound_records = [row for row in test if row[0] == compound_id]
                    self.assertEqual(
                        sum(row[2] > 0 for row in compound_records),
                        sum(row[2] == 0 for row in compound_records),
                    )

    def test_strict_manifest_rejects_changed_split_strategy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\nc3\tp1\t1\n',
                encoding='utf-8',
            )
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'split.seed': '17',
                'split.dir': str(dataset_dir / 'strict_split'),
                'split.reuse': 'True',
            })
            DataSplit.prepareStrictFolds(conf, str(datapath), 2)
            conf.config['split.strategy'] = 'compound_cold_start'

            with self.assertRaisesRegex(ValueError, 'split strategy'):
                DataSplit.prepareStrictFolds(conf, str(datapath), 2)

    def test_split_seed_is_independent_from_training_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\nc3\tp1\t1\n',
                encoding='utf-8',
            )
            split_dir = dataset_dir / 'strict_split'
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'split.seed': '17',
                'random.seed': '2026',
                'split.dir': str(split_dir),
                'split.reuse': 'True',
            })

            first_folds, first_manifest = DataSplit.prepareStrictFolds(
                conf, str(datapath), 2
            )
            first_assignments = (split_dir / 'fold_assignments.tsv').read_bytes()
            conf.config['random.seed'] = '2027'
            second_folds, second_manifest = DataSplit.prepareStrictFolds(
                conf, str(datapath), 2
            )

            self.assertEqual(first_manifest['seed'], 17)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_folds, second_folds)
            self.assertEqual(
                first_assignments,
                (split_dir / 'fold_assignments.tsv').read_bytes(),
            )

            conf.config['split.seed'] = '18'
            with self.assertRaisesRegex(ValueError, 'seed'):
                DataSplit.prepareStrictFolds(conf, str(datapath), 2)

    def test_strict_manifest_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'random.seed': '3',
                'split.dir': str(dataset_dir / 'split'),
                'split.reuse': 'True',
            })
            DataSplit.prepareStrictFolds(conf, str(datapath), 2)
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\nc2\tp2\t1\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'source files or hashes'):
                DataSplit.prepareStrictFolds(conf, str(datapath), 2)

    def test_strict_manifest_rejects_changed_assignments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
            split_dir = dataset_dir / 'split'
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'random.seed': '3',
                'split.dir': str(split_dir),
                'split.reuse': 'True',
            })
            DataSplit.prepareStrictFolds(conf, str(datapath), 2)
            assignments_path = split_dir / 'fold_assignments.tsv'
            assignments_path.write_text(
                assignments_path.read_text(encoding='utf-8') + 'tampered\n',
                encoding='utf-8',
            )

            with self.assertRaisesRegex(ValueError, 'assignments hash'):
                DataSplit.prepareStrictFolds(conf, str(datapath), 2)

    def test_strict_manifest_tracks_side_relations_but_not_hd(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
            (dataset_dir / 'H_C.txt').write_text('h0\tc0\nh1\tc1\n', encoding='utf-8')
            (dataset_dir / 'C_P.txt').write_text('c0\tp0\nc1\tp1\n', encoding='utf-8')
            (dataset_dir / 'P_D.txt').write_text('p0\td0\np1\td1\n', encoding='utf-8')
            hd_path = dataset_dir / 'H_D.txt'
            hd_path.write_text('h0\td0\n', encoding='utf-8')
            conf = DummyConf({
                'ratings.setup': '-columns 0 1 2',
                'random.seed': '3',
                'split.dir': str(dataset_dir / 'split'),
                'split.reuse': 'True',
            })
            _, manifest = DataSplit.prepareStrictFolds(conf, str(datapath), 2)
            self.assertNotIn('H_D', manifest['sources']['side_relations'])

            hd_path.write_text('h0\td0\nh1\td1\n', encoding='utf-8')
            DataSplit.prepareStrictFolds(conf, str(datapath), 2)

            (dataset_dir / 'H_C.txt').write_text('h0\tc0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source files or hashes'):
                DataSplit.prepareStrictFolds(conf, str(datapath), 2)

    def test_rating_uses_only_fold_training_positives(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text('h0\tc0\n h1\tc1\n', encoding='utf-8')
            (dataset_dir / 'C_P.txt').write_text('c0\tp0\nc1\tp1\n', encoding='utf-8')
            (dataset_dir / 'P_D.txt').write_text('p0\td0\np1\td1\n', encoding='utf-8')
            (dataset_dir / 'H_D.txt').write_text('h0\td0\nh1\td1\n', encoding='utf-8')
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
            conf = DummyConf({
                'datapath': str(datapath),
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
            })
            training = [['c0', 'p0', 1.0], ['c0', 'p1', 0.0]]
            test = [['c1', 'p1', 1.0], ['c1', 'p0', 0.0]]

            data = Rating(conf, training, test)

            self.assertEqual(data.cpassociation, [['c0', 'p0', 1.0]])
            self.assertEqual(len(data.full_cpassociation), 2)
            self.assertEqual(data.hdassociation, [])

    def test_bipartite_pagerank_keeps_entity_types_separate(self):
        adjacency = coo_matrix(
            (np.asarray([1.0]), (np.asarray([0]), np.asarray([0]))),
            shape=(2, 2),
            dtype=np.float32,
        )

        left, right = bipartite_pagerank(adjacency)

        self.assertEqual(left.shape, (2,))
        self.assertEqual(right.shape, (2,))
        self.assertTrue(np.all(left > 0))
        self.assertTrue(np.all(right > 0))
        self.assertAlmostEqual(float(left.sum() + right.sum()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
