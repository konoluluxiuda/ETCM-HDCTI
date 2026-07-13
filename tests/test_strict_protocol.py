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
