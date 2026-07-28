import json
import os
import tempfile
import unittest
from pathlib import Path


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class LightGCNModelSmokeTest(unittest.TestCase):
    def test_training_graph_uses_only_inner_train_positive_pairs(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from LightGCNCTI import LightGCNCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc2\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\n',
                encoding='utf-8',
            )
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'LightGCNCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '2',
                'lightgcn.layers': '2',
                'lightgcn.objective': 'bce',
                'pair.decoder': 'dot',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0], ['c0', 'p1', 0.0],
                ['c1', 'p1', 1.0], ['c1', 'p0', 0.0],
            ]
            test = [['c2', 'p0', 1.0], ['c2', 'p1', 0.0]]
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                model = LightGCNCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                self.assertEqual(model.graph_metadata['edge_count'], 2)
                self.assertEqual(model.n_layers, 2)
                operation_names = {
                    operation.name
                    for operation in tf.get_default_graph().get_operations()
                }
                self.assertTrue(any(
                    name.startswith('lightgcn_propagation_layer_1')
                    for name in operation_names
                ))
                self.assertTrue(any(
                    name.startswith('lightgcn_uniform_layer_mean')
                    for name in operation_names
                ))
                self.assertFalse(any(
                    name.startswith('hc_') or name.startswith('pd_')
                    for name in operation_names
                ))
                self.assertEqual(
                    {
                        variable.name.split(':')[0]
                        for variable in tf.trainable_variables()
                    },
                    {'U', 'V'},
                )
                model.trainModel()
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob(
                        'lightgcn_cti.json'
                    )
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                self.assertEqual(
                    metadata['graph_source'],
                    'strict_inner_train_positive_C-P',
                )
                self.assertEqual(metadata['graph']['edge_count'], 2)
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
