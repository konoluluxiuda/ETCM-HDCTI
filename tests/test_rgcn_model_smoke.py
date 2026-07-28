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


class RGCNModelSmokeTest(unittest.TestCase):
    def test_training_graph_uses_only_inner_train_cp_edges(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from RGCNCTI import RGCNCTI

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
                'model.name': 'RGCNCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '2',
                'rgcn.layers': '2',
                'rgcn.activation': 'relu',
                'rgcn.objective': 'bce',
                'pair.decoder': 'dot',
                'weight.reg': '0.01',
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
                model = RGCNCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                self.assertEqual(
                    model.graph_metadata['source_edge_counts']['C_P'], 2
                )
                self.assertEqual(model.n_layers, 2)
                self.assertEqual(len(model.rgcn_weights), 14)
                operation_names = {
                    operation.name
                    for operation in tf.get_default_graph().get_operations()
                }
                self.assertTrue(any(
                    name.startswith(
                        'rgcn_message_herb_to_compound_layer_1'
                    )
                    for name in operation_names
                ))
                self.assertTrue(any(
                    name.startswith(
                        'rgcn_message_compound_to_protein_layer_1'
                    )
                    for name in operation_names
                ))
                self.assertFalse(any(
                    name.startswith('lightgcn_propagation_layer')
                    for name in operation_names
                ))
                trainable_names = {
                    variable.name.split(':')[0]
                    for variable in tf.trainable_variables()
                }
                self.assertIn('rgcn_herb_embeddings', trainable_names)
                self.assertIn('rgcn_disease_embeddings', trainable_names)
                self.assertIn(
                    'rgcn_compound_to_protein_layer_1', trainable_names
                )

                model.trainModel()
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob('rgcn_cti.json')
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                self.assertEqual(
                    metadata['graph_source']['C_P'],
                    'strict_inner_train_positive_C-P',
                )
                self.assertEqual(
                    metadata['source_edge_counts']['C_P'], 2
                )
                self.assertEqual(metadata['relations'], 6)
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
