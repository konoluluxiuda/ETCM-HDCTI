import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class HyperedgeAttentionModelSmokeTest(unittest.TestCase):
    def test_factorized_attention_trains_and_writes_metadata(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc1\nh1\tc2\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np0\td1\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\n', encoding='utf-8'
            )
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'HDCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '2',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'pair.decoder': 'dot',
                'context.interaction': 'True',
                'context.compound_disease': 'False',
                'context.herb_protein': 'True',
                'context.herb_protein.mode': 'static',
                'context.herb_disease': 'False',
                'counterfactual.context': 'False',
                'hyperedge.attention': 'True',
                'hyperedge.attention.mode': 'factorized_specificity',
                'hyperedge.attention.hc': 'True',
                'hyperedge.attention.pd': 'True',
                'hyperedge.attention.temperature': '1.0',
                'hyperedge.attention.prior.scale': '0.1',
                'attention.max.nodes': '0',
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
                model = HDCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                operation_names = {
                    operation.name for operation in tf.get_default_graph().get_operations()
                }
                self.assertTrue(any(
                    name.startswith('hc_factorized_attention_layer_1_node_to_edge')
                    for name in operation_names
                ))
                self.assertTrue(any(
                    name.startswith('pd_factorized_attention_layer_1_edge_to_node')
                    for name in operation_names
                ))
                model.trainModel()

                learned_parameters = [
                    model.weight[name]
                    for name in model.weight
                    if '_hyper_node_' in name or '_hyper_edge_' in name
                ]
                self.assertTrue(any(
                    float(np.linalg.norm(value)) > 0.0
                    for value in learned_parameters
                ))
                self.assertIsNotNone(model.hyperedge_attention_summary)
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob('hyperedge_attention.json')
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                self.assertTrue(metadata['hc_enabled'])
                self.assertTrue(metadata['pd_enabled'])
                self.assertEqual(metadata['structure']['hc']['incidences'], 4)
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
