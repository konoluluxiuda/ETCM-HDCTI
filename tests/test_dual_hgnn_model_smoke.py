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


class DualHGNNModelSmokeTest(unittest.TestCase):
    def test_frozen_profile_uses_only_dual_hypergraph_encoder(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc1\nh1\tc2\n',
                encoding='utf-8',
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\n',
                encoding='utf-8',
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np0\td1\np1\td1\n',
                encoding='utf-8',
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
                'model.name': 'HDCTI',
                'encoder.profile': 'dual_hgnn_cti',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '2',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'pair.decoder': 'dot',
                'context.interaction': 'False',
                'context.herb_protein.mode': 'static',
                'counterfactual.context': 'False',
                'hyperedge.attention': 'False',
                'global.token.attention': 'False',
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

                self.assertEqual(
                    model.encoder_profile['name'], 'dual_hgnn_cti'
                )
                self.assertEqual(model.attention_weights, {})
                self.assertFalse(any(
                    name.startswith('gating')
                    for name in model.weights
                ))
                self.assertFalse(any(
                    name.startswith('layer_att_')
                    or name.startswith('layer_2_')
                    for name in model.weights
                ))
                operation_names = {
                    operation.name
                    for operation in tf.get_default_graph().get_operations()
                }
                self.assertTrue(any(
                    name.startswith('hc_node_to_edge_layer_1')
                    for name in operation_names
                ))
                self.assertTrue(any(
                    name.startswith('pd_edge_to_node_layer_1')
                    for name in operation_names
                ))
                model.trainModel()
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
