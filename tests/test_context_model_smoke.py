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


class ContextModelSmokeTest(unittest.TestCase):
    def test_context_model_builds_and_optimizes_one_step(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc1\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np0\td1\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
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
                'context.interaction': 'True',
                'attention.max.nodes': '10',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [['c0', 'p0', 1.0], ['c0', 'p1', 0.0]]
            test = [['c1', 'p1', 1.0], ['c1', 'p0', 0.0]]
            model = HDCTI(conf, training, test, '[1]')
            model.readConfiguration()
            model.initModel()
            graph_operations = tf.get_default_graph().get_operations()
            self.assertTrue(any(
                operation.name.startswith('hc_edge_to_node_layer_1')
                and operation.type == 'SparseTensorDenseMatMul'
                for operation in graph_operations
            ))
            logits = model.buildPairLogits()
            labels = model.neg_disease_embedding
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits
            )) + model.buildRegularizationLoss()
            train = tf.train.AdamOptimizer(model.lRate).minimize(loss)
            model.sess.run(tf.global_variables_initializer())
            feed = {
                model.u_idx: [model.data.compound['c0'], model.data.compound['c0']],
                model.v_idx: [model.data.protein['p0'], model.data.protein['p1']],
                model.neg_idx: [1.0, 0.0],
                model.isTraining: 1,
            }
            before = model.sess.run(loss, feed_dict=feed)
            model.sess.run(train, feed_dict=feed)
            after, context_weight = model.sess.run(
                [loss, model.weights['context_compound_disease']], feed_dict=feed
            )
            model.sess.close()

        self.assertTrue(np.isfinite(before))
        self.assertTrue(np.isfinite(after))
        self.assertGreater(float(np.linalg.norm(context_weight)), 0.0)


if __name__ == '__main__':
    unittest.main()
