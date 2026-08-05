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


class HerbPrototypeTransferModelTest(unittest.TestCase):
    def test_tensorflow_and_checkpoint_pair_scores_match(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh0\tc2\nh1\tc3\n',
                encoding='utf-8',
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc3\tp1\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc3\tp1\t1\n',
                encoding='utf-8',
            )
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'HDCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '4',
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
                'context.mask.training': 'False',
                'support.router': 'False',
                'support.experts': 'False',
                'support.state.routing': 'False',
                'hyperedge.attention': 'False',
                'global.token.attention': 'False',
                'hplga.enabled': 'False',
                'inductive.context': 'True',
                'inductive.context.suppress.base.zero.support': 'True',
                'inductive.context.self.excluded': 'False',
                'herb.prototype.transfer': 'True',
                'herb.prototype.mode': 'support_calibrated_loco',
                'herb.prototype.prior': '1.0',
                'herb.prototype.replace.compound.pagerank': 'True',
                'attention.max.nodes': '0',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0], ['c0', 'p1', 0.0],
                ['c1', 'p1', 1.0], ['c1', 'p0', 0.0],
                ['c3', 'p1', 1.0], ['c3', 'p0', 0.0],
            ]
            test = [
                ['c2', 'p0', 1.0], ['c2', 'p1', 0.0],
            ]
            model = HDCTI(conf, training, test, '[1]')
            model.readConfiguration()
            model.initModel()
            model.sess.run(tf.global_variables_initializer())
            model.sess.run(
                model.weights['herb_prototype_scale'].assign([2.0])
            )
            compound_indices = np.asarray([
                model.data.compound['c0'], model.data.compound['c2'],
            ], dtype=np.int32)
            protein_indices = np.asarray([
                model.data.protein['p0'], model.data.protein['p0'],
            ], dtype=np.int32)
            graph_logits = model.sess.run(
                model.buildPairLogits(),
                feed_dict={
                    model.u_idx: compound_indices,
                    model.v_idx: protein_indices,
                    model.isTraining: 0,
                },
            )
            graph_residuals = model.sess.run(
                model.herb_prototype_pair_residual,
                feed_dict={
                    model.u_idx: compound_indices,
                    model.v_idx: protein_indices,
                    model.isTraining: 0,
                },
            )
            numpy_residuals = model.herbPrototypePairResiduals(
                compound_indices, protein_indices
            )
            np.testing.assert_allclose(
                numpy_residuals, graph_residuals, rtol=1e-5, atol=1e-5
            )
            state = model.fetchModelState()
            model.u = state['compound']
            model.i = state['protein']
            model.u_context = state['compound_context']
            model.i_context = state['protein_context']
            model.herb_edge = state['herb_edge']
            model.weight = state['weights']
            numpy_logits = model.predictForPairs(
                compound_indices, protein_indices
            )
            model.sess.close()

        np.testing.assert_allclose(
            numpy_logits, graph_logits, rtol=1e-5, atol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
